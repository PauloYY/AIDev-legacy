import logging
from typing import Any

from app.config import Config
from app.llm.client import LLMClient
from app.llm.json_extraction import parse_json_object
from app.llm.models import Message

logger = logging.getLogger(__name__)


class FinalVerificationResult:
    UNAVAILABLE = "unavailable"
    OK = "ok"
    PROBLEMS_FOUND = "problems_found"

    def __init__(self, status: str, report: str = ""):
        self.status = status
        self.report = report

    @property
    def is_ok(self) -> bool:
        return self.status == self.OK

    @property
    def is_unavailable(self) -> bool:
        return self.status == self.UNAVAILABLE

    @property
    def is_problems(self) -> bool:
        return self.status == self.PROBLEMS_FOUND


class FinalVerification:
    """Verificação final do projeto antes de permitir o finish.

    Rodada depois que check_project (sintaxe/compilação) e
    ErrorChecklist (falhas de teste/build) estão limpos. Esta camada
    analisa SEMÂNTICAMENTE se o objetivo foi realmente atendido,
    conferindo integração entre arquivos, imports, símbolos e se o
    código escrito corresponde ao que o objetivo pedia.

    Se a LLM falhar, o status fica como "unavailable" — isso impede
    o finish, forçando o agente a tentar novamente ou a prosseguir
    com um aviso explícito.
    """

    MAX_CONTEXT_CHARS = 5000
    MAX_FILES_FOR_ANALYSIS = 20

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(
        self,
        objective: str,
        project_name: str,
        summary: str,
        tools_execute,
    ) -> FinalVerificationResult:
        """Roda a verificação final.

        Coleta informações reais do projeto (arquivos, conteúdo de
        arquivos relevantes, símbolos exportados) e pede à LLM para
        comparar o estado atual com o objetivo.

        Retorna FinalVerificationResult com status e relatório.
        """

        project_info = self._gather_project_info(
            project_name, tools_execute
        )

        if project_info is None:
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        prompt = self._build_prompt(objective, summary, project_info)

        try:
            response = self.llm.generate(
                messages=[Message(role="user", content=prompt)],
                component="FinalVerification",
            )
        except Exception as error:
            logger.warning(
                "Falha ao rodar verificação final (LLM): %s", error
            )
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        if response.content is None:
            logger.warning("Verificação final: LLM retornou conteúdo nulo.")
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        try:
            data = parse_json_object(response.content)
        except Exception as error:
            logger.warning(
                "Verificação final: resposta JSON inválida da LLM: %s", error
            )
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        problems = data.get("problems")
        passed = data.get("passed", False)

        if not isinstance(problems, list):
            logger.warning(
                "Verificação final: campo 'problems' ausente ou inválido na resposta da LLM."
            )
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        if not isinstance(passed, bool):
            logger.warning(
                "Verificação final: campo 'passed' ausente ou inválido na resposta da LLM."
            )
            return FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)

        cleaned = [str(p).strip() for p in problems if str(p).strip()]

        if passed and not cleaned:
            return FinalVerificationResult(FinalVerificationResult.OK)

        report_lines = [
            "FINAL PROJECT VERIFICATION:",
            "Problems found that must be fixed before finishing:",
        ]

        for i, problem in enumerate(cleaned, start=1):
            report_lines.append(f"- {i}. {problem}")

        if not passed and not cleaned:
            report_lines.insert(
                2,
                "(The LLM judged that the objective was not fully met, "
                "but did not specify detailed problems.)",
            )

        return FinalVerificationResult(
            FinalVerificationResult.PROBLEMS_FOUND,
            "\n".join(report_lines),
        )

    def _gather_project_info(
        self,
        project_name: str,
        tools_execute,
    ) -> dict[str, Any] | None:
        """Coleta informações reais do projeto usando as tools existentes."""

        try:
            files_result = tools_execute(
                "list_files", {"project_name": project_name}
            )
        except Exception as error:
            logger.warning(
                "Verificação final: não foi possível listar arquivos do projeto: %s",
                error,
            )
            return None

        if not files_result:
            return {
                "files": [],
                "file_contents": {},
                "symbols": {},
            }

        files = sorted(files_result)

        limited_files = files[: self.MAX_FILES_FOR_ANALYSIS]
        omitted_count = len(files) - len(limited_files)

        file_contents: dict[str, str] = {}
        symbols: dict[str, list[str]] = {}

        use_parallel = (
            len(limited_files) > 1 and Config.parallel_tools
        )
        if use_parallel:
            try:
                from app.agent.parallel import run_concurrent

                batch, _wall_ms = run_concurrent(
                    [
                        (
                            lambda path=file_path: self._read_one_file(
                                project_name, path, tools_execute
                            )
                        )
                        for file_path in limited_files
                    ]
                )
                gathered = [
                    item.value if item.success else (None, [])
                    for item in batch
                ]
            except Exception as error:
                logger.warning(
                    "Verificação final: batch paralelo falhou "
                    "(usando sequencial): %s",
                    error,
                )
                use_parallel = False

        if not use_parallel:
            gathered = [
                self._read_one_file(
                    project_name, file_path, tools_execute
                )
                for file_path in limited_files
            ]

        for file_path, (content, file_symbols) in zip(
            limited_files, gathered
        ):
            if content:
                file_contents[file_path] = content
            if file_symbols:
                symbols[file_path] = file_symbols

        return {
            "files": files,
            "files_shown": len(limited_files),
            "files_omitted": omitted_count,
            "file_contents": file_contents,
            "symbols": symbols,
        }

    def _read_one_file(
        self,
        project_name: str,
        file_path: str,
        tools_execute,
    ) -> tuple[str | None, list[str]]:
        """Lê conteúdo (+símbolos p/.py) de UM arquivo.

        Mesma lógica do loop sequencial legado, extraída para reúso
        pelo batch paralelo. Falhas por arquivo retornam
        (None, []) como antes (erros são ignorados por arquivo).
        """

        content: str | None = None
        file_symbols: list[str] = []

        try:
            content_result = tools_execute(
                "read_file",
                {"project_name": project_name, "file_path": file_path},
            )
            if content_result:
                truncated = str(content_result)
                if len(truncated) > self.MAX_CONTEXT_CHARS:
                    truncated = (
                        f"{truncated[:self.MAX_CONTEXT_CHARS]}\n"
                        f"...[truncated, {len(str(content_result)) - self.MAX_CONTEXT_CHARS} characters omitted]"
                    )
                content = truncated
        except Exception:
            pass

        if file_path.endswith(".py"):
            try:
                sym_result = tools_execute(
                    "list_symbols",
                    {
                        "project_name": project_name,
                        "file_path": file_path,
                    },
                )
                if sym_result:
                    file_symbols = [
                        line.strip()
                        for line in str(sym_result).splitlines()
                        if line.strip()
                    ]
            except Exception:
                pass

        return content, file_symbols

    def detect_unnecessary_files(
        self,
        project_name: str,
        tools_execute,
    ):
        """Detecta arquivos potencialmente desnecessários (determinístico).

        Dimensão "arquivos desnecessários" da verificação final:
        coleta evidências (referências/imports/configuração/
        entrypoints/testes) e classifica cada candidato como
        SAFE / UNCERTAIN / KEEP, de forma conservadora. Não chama a
        LLM e nunca levanta (falha => relatório vazio).
        """
        from app.agent.context.unnecessary_files import (
            analyze_unnecessary_files,
        )

        return analyze_unnecessary_files(project_name, tools_execute)

    def _build_prompt(
        self,
        objective: str,
        summary: str,
        project_info: dict[str, Any],
    ) -> str:
        files_section = ""
        if project_info.get("file_contents"):
            parts = []
            for path, content in sorted(project_info["file_contents"].items()):
                parts.append(f"--- {path} ---\n{content}")
            files_section = "\n\n".join(parts)
        else:
            files_section = "(no files found)"

        symbols_section = ""
        if project_info.get("symbols"):
            parts = []
            for path, syms in sorted(project_info["symbols"].items()):
                parts.append(f"## {path}:\n" + "\n".join(f"- {s}" for s in syms))
            symbols_section = "\n\n".join(parts)
        else:
            symbols_section = "(no symbols detected)"

        return f"""
You are a software quality reviewer in charge of the final
verification of an autonomous software development project.

PROJECT OBJECTIVE:
{objective}

PROJECT SUMMARY (generated during development):
{summary}

PROJECT FILES ({project_info.get('files_shown', 0)} shown,
{project_info.get('files_omitted', 0)} omitted due to limit):
{chr(10).join(f'- {f}' for f in project_info.get('files', []))}

FILE CONTENTS:
{files_section}

EXPORTED/DEFINED SYMBOLS (Python — AST):
{symbols_section}

YOUR TASK:
Analyze whether the CURRENT project meets the OBJECTIVE above. Look for:

1. OBJECTIVE REQUIREMENTS: is each requirement implemented?
   Do the mentioned functions/classes/features exist?
2. INTEGRATION: do imports and cross-references make sense? Does a
   file reference something that does not exist? Were imported
   symbols actually defined in the source files?
3. LOGIC: is there clearly incorrect logic (inverted conditions,
   wrong values, flows that never run)?
4. CONSISTENCY: are names, types and interfaces consistent across
   files?

The project summary may be outdated — trust the real file contents
above, not the summary, when they conflict.

Return ONLY JSON in the format:
{{"passed": true/false, "problems": ["problem 1", "problem 2", ...]}}

If passed=true, problems must be an empty list [].
If passed=false, problems must contain each problem found as a
descriptive, specific string (e.g. "auth.py imports 'UserService'
which is not exported in users.py — list_symbols shows only 'UserModel'").

    If the objective is simple and the project looks complete and correct,
    return {{"passed": true, "problems": []}}.
"""
