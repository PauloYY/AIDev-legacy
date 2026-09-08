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
            "VERIFICAÇÃO FINAL DO PROJETO:",
            "Problemas encontrados que precisam ser corrigidos antes de finalizar:",
        ]

        for i, problem in enumerate(cleaned, start=1):
            report_lines.append(f"- {i}. {problem}")

        if not passed and not cleaned:
            report_lines.insert(
                2,
                "(A LLM avaliou que o objetivo não foi plenamente atingido, "
                "mas não especificou problemas detalhados.)",
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

        # Etapa 3: leituras independentes (read_file + list_symbols são
        # puras) podem rodar em paralelo; a montagem dos dicts continua
        # em ordem de arquivo (determinístico). Qualquer outro caso usa
        # o caminho sequencial legado.
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
        """Lê conteúdo (+símbolos p/ .py) de UM arquivo.

        Mesma lógica do loop sequencial legado, extraída para reúso
        pelo batch paralelo (Etapa 3). Falhas por arquivo retornam
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
                        f"...[truncado, {len(str(content_result)) - self.MAX_CONTEXT_CHARS} caracteres omitidos]"
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
            files_section = "(nenhum arquivo encontrado)"

        symbols_section = ""
        if project_info.get("symbols"):
            parts = []
            for path, syms in sorted(project_info["symbols"].items()):
                parts.append(f"## {path}:\n" + "\n".join(f"- {s}" for s in syms))
            symbols_section = "\n\n".join(parts)
        else:
            symbols_section = "(nenhum símbolo detectado)"

        return f"""
Você é um revisor de qualidade de software responsável por fazer uma
verificação final em um projeto de desenvolvimento autônomo.

OBJETIVO DO PROJETO:
{objective}

RESUMO DO PROJETO (gerado durante o desenvolvimento):
{summary}

ARQUIVOS DO PROJETO ({project_info.get('files_shown', 0)} mostrados,
{project_info.get('files_omitted', 0)} omitidos por limite):
{chr(10).join(f'- {f}' for f in project_info.get('files', []))}

CONTEÚDOS DOS ARQUIVOS:
{files_section}

SÍMBOLOS EXPORTADOS/DEFINIDOS (Python — AST):
{symbols_section}

SUA TAREFA:
Analise se o projeto ATUAL atende ao OBJETIVO acima. Procure por:

1. OBRIGATORIEDADES DO OBJETIVO: cada requisito do objetivo está
   implementado? Funções/classes/funcionalidades mencionadas existem?
2. INTEGRACAO: imports e referências cruzadas fazem sentido? Um
   arquivo referencia algo que não existe? Símbolos importados foram
   realmente definidos nos arquivos de origem?
3. LÓGICA: há lógica claramente incorreta (condições invertidas,
   valores errados, fluxos que nunca executam)?
4. CONSISTENCIA: nomes, tipos e interfaces são consistentes entre
   arquivos?

O resumo do projeto pode estar desatualizado — confie nos conteúdos
dos arquivos reais acima, não no resumo, quando houver conflito.

Retorne SOMENTE um JSON no formato:
{{"passed": true/false, "problems": ["problema 1", "problema 2", ...]}}

Se passed=true, problems deve ser uma lista vazia [].
Se passed=false, problems deve conter cada problema encontrado como
string descritiva e específica (ex.: "auth.py importa 'UserService'
que não é exportado em users.py — list_symbols mostra apenas 'UserModel').

    Se o objetivo for simples e o projeto parecer completo e correto,
    retorne {{"passed": true, "problems": []}}.
"""
