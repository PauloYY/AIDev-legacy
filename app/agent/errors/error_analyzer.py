"""Fase 5 — Error Analyzer: resultado bruto → problemas estruturados.

Regras:
- UMA chamada LLM por resultado com falhas (nunca por erro individual).
- Só para comando de teste/verificação com falha (o chamador filtra;
  aqui há guarda defensiva extra).
- Distingue erro observado / causa provável / solução sugerida; sem
  evidência: probable_cause="unknown", suggested_solution="investigate".
- Anti-invenção: affected_files só com grounding no output; ids e
  contagens validados; qualquer falha → fallback determinístico
  (parse de FAILED/ERROR + referências de arquivo) ou lista vazia.
- Nunca levanta.
"""

import json
import re
from dataclasses import dataclass, field

# Linhas que indicam falha de teste (1ª passada determinística).
_FAILURE_LINE_RE = re.compile(
    r"^\s*(FAILED|ERROR|FAIL:|FAILED:|Error)\b(.*)$",
    re.IGNORECASE,
)
# Referências a arquivos com linha (tracebacks, pytest short summary).
_FILE_REF_RE = re.compile(
    r"([\w\-./\\]+\.(?:py|js|jsx|ts|tsx|go|java|rb|php|c|cc|cpp|h|rs))"
    r"(?::(\d+))?",
)
# test_id tipo pytest: path::test (com ou sem linha).
_TEST_ID_RE = re.compile(
    r"([\w\-./\\]+\.(?:py|js|jsx|ts|tsx|go))\s*::\s*([\w\-]+)",
)

MAX_PROBLEMS = 20
MAX_ERROR_CHARS = 300
MAX_INPUT_CHARS = 6000


def normalize_test_id(text: str) -> str:
    """Identificador estável do teste ("" quando não identificável)."""
    try:
        match = _TEST_ID_RE.search(text or "")
    except Exception:
        return ""
    if not match:
        return ""
    path, name = match.group(1).strip(), match.group(2).strip()
    base = path.replace("\\", "/").split("/")[-1]
    return f"{base}::{name}" if base and name else ""


def normalize_error(text: str) -> str:
    """Erro normalizado p/ dedup conservadora (colapsa whitespace)."""
    try:
        collapsed = re.sub(r"\s+", " ", text or "").strip()
    except Exception:
        return ""
    if len(collapsed) <= MAX_ERROR_CHARS:
        return collapsed
    return collapsed[:MAX_ERROR_CHARS]


def _grounded_files(candidates, result_text: str) -> list[str]:
    """Mantém só arquivos evidenciados no output (anti-invenção)."""
    grounded: list[str] = []
    try:
        haystack = result_text or ""
        for candidate in candidates or []:
            if not isinstance(candidate, str):
                continue
            name = candidate.strip().replace("\\", "/")
            if not name:
                continue
            base = name.split("/")[-1]
            if base and base in haystack:
                if name not in grounded:
                    grounded.append(name[:200])
    except Exception:
        pass
    return grounded


def _extract_candidates(result_text: str) -> list[dict]:
    """Pré-parse determinístico: falhas + arquivos (sem LLM)."""
    candidates: list[dict] = []
    try:
        text = result_text if isinstance(result_text, str) else str(
            result_text)
    except Exception:
        return candidates
    for line in text.splitlines():
        if len(candidates) >= MAX_PROBLEMS:
            break
        try:
            match = _FAILURE_LINE_RE.match(line)
        except Exception:
            continue
        if not match:
            continue
        error = line.strip()[:MAX_ERROR_CHARS]
        if not error:
            continue
        files: list[str] = []
        try:
            for file_match in _FILE_REF_RE.finditer(line):
                path = file_match.group(1).replace("\\", "/")
                if path not in files:
                    files.append(path[:200])
                if len(files) >= 5:
                    break
        except Exception:
            pass
        candidates.append({
            "error": error,
            "test": normalize_test_id(line),
            "affected_files": files,
        })
    return candidates


@dataclass
class AnalyzedProblem:
    """Um problema extraído do resultado (hipótese, não certeza)."""

    error: str = ""
    probable_cause: str = "unknown"
    suggested_solution: str = "investigate"
    affected_files: list[str] = field(default_factory=list)
    test: str = ""


@dataclass
class FailureFact:
    """Fato determinístico (código extrai, LLM não reconstrói)."""

    test: str = ""
    error: str = ""
    files: list[str] = field(default_factory=list)


def extract_failure_facts(result) -> list[FailureFact]:
    """Fatos de falha sem LLM (base da projeção + fallback)."""
    facts: list[FailureFact] = []
    try:
        for candidate in _extract_candidates(
                result if isinstance(result, str) else str(result or "")):
            facts.append(FailureFact(
                test=candidate.get("test", ""),
                error=candidate.get("error", ""),
                files=list(candidate.get("affected_files", [])),
            ))
    except Exception:
        pass
    return facts


@dataclass
class ErrorAnalysis:
    command: str = ""
    problems: list[AnalyzedProblem] = field(default_factory=list)
    llm_calls: int = 0
    fallback_used: bool = False
    error: str | None = None
    # --- Fase 6: fatos determinísticos (sempre preenchidos) ---
    failures: list[FailureFact] = field(default_factory=list)
    exit_code: int | None = None


@dataclass
class UnifiedErrorAnalysis:
    """Uma análise → TaskState + Checklist + Planner + Executor.

    `failures`: fatos determinísticos (código). `problems`: hipóteses
    (1 LLM call ou fallback). `checklist_items`: descrições prontas
    p/ projeção (sem segunda análise). `exit_code`: parseado do output.
    """

    command: str = ""
    exit_code: int | None = None
    failures: list[FailureFact] = field(default_factory=list)
    problems: list[AnalyzedProblem] = field(default_factory=list)
    checklist_items: list[str] = field(default_factory=list)
    llm_calls: int = 0
    fallback_used: bool = False
    error: str | None = None


class ErrorAnalyzer:
    """Analisa UM resultado com falhas → N problemas (1 LLM call max)."""

    COMPONENT = "ErrorAnalyzer"

    def __init__(self, llm=None, max_problems: int = MAX_PROBLEMS):
        self.llm = llm
        self.max_problems = max_problems if max_problems and (
            max_problems > 0) else MAX_PROBLEMS

    def analyze(
        self,
        command,
        result,
        iteration: int | None = None,
    ) -> ErrorAnalysis:
        try:
            command_text = command if isinstance(command, str) else str(
                command or "")
        except Exception:
            command_text = ""
        try:
            result_text = result if isinstance(result, str) else str(
                result or "")
        except Exception:
            result_text = ""
        analysis = ErrorAnalysis(command=command_text[:200])
        # Fase 6: fatos determinísticos sempre (código, não LLM).
        try:
            analysis.failures = extract_failure_facts(result_text)
            from app.agent.trace import parse_exit_code

            analysis.exit_code = parse_exit_code(result_text)
        except Exception:
            pass
        # Guarda defensiva: sem falha aparente, nada a analisar.
        if "exit code 0" in result_text.split("\n", 1)[0]:
            return analysis
        if self.llm is None:
            return self._fallback(analysis, result_text)
        # Conta a tentativa mesmo se falhar (métrica honesta: 1 call
        # feita, sem retry em loop — o fallback não chama LLM).
        analysis.llm_calls = 1
        try:
            content = self._generate(result_text, command_text,
                                     iteration)
        except Exception as error:
            analysis.error = f"llm_error: {type(error).__name__}"
            return self._fallback(analysis, result_text)
        parsed = self._validate(content, result_text)
        if parsed is None:
            analysis.error = "invalid_response"
            return self._fallback(analysis, result_text)
        analysis.problems = parsed
        return analysis

    def _generate(self, result_text: str, command: str,
                  iteration: int | None):
        from app.llm.models import Message

        truncated = result_text
        if len(truncated) > MAX_INPUT_CHARS:
            truncated = truncated[:MAX_INPUT_CHARS]
        prompt = (
            "Analyze this failed test/build output. Extract EVERY "
            "distinct failure (do not stop at the first one).\n"
            "For each failure report ONLY what the output shows:\n"
            "- \"error\": the observed error line (verbatim, short).\n"
            "- \"test\": the failing test identifier if shown "
            "(e.g. path::test_name), else \"\".\n"
            "- \"probable_cause\": likely cause as HYPOTHESIS, or "
            "\"unknown\" when the output does not show enough evidence.\n"
            "- \"suggested_solution\": next step as SUGGESTION, or "
            "\"investigate\" when unclear.\n"
            "- \"affected_files\": files MENTIONED in the output only. "
            "NEVER invent files, functions, requirements or causes.\n"
            "Return ONLY JSON: {\"problems\": [{\"error\": ..., "
            "\"test\": ..., \"probable_cause\": ..., "
            "\"suggested_solution\": ..., \"affected_files\": [...]}]}.\n"
            "\nCOMMAND:\n" + command_text_safe(command) + "\n"
            "\nOUTPUT:\n" + truncated
        )
        try:
            response = self.llm.generate(
                messages=[Message(role="user", content=prompt)],
                component=self.COMPONENT,
                iteration=iteration,
            )
        except TypeError:
            response = self.llm.generate(
                messages=[Message(role="user", content=prompt)],
            )
        return getattr(response, "content", None)

    def _validate(self, content, result_text: str,
                  ) -> list[AnalyzedProblem] | None:
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            from app.llm.json_extraction import parse_json_object

            data = parse_json_object(content)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        raw = data.get("problems")
        if not isinstance(raw, list) or not raw:
            return None
        problems: list[AnalyzedProblem] = []
        for entry in raw[:self.max_problems]:
            try:
                if not isinstance(entry, dict):
                    continue
                error = entry.get("error", "")
                error_text = error if isinstance(
                    error, str) else str(error)
                error_text = error_text.strip()[:MAX_ERROR_CHARS]
                if not error_text:
                    continue
                cause = entry.get("probable_cause", "unknown")
                cause_text = (cause if isinstance(cause, str)
                              else str(cause)).strip()[:MAX_ERROR_CHARS]
                solution = entry.get("suggested_solution",
                                     "investigate")
                solution_text = (solution if isinstance(solution, str)
                                 else str(solution)).strip()[
                    :MAX_ERROR_CHARS]
                test = entry.get("test", "")
                test_text = (test if isinstance(test, str)
                             else str(test)).strip()[:200]
                if not test_text:
                    test_text = normalize_test_id(error_text)
                problems.append(AnalyzedProblem(
                    error=error_text,
                    probable_cause=cause_text or "unknown",
                    suggested_solution=solution_text or "investigate",
                    affected_files=_grounded_files(
                        entry.get("affected_files"), result_text),
                    test=test_text,
                ))
            except Exception:
                continue
        return problems or None

    def _fallback(self, analysis: ErrorAnalysis,
                  result_text: str) -> ErrorAnalysis:
        analysis.fallback_used = True
        try:
            for candidate in _extract_candidates(result_text):
                analysis.problems.append(AnalyzedProblem(
                    error=candidate["error"],
                    affected_files=candidate["affected_files"],
                    test=candidate["test"],
                ))
                if len(analysis.problems) >= self.max_problems:
                    break
        except Exception:
            pass
        return analysis


def command_text_safe(command) -> str:
    """Comando truncado p/ o prompt (nunca levanta)."""
    try:
        text = command if isinstance(command, str) else str(
            command or "")
    except Exception:
        return ""
    return text[:500]


def checklist_description(test: str, error: str) -> str:
    """Descrição operacional p/ projeção (estilo legado, sem LLM)."""
    try:
        err = (error if isinstance(error, str) else str(
            error or "")).strip()
        tst = (test if isinstance(test, str) else str(
            test or "")).strip()
        if tst:
            text = f"{tst} failed: {err}" if err else f"{tst} failed"
        else:
            text = f"test failed: {err}" if err else "test failed"
        return text[:200] or "test failed"
    except Exception:
        return "test failed"


class UnifiedErrorAnalyzer:
    """Orquestra UMA análise → TaskState + Checklist + Planner + Executor.

    Fase 6: compõe extração determinística (fatos) + UM ErrorAnalyzer
    (hipóteses, 1 LLM call max). Não é um terceiro analisador — reusa
    ErrorAnalyzer; o ganho é UMA chamada alimentando N consumidores.
    Nunca levanta.
    """

    def __init__(self, llm=None, max_problems: int = MAX_PROBLEMS):
        self.llm = llm
        self.max_problems = max_problems if max_problems and (
            max_problems > 0) else MAX_PROBLEMS

    def analyze(
        self,
        command,
        result,
        iteration: int | None = None,
    ) -> UnifiedErrorAnalysis:
        try:
            command_text = command if isinstance(command, str) else str(
                command or "")
        except Exception:
            command_text = ""
        unified = UnifiedErrorAnalysis(command=command_text[:200])
        try:
            result_text = result if isinstance(result, str) else str(
                result or "")
        except Exception:
            result_text = ""
        try:
            unified.failures = extract_failure_facts(result_text)
            from app.agent.trace import parse_exit_code

            unified.exit_code = parse_exit_code(result_text)
        except Exception:
            pass
        # Guarda defensiva: sem falha aparente, só fatos (vazios).
        if "exit code 0" in result_text.split("\n", 1)[0]:
            return unified
        try:
            analysis = ErrorAnalyzer(
                llm=self.llm,
                max_problems=self.max_problems).analyze(
                    command_text, result_text, iteration)
        except Exception:
            analysis = None
        if analysis is None:
            unified.fallback_used = True
            unified.error = "analyzer_crashed"
        else:
            unified.problems = list(analysis.problems or [])
            unified.llm_calls = analysis.llm_calls
            unified.fallback_used = analysis.fallback_used
            unified.error = analysis.error
        # Projeção do checklist (determinística, sem segunda análise):
        # hipóteses primeiro; fatos quando só eles existirem.
        items: list[str] = []
        try:
            for problem in unified.problems[:self.max_problems]:
                items.append(checklist_description(
                    problem.test, problem.error))
            if not items:
                for fact in unified.failures[:self.max_problems]:
                    items.append(checklist_description(
                        fact.test, fact.error))
        except Exception:
            pass
        unified.checklist_items = items[:self.max_problems]
        return unified
