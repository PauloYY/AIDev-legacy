"""Trace persistente e estruturado de execução (Fase 3 — observabilidade).

Somente observa: nunca altera decisões, prompts, regras de correção,
ciclo de erros ou lógica de finish. Cada execução do Runner gera um
arquivo JSONL próprio (um evento por linha), que sobrevive ao término
do processo e permite reconstruir posteriormente o que aconteceu em
cada iteração (planner attempts/retries, erros do executor, testes,
bloqueios de finish, verificação final).

Garantias:
- Nunca sobrescreve traces anteriores (run_id único por execução).
- A escrita nunca quebra a execução principal: qualquer falha ao
  persistir é apenas logada e a run continua normalmente.
- Nunca armazena conteúdo integral de `write_file`, prompts completos,
  stdout/stderr enormes ou valores com cara de segredo — apenas
  resumos truncados e metadados (tool, file_path, exit code etc.).
"""

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# Limites de representação (só afetam o que vai para o trace, nunca os
# dados reais da execução).
MAX_COMMAND_CHARS = 500
MAX_RESULT_CHARS = 2000
MAX_ERROR_CHARS = 500
MAX_OBJECTIVE_CHARS = 2000
MAX_FILE_PATH_CHARS = 500
MAX_ARG_VALUE_CHARS = 500
MAX_CHECKLIST_ITEM_CHARS = 200

_EXIT_CODE_RE = re.compile(r"exit code (-?\d+)")

# Chaves cujo valor nunca é persistido integralmente.
_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "auth",
    "bearer",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: Any, limit: int) -> str:
    try:
        text = text if isinstance(text, str) else str(text)
    except Exception:
        return "<unrepresentable>"
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [+{omitted} chars]"


def truncate_text(text: Any, limit: int = MAX_RESULT_CHARS) -> str:
    """Truncamento público para reúso (ex.: Runner monta resumos)."""
    return _truncate(text, limit)


def _looks_sensitive(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_SUBSTRINGS)


def sanitize_arguments(tool: Any, arguments: Any) -> dict[str, Any]:
    """Representação segura dos argumentos de uma tool para o trace.

    - `write_file.content` nunca vai integral (vira "<omitted: N chars>").
    - `edit_file.old_text`/`edit_file.new_text` seguem a mesma regra
      (podem conter o arquivo quase inteiro).
    - Valores com cara de segredo (token, api_key, ...) viram "<redacted>".
    - Strings longas são truncadas; valores não-serializáveis viram str.
    - Nunca muta o dict original.
    """

    if not isinstance(arguments, dict):
        return {"_value": _truncate(arguments, MAX_ARG_VALUE_CHARS)}

    _LARGE_TEXT_ARGS = {
        ("write_file", "content"),
        ("edit_file", "old_text"),
        ("edit_file", "new_text"),
    }

    compact: dict[str, Any] = {}
    for key, value in arguments.items():
        try:
            if (tool, key) in _LARGE_TEXT_ARGS:
                if value is None:
                    compact[key] = "<empty>"
                else:
                    text = value if isinstance(value, str) else str(value)
                    if len(text) == 0:
                        compact[key] = "<empty>"
                    else:
                        compact[key] = f"<omitted: {len(text)} chars>"
                continue

            if _looks_sensitive(key):
                compact[key] = "<redacted>"
                continue

            if isinstance(value, str):
                compact[key] = _truncate(value, MAX_ARG_VALUE_CHARS)
                continue

            # Não-string: mantém se for JSON-serializável e curto,
            # senão resume como string truncada.
            try:
                encoded = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                encoded = str(value)
            if len(encoded) > MAX_ARG_VALUE_CHARS + 20:
                compact[key] = _truncate(str(value), MAX_ARG_VALUE_CHARS)
            else:
                compact[key] = value
        except Exception:
            compact[key] = "<unrepresentable>"

    return compact


def sanitize_result(result: Any, limit: int = MAX_RESULT_CHARS) -> str:
    """Resumo truncado do resultado de uma tool (stdout/stderr)."""
    return _truncate(result, limit)


def error_type_and_signature(error: BaseException | Any) -> tuple[str, str]:
    """Tipo + assinatura curta de um erro (sem stack, sem dados)."""
    try:
        return type(error).__name__, _truncate(str(error), MAX_ERROR_CHARS)
    except Exception:
        return "UnknownError", "<unrepresentable>"


def extract_file_path(arguments: Any) -> str | None:
    """file_path dos argumentos, quando presente (só metadado)."""
    if isinstance(arguments, dict):
        value = arguments.get("file_path")
        if isinstance(value, str) and value:
            return _truncate(value, MAX_FILE_PATH_CHARS)
    return None


def parse_exit_code(result: Any) -> int | None:
    """Extrai o exit code do formato de run_command ("exit code N")."""
    try:
        match = _EXIT_CODE_RE.search(str(result))
    except Exception:
        return None
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def is_timeout_result(result: Any) -> bool:
    try:
        return str(result).lstrip().startswith("TIMEOUT:")
    except Exception:
        return False


def default_trace_dir() -> Path:
    configured = os.getenv("AIDEV_TRACE_DIR")
    if configured and configured.strip():
        return Path(configured.strip())
    return Path(tempfile.gettempdir()) / "aidev-traces"


# ---------- P6: rotação dos traces (por quantidade + tamanho total) ----------

# Cada run gera um arquivo próprio (aidev-trace-<stamp>-<run_id>.jsonl),
# então o crescimento é em Nº de arquivos, não num arquivo único. A
# rotação aqui é poda dos mais antigos, com defaults seguros e
# sobrescrevíveis por env (lidos a cada chamada, como default_trace_dir,
# para permitir override em testes sem reload).
TRACE_FILE_PREFIX = "aidev-trace-"
TRACE_FILE_SUFFIX = ".jsonl"
DEFAULT_TRACE_KEEP_FILES = 20
DEFAULT_TRACE_MAX_TOTAL_MB = 50


def _positive_int_env(name: str, default: int) -> int:
    """Env int com default seguro: ausente/inválido → default."""
    try:
        return int((os.getenv(name) or "").strip() or default)
    except (TypeError, ValueError):
        return default


def trace_rotation_limits() -> tuple[int, int]:
    """(keep_files, max_total_bytes) vigentes (P6, só observa).

    `AIDEV_TRACE_KEEP_FILES` (default 20) e `AIDEV_TRACE_MAX_TOTAL_MB`
    (default 50). Ausente/inválido → default; <= 0 desativa a dimensão
    correspondente (sem poda por ela).
    """
    keep = _positive_int_env(
        "AIDEV_TRACE_KEEP_FILES", DEFAULT_TRACE_KEEP_FILES)
    max_mb = _positive_int_env(
        "AIDEV_TRACE_MAX_TOTAL_MB", DEFAULT_TRACE_MAX_TOTAL_MB)
    return keep, max_mb * 1024 * 1024


def prune_old_traces(
    trace_dir: str | Path | None = None,
    keep_path: str | Path | None = None,
    keep_files: int | None = None,
    max_total_bytes: int | None = None,
) -> dict[str, int]:
    """Remove traces antigos além dos limites (P6). Nunca levanta.

    Mantém os `keep_files` mais recentes e garante total <=
    `max_total_bytes` (remove os mais antigos primeiro). `keep_path`
    (o log atual) nunca é removido. Só considera arquivos
    `aidev-trace-*.jsonl` — nada mais no diretório é tocado. Limites
    None → vigentes via `trace_rotation_limits()`; <= 0 desativa a
    dimensão. Retorna {"kept", "removed", "freed_bytes"}.
    """
    empty = {"kept": 0, "removed": 0, "freed_bytes": 0}
    try:
        directory = Path(trace_dir) if trace_dir else default_trace_dir()
        if keep_files is None or max_total_bytes is None:
            default_keep, default_bytes = trace_rotation_limits()
            if keep_files is None:
                keep_files = default_keep
            if max_total_bytes is None:
                max_total_bytes = default_bytes
        try:
            keep_resolved = Path(keep_path).resolve() if keep_path else None
        except Exception:
            keep_resolved = None

        candidates: list[tuple[float, str, Path, int]] = []
        try:
            entries = list(directory.iterdir())
        except OSError:
            return dict(empty)
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
                name = entry.name
                if not (
                    name.startswith(TRACE_FILE_PREFIX)
                    and name.endswith(TRACE_FILE_SUFFIX)
                ):
                    continue
                if keep_resolved is not None:
                    try:
                        if entry.resolve() == keep_resolved:
                            continue
                    except OSError:
                        continue
                stat = entry.stat()
                candidates.append(
                    (stat.st_mtime, name, entry, stat.st_size))
            except OSError:
                continue

        # Mais recentes por último (mtime, desempate por nome).
        candidates.sort(key=lambda item: (item[0], item[1]))
        removed = 0
        freed = 0

        def _drop(item: tuple[float, str, Path, int]) -> None:
            nonlocal removed, freed
            try:
                item[2].unlink()
            except OSError:
                return
            removed += 1
            freed += item[3]

        if keep_files is not None and keep_files > 0:
            while len(candidates) > keep_files:
                _drop(candidates.pop(0))

        if max_total_bytes is not None and max_total_bytes > 0:
            total = sum(item[3] for item in candidates)
            while candidates and total > max_total_bytes:
                victim = candidates.pop(0)
                _drop(victim)
                total -= victim[3]

        result = dict(empty)
        result["kept"] = len(candidates)
        result["removed"] = removed
        result["freed_bytes"] = freed
        return result
    except Exception as error:
        logger.warning("Trace: falha ao podar traces antigos: %s", error)
        return dict(empty)


class ExecutionTrace:
    """Abstração mínima do trace persistente (API: `record`)."""

    def __init__(
        self,
        trace_dir: str | Path | None = None,
        run_id: str | None = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.trace_dir = Path(trace_dir) if trace_dir else default_trace_dir()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.trace_dir / f"aidev-trace-{stamp}-{self.run_id}.jsonl"
        self._events = 0
        self._ensure_parent()
        # P6: cada run nova poda os traces de runs anteriores além dos
        # limites (best-effort, nunca quebra a run; o arquivo atual —
        # ainda nem criado — é excluído da poda via keep_path).
        try:
            prune_old_traces(self.trace_dir, keep_path=self.path)
        except Exception as error:
            logger.warning("Trace: falha na rotação inicial: %s", error)

    def _ensure_parent(self) -> None:
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            # Não quebra nada: cada record() tenta de novo e loga.
            logger.warning("Trace: não foi possível criar %s: %s",
                           self.trace_dir, error)

    def record(self, event: str, **fields: Any) -> None:
        """Anexa um evento ao JSONL. Nunca levanta exceção."""

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": _utcnow_iso(),
            "event": event,
        }
        for key, value in fields.items():
            try:
                json.dumps(value, ensure_ascii=False, default=str)
                payload[key] = value
            except Exception:
                try:
                    payload[key] = str(value)
                except Exception:
                    payload[key] = "<unrepresentable>"

        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False,
                                        default=str) + "\n")
            self._events += 1
        except Exception as error:
            logger.warning("Trace: falha ao persistir evento '%s': %s",
                           event, error)

    @property
    def event_count(self) -> int:
        return self._events

    def read_events(self) -> list[dict[str, Any]]:
        """Lê de volta os eventos persistidos (diagnóstico/testes)."""
        events: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


class NullTrace:
    """Trace desativado com a mesma API (`record` vira no-op)."""

    run_id = "disabled"
    path = None
    event_count = 0

    def record(self, event: str, **fields: Any) -> None:
        return None

    def read_events(self) -> list[dict[str, Any]]:
        return []
