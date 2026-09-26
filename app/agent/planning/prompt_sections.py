"""Decomposição observacional do prompt do Planner."""

from app.llm.utils import estimate_tokens

DYNAMIC_COMPONENTS = [
    "project_summary",
    "objective_checklist",
    "error_checklist",
    "planner_error_memory",
    "action_history",
    "known_commands",
    "file_list",
    "task_context",
    "execution_result",
    "error_blocks",
    "other_context",
]

PLANNER_COMPONENTS = ["static_template", "objective"] + DYNAMIC_COMPONENTS

PROJECT_SUMMARY_MARKER = "PROJECT SUMMARY:"
OBJECTIVE_CHECKLIST_MARKER = "OBJECTIVE CHECKLIST"
ERROR_CHECKLIST_MARKER = "CURRENT ERROR CHECKLIST"
PLANNER_ERROR_MEMORY_MARKER = "FORBIDDEN ERRORS"
ACTION_HISTORY_MARKER = "ACTION HISTORY"
KNOWN_COMMANDS_MARKER = "KNOWN WORKING COMMANDS"
FILE_LIST_MARKER = "CURRENT PROJECT FILES"
TASK_CONTEXT_MARKER = "PARENT TASK:"
EXECUTION_RESULT_MARKER = "EXECUTION RESULT:"

ERROR_BLOCK_MARKERS = (
    "VALIDATION ERROR BEFORE FINISH:",
    "REPETITION/STAGNATION ERROR:",
    "REPETITION ERROR:",
    "INVESTIGATION PERFORMED AS A LAST RESORT",
    "PREVIOUS ATTEMPT CORRECTION:",
    "REPEATED ERROR — READ CAREFULLY:",
)


def _coerce_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _find_first(haystack: str, needle: str, start: int = 0) -> int:
    return haystack.find(needle, start)


def _find_earliest_error_block(context: str, start: int) -> int:
    """Menor índice de qualquer marcador de erro a partir de `start`, ou -1."""
    best = -1
    for marker in ERROR_BLOCK_MARKERS:
        idx = context.find(marker, start)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    return best


def split_planner_context(context: str | None) -> dict[str, str]:
    """Fatia o `context` dinâmico do Planner em componentes.

    Retorna dict com todas as chaves de DYNAMIC_COMPONENTS ("" quando ausente).
    Nunca altera o conteúdo — apenas observa. Aceita None/não-string.
    """
    text = _coerce_to_str(context)
    sections: dict[str, str] = {name: "" for name in DYNAMIC_COMPONENTS}
    if not text:
        return sections

    i_summary = _find_first(text, PROJECT_SUMMARY_MARKER)
    i_obj = _find_first(text, OBJECTIVE_CHECKLIST_MARKER)
    i_err_check = _find_first(text, ERROR_CHECKLIST_MARKER)
    i_err_mem = _find_first(text, PLANNER_ERROR_MEMORY_MARKER)
    i_hist = _find_first(text, ACTION_HISTORY_MARKER)
    i_known = _find_first(text, KNOWN_COMMANDS_MARKER)
    i_files = _find_first(text, FILE_LIST_MARKER)
    i_task = _find_first(text, TASK_CONTEXT_MARKER)
    i_result = _find_first(text, EXECUTION_RESULT_MARKER)

    anchor = 0
    for candidate in (i_result, i_task, i_files):
        if candidate != -1:
            anchor = candidate
            break
    i_error = _find_earliest_error_block(text, anchor if anchor else 0)
    if i_error != -1 and anchor and i_error < anchor:
        i_error = _find_earliest_error_block(text, anchor + 1)

    ordered: list[tuple[str, int]] = []
    if i_summary != -1:
        ordered.append(("project_summary", i_summary))
    if i_obj != -1:
        ordered.append(("objective_checklist", i_obj))
    if i_err_check != -1:
        ordered.append(("error_checklist", i_err_check))
    if i_err_mem != -1:
        ordered.append(("planner_error_memory", i_err_mem))
    if i_hist != -1:
        ordered.append(("action_history", i_hist))
    if i_known != -1:
        ordered.append(("known_commands", i_known))
    if i_files != -1:
        ordered.append(("file_list", i_files))
    if i_task != -1:
        ordered.append(("task_context", i_task))
    if i_result != -1:
        ordered.append(("execution_result", i_result))
    if i_error != -1:
        ordered.append(("error_blocks", i_error))

    if not ordered:
        sections["other_context"] = text
        return sections

    ordered.sort(key=lambda item: item[1])

    first_idx = ordered[0][1]
    leading = text[:first_idx]
    if leading.strip():
        sections["other_context"] = leading

    for pos, (name, start) in enumerate(ordered):
        end = ordered[pos + 1][1] if pos + 1 < len(ordered) else len(text)
        sections[name] = text[start:end]


    return sections


def measure_sections(sections: dict[str, str | None]) -> dict[str, dict[str, int]]:
    """Mede chars + tokens estimados por componente via estimate_tokens().

    Aceita valores None/ausentes (vira ""). Nunca falha por componente vazio.
    Retorna {component: {"chars": int, "estimated_tokens": int}}.
    """
    measured: dict[str, dict[str, int]] = {}
    for name, value in sections.items():
        text = _coerce_to_str(value)
        measured[name] = {
            "chars": len(text),
            "estimated_tokens": estimate_tokens(text),
        }
    return measured


def summarize_measurements(
    measured: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Totais agregados de uma medição (soma dos componentes)."""
    total_chars = sum(entry.get("chars", 0) for entry in measured.values())
    total_tokens = sum(
        entry.get("estimated_tokens", 0) for entry in measured.values()
    )
    return {"chars": total_chars, "estimated_tokens": total_tokens}
