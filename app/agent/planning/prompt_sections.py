"""Decomposição observacional do prompt do Planner (Etapa 2B).

NÃO otimiza nem remove contexto — apenas mede.

O Planner recebe `objective: str` e `context: str` já concatenados pelo
Runner. Este módulo recupera os componentes individuais a partir das
marcações estáveis produzidas por:

- Runner._build_memory_block (RESUMO, CHECKLIST, ERROS PROIBIDOS, etc.)
- OperationalMemory.render (HISTÓRICO, COMANDOS CONHECIDOS, ARQUIVOS)
- TaskContextBuilder.build (TASK PAI:)
- Runner pós-execução (RESULTADO DA EXECUÇÃO:)
- Blocos de erro do Runner + retry do Planner (ERRO.../CORREÇÃO...)

Se o formato do contexto mudar no futuro, as seções ausentes retornam ""
e o resíduo vai para `other_context` — nunca levanta exceção.
"""

from app.llm.utils import estimate_tokens

# Componentes dinâmicos extraídos do `context` (ordem de aparição esperada).
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

# Todos os componentes medidos por chamada (estáticos + dinâmicos + objective).
PLANNER_COMPONENTS = ["static_template", "objective"] + DYNAMIC_COMPONENTS

# Marcadores estáveis (prefixos) usados para fatiar o contexto.
PROJECT_SUMMARY_MARKER = "RESUMO DO PROJETO:"
OBJECTIVE_CHECKLIST_MARKER = "CHECKLIST DO OBJETIVO"
ERROR_CHECKLIST_MARKER = "CHECKLIST DE ERROS ATUAIS"
PLANNER_ERROR_MEMORY_MARKER = "ERROS PROIBIDOS"
ACTION_HISTORY_MARKER = "HISTÓRICO DE AÇÕES"
KNOWN_COMMANDS_MARKER = "COMANDOS CONHECIDOS QUE JÁ FUNCIONARAM"
FILE_LIST_MARKER = "ARQUIVOS ATUAIS DO PROJETO"
TASK_CONTEXT_MARKER = "TASK PAI:"
EXECUTION_RESULT_MARKER = "RESULTADO DA EXECUÇÃO:"

# Blocos de erro/ retry anexados ao final do contexto. Ordem não importa;
# a detecção usa o menor índice dentre todos após o resultado/arquivos.
# NOTA: "ERRO DE REPETIÇÃO:" é prefixo de "ERRO DE REPETIÇÃO/ESTAGNAÇÃO:"?
# Não — o segundo tem "/" após REPETIÇÃO, então a busca com ":" não colide.
ERROR_BLOCK_MARKERS = (
    "ERRO DE VALIDAÇÃO ANTES DO FINISH:",
    "ERRO DE REPETIÇÃO/ESTAGNAÇÃO:",
    "ERRO DE REPETIÇÃO:",
    "INVESTIGAÇÃO REALIZADA COMO ÚLTIMO RECURSO",
    "CORREÇÃO DA TENTATIVA ANTERIOR:",
    "ERRO REPETIDO — LEIA COM ATENÇÃO:",
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

    # Localiza cada marcador principal (primeira ocorrência).
    i_summary = _find_first(text, PROJECT_SUMMARY_MARKER)
    i_obj = _find_first(text, OBJECTIVE_CHECKLIST_MARKER)
    i_err_check = _find_first(text, ERROR_CHECKLIST_MARKER)
    i_err_mem = _find_first(text, PLANNER_ERROR_MEMORY_MARKER)
    i_hist = _find_first(text, ACTION_HISTORY_MARKER)
    i_known = _find_first(text, KNOWN_COMMANDS_MARKER)
    i_files = _find_first(text, FILE_LIST_MARKER)
    i_task = _find_first(text, TASK_CONTEXT_MARKER)
    i_result = _find_first(text, EXECUTION_RESULT_MARKER)

    # Blocos de erro: procura a partir do resultado (ou task, ou arquivos,
    # ou início) para evitar confundir conteúdo do summary com erro real.
    # Na prática os erros ficam no final; usar o menor índice após o ponto
    # de ancoragem captura o início da cauda de erros.
    anchor = 0
    for candidate in (i_result, i_task, i_files):
        if candidate != -1:
            anchor = candidate
            break
    # Se há resultado/task/arquivos, erros vêm depois; senão, do início.
    i_error = _find_earliest_error_block(text, anchor if anchor else 0)
    # Se o "erro" encontrado está antes da âncora (ex.: palavra ERRO dentro
    # do summary), ignora e procura após a âncora + len do marcador âncora.
    # Como anchor já é o início da seção âncora, qualquer erro antes dela
    # seria falso positivo — re-procura estritamente depois.
    if i_error != -1 and anchor and i_error < anchor:
        # Avança a busca para depois da âncora.
        i_error = _find_earliest_error_block(text, anchor + 1)
        # Ainda pode estar dentro da seção âncora (ex.: resultado contém
        # texto de erro). Para execução/result/task/files, o bloco de erro
        # real sempre começa com "\n\n" + marcador no final; conteúdo
        # interno raramente tem o marcador exato no início de linha.
        # Aceitamos o primeiro após a âncora como início da cauda.

    # Monta lista ordenada de (nome, índice) para fatiamento sequencial.
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
        # Nenhum marcador conhecido: tudo é contexto residual.
        sections["other_context"] = text
        return sections

    ordered.sort(key=lambda item: item[1])

    # Texto antes do primeiro marcador (normalmente vazio) -> other_context.
    first_idx = ordered[0][1]
    leading = text[:first_idx]
    # Separadores "\n\n" entre blocos não pertencem a nenhuma seção;
    # o overhead é documentado via prompt_chars vs soma. Aqui guardamos
    # apenas resíduo não-branco como other_context.
    if leading.strip():
        sections["other_context"] = leading

    for pos, (name, start) in enumerate(ordered):
        end = ordered[pos + 1][1] if pos + 1 < len(ordered) else len(text)
        sections[name] = text[start:end]

    # Se não há marcadores de erro mas há cauda após execution_result que
    # não foi capturada (não deveria ocorrer pelo fatiamento acima), ela já
    # está dentro de execution_result. Nada a fazer.

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
