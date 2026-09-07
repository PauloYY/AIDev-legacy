"""Resumo de performance de uma run (Fase 3).

Combina dados já coletados (UsageTracker p/ LLM e tools) com contadores
do Runner. Não coleta nada sozinho — só formata. Sem conteúdo sensível:
apenas contagens, tempos e tokens.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStats:
    iterations: int = 0
    planner_calls: int = 0
    corrections: int = 0
    executor_errors: int = 0
    finish_blocks: int = 0
    loop_hits: int = 0
    test_runs: int = 0
    test_passed: int = 0
    test_failed: int = 0
    wall_ms: float = 0.0
    # Etapa 3 (paralelismo): batches de operações independentes.
    # parallel_ops = nº de operações que rodaram em batch;
    # sequential_ops = nº que rodaram pelo caminho sequencial;
    # parallel_saved_ms = soma(durações) − parede, por batch (estimativa).
    parallel_batches: int = 0
    parallel_ops: int = 0
    sequential_ops: int = 0
    parallel_saved_ms: float = 0.0
    # Etapa 4: atualizações de resumo puladas por política (leituras
    # puras). Não é erro nem falha — é chamada LLM evitada.
    summary_skipped: int = 0


def _llm_latency(records: list) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    durations = [r.duration_ms for r in records]
    return sum(durations) / len(durations), max(durations)


def format_performance_summary(
    result: str,
    usage=None,
    tool_stats: dict[str, Any] | None = None,
    agent: AgentStats | None = None,
) -> str:
    """Monta o bloco de performance. Tolera trackers/fakes antigos."""

    agent = agent or AgentStats()
    tool_stats = tool_stats or {}

    try:
        records = usage.get_records() if usage is not None else []
    except (AttributeError, TypeError):
        records = []

    try:
        llm_calls = usage.calls if usage is not None else 0
        total_tokens = usage.total.total_tokens if usage is not None else 0
    except (AttributeError, TypeError):
        llm_calls, total_tokens = 0, 0

    avg_lat, max_lat = _llm_latency(records)

    # Etapa 4: latência total real (soma) + contagens por componente.
    # Tudo derivado dos registros existentes; tolera trackers/fakes.
    total_lat_ms = 0.0
    comp_calls: dict[str, int] = {}
    planner_retries = (0, 0, 0)  # (retries, short, full)
    try:
        comp_stats = usage.component_stats() if usage is not None else {}
    except (AttributeError, TypeError):
        comp_stats = {}
    if isinstance(comp_stats, dict):
        for comp_name, comp in comp_stats.items():
            try:
                comp_calls[comp_name] = int(comp.get("calls", 0))
                total_lat_ms += float(comp.get("latency_ms_total", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
        try:
            req_stats = (
                usage.planner_request_stats()
                if usage is not None else {}
            )
        except (AttributeError, TypeError):
            req_stats = {}
        if isinstance(req_stats, dict):
            short = int(req_stats.get("short_repair", {}).get("calls", 0))
            full = int(req_stats.get("full_retry", {}).get("calls", 0))
            planner_retries = (short + full, short, full)

    tool_calls = tool_stats.get("calls", 0)
    tool_ms = tool_stats.get("total_ms", 0.0)
    by_tool = tool_stats.get("by_tool", {})

    lines = [
        "========== PERFORMANCE SUMMARY ==========",
        f"Result: {result}",
        f"Wall time: {agent.wall_ms / 1000:.1f}s",
        "",
        "LLM:",
        f"  Calls: {llm_calls}",
        f"  Tokens: {total_tokens:,}",
        f"  Avg latency: {avg_lat / 1000:.2f}s",
        f"  Max latency: {max_lat / 1000:.2f}s",
        f"  Total latency: {total_lat_ms / 1000:.1f}s",
        f"  Executor calls: {comp_calls.get('TaskDecisionMaker', 0)}",
        f"  Summary calls: {comp_calls.get('ProjectSummaryUpdater', 0)}",
        f"  Summary skipped: {getattr(agent, 'summary_skipped', 0)}",
        f"  Planner retries: {planner_retries[0]} "
        f"(short: {planner_retries[1]}, full: {planner_retries[2]})",
        "",
        "Tools:",
        f"  Calls: {tool_calls}",
        f"  Total time: {tool_ms / 1000:.1f}s",
    ]

    for name in sorted(by_tool):
        entry = by_tool[name]
        lines.append(
            f"  {name}: {entry['calls']} "
            f"({entry['total_ms'] / 1000:.1f}s)"
        )

    lines += [
        "",
        "Agent:",
        f"  Iterations: {agent.iterations}",
        f"  Planner calls: {agent.planner_calls}",
        f"  Corrections: {agent.corrections}",
        f"  Executor errors: {agent.executor_errors}",
        f"  Finish blocks: {agent.finish_blocks}",
        f"  Loops: {agent.loop_hits}",
        "",
        "Tests:",
        f"  Runs: {agent.test_runs}",
        f"  Passed: {agent.test_passed}",
        f"  Failed: {agent.test_failed}",
        "",
        "Parallel:",
        f"  Batches: {agent.parallel_batches}",
        f"  Parallel ops: {agent.parallel_ops}",
        f"  Sequential ops: {agent.sequential_ops}",
        f"  Est. saved: {agent.parallel_saved_ms / 1000:.2f}s",
    ]

    component_section = format_llm_component_section(usage)
    if component_section:
        lines += ["", component_section]

    return "\n".join(lines)


def format_llm_component_section(usage=None) -> str:
    """Seção por componente LLM (Etapa 4, aditiva).

    Uma linha por componente com chamadas, prompt médio/máximo (chars),
    tokens reais de prompt e latência média/máxima. Fonte: registros
    reais do UsageTracker (tokens do provider quando informados).
    Retorna "" sem registros — nunca quebra o resumo existente.
    """

    try:
        stats = usage.component_stats() if usage is not None else {}
    except (AttributeError, TypeError):
        return ""

    if not isinstance(stats, dict) or not stats:
        return ""

    header = (
        f"{'Component':<22} | {'Calls':>5} | "
        f"{'AvgChars':>8} | {'MaxChars':>8} | "
        f"{'AvgLat':>7} | {'MaxLat':>7}"
    )
    separator = "-" * len(header)
    lines = [
        "LLM by component:",
        header,
        separator,
    ]

    try:
        ordered = sorted(
            stats.keys(),
            key=lambda c: stats[c].get("latency_ms_total", 0.0),
            reverse=True,
        )
    except (TypeError, AttributeError):
        return ""

    for name in ordered:
        comp = stats[name]
        try:
            lines.append(
                f"{str(name)[:22]:<22} | "
                f"{int(comp.get('calls', 0)):>5} | "
                f"{int(comp.get('prompt_chars_avg', 0)):>8,} | "
                f"{int(comp.get('prompt_chars_max', 0)):>8,} | "
                f"{float(comp.get('latency_ms_avg', 0.0)) / 1000:>6.2f}s | "
                f"{float(comp.get('latency_ms_max', 0.0)) / 1000:>6.2f}s"
            )
        except (TypeError, ValueError):
            continue

    return "\n".join(lines)
