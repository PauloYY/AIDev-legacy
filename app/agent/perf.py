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
    ]

    return "\n".join(lines)
