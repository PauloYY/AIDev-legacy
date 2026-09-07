from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.llm.models import Usage
from app.llm.utils import estimate_tokens


def _estimated_tokens_for_chars(chars: int) -> int:
    """estimate_tokens() sem o texto real (só p/ agregados do tracker)."""
    if not chars:
        return 0
    return max(1, chars // 4)


@dataclass
class ToolCallRecord:
    """Uma execução de tool (Fase 3): sem tokens, sem conteúdo sensível.

    `command` só para run_command (truncado na origem); nunca inclui
    conteúdo de write_file.
    """

    tool: str
    duration_ms: float
    success: bool
    timestamp: str
    error: str | None = None
    timeout: bool = False
    exit_code: int | None = None
    command: str | None = None


@dataclass
class CallRecord:
    component: str
    provider: str | None
    usage: Usage
    iteration: int | None
    timestamp: str
    prompt_chars: int = 0
    completion_chars: int = 0
    # Etapa 2B: decomposição observacional do prompt do Planner.
    # {component: {"chars": int, "estimated_tokens": int}}. Apenas para
    # component="Planner"; None para demais componentes/chamadas antigas.
    context_breakdown: dict[str, dict[str, int]] | None = None
    # Etapa 2E: observabilidade por chamada (defaults mantêm compat).
    duration_ms: float = 0.0
    attempt: int = 1
    empty_response: bool = False
    success: bool = True
    error: str | None = None
    # Fase 3 Etapa 2 (short repair): qual prompt originou a chamada do
    # Planner — "normal" (primeira tentativa), "short_repair" (reparo
    # curto) ou "full_retry" (retry com contexto completo). Default
    # mantém compat com registros antigos e outros componentes.
    request_type: str = "normal"

    @property
    def total_chars(self) -> int:
        return self.prompt_chars + self.completion_chars


class UsageTracker:
    """Acumula o consumo de tokens ao longo de uma execução do agente.

    Isso resolve a falta de qualquer telemetria de custo/uso: sem isso,
    não havia como saber quantos tokens (e, por extensão, quanto custo)
    uma execução do agente consumiu.
    """

    def __init__(self):
        self.total: Usage = Usage()
        self.by_provider: dict[str, Usage] = {}
        self.calls: int = 0
        self._records: list[CallRecord] = []
        self._tool_records: list[ToolCallRecord] = []
        self._by_component: dict[str, Usage] = {}
        self._component_calls: dict[str, int] = {}

    def record(
        self,
        usage: Usage | None,
        provider: str | None,
        component: str | None = None,
        iteration: int | None = None,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        context_breakdown: dict[str, dict[str, int]] | None = None,
        duration_ms: float = 0.0,
        attempt: int = 1,
        empty_response: bool = False,
        success: bool = True,
        error: str | None = None,
        request_type: str = "normal",
    ) -> None:
        if usage is None:
            return

        self.calls += 1
        self.total = self.total + usage

        provider_key = provider or "unknown"
        current = self.by_provider.get(provider_key, Usage())
        self.by_provider[provider_key] = current + usage

        if component:
            current = self._by_component.get(component, Usage())
            self._by_component[component] = current + usage
            self._component_calls[component] = (
                self._component_calls.get(component, 0) + 1
            )

        self._records.append(
            CallRecord(
                component=component or "unknown",
                provider=provider_key,
                usage=usage,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
                context_breakdown=dict(context_breakdown)
                if context_breakdown is not None
                else None,
                duration_ms=duration_ms,
                attempt=attempt,
                empty_response=empty_response,
                success=success,
                error=error,
                request_type=request_type or "normal",
            )
        )

    def summary(self) -> str:
        lines = [
            f"Chamadas à LLM: {self.calls}",
            f"Tokens totais: {self.total.total_tokens} "
            f"(prompt: {self.total.prompt_tokens}, "
            f"completion: {self.total.completion_tokens})",
        ]

        for provider, usage in self.by_provider.items():
            lines.append(
                f"  - {provider}: {usage.total_tokens} tokens"
            )

        return "\n".join(lines)

    def breakdown(self) -> str:
        """Retorna um relatório detalhado do uso de tokens por componente."""

        if not self._by_component:
            return self.summary()

        header = (
            f"{'Component':<28} | {'Calls':>5} | "
            f"{'Prompt':>8} | {'Completion':>10} | {'Total':>8}"
        )
        separator = "-" * len(header)

        lines = [
            "========== TOKEN USAGE BREAKDOWN ==========",
            "",
            header,
            separator,
        ]

        ordered = sorted(
            self._by_component.keys(),
            key=lambda c: self._by_component[c].total_tokens,
            reverse=True,
        )

        for component in ordered:
            usage = self._by_component[component]
            call_count = self._component_calls[component]
            lines.append(
                f"{component:<28} | {call_count:>5} | "
                f"{usage.prompt_tokens:>8,} | "
                f"{usage.completion_tokens:>10,} | "
                f"{usage.total_tokens:>8,}"
            )

        lines.append(separator)
        lines.append(
            f"{'TOTAL':<28} | {self.calls:>5} | "
            f"{self.total.prompt_tokens:>8,} | "
            f"{self.total.completion_tokens:>10,} | "
            f"{self.total.total_tokens:>8,}"
        )
        lines.append("")

        return "\n".join(lines)

    def get_records(self) -> list[CallRecord]:
        return list(self._records)

    # ---------- Fase 3 Etapa 2: short repair do Planner ----------

    def planner_request_stats(self) -> dict[str, Any]:
        """Agregados do Planner por request_type (só observa).

        Retorna {request_type: {"calls", "prompt_tokens",
        "completion_tokens", "total_tokens", "avg_prompt_chars",
        "avg_duration_ms"}}. Novo método — não altera breakdown() nem
        nenhum formato público existente.
        """

        groups: dict[str, dict[str, Any]] = {}
        for record in self._records:
            if record.component != "Planner":
                continue
            key = record.request_type or "normal"
            entry = groups.setdefault(key, {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_chars": 0,
                "duration_ms": 0.0,
            })
            entry["calls"] += 1
            entry["prompt_tokens"] += record.usage.prompt_tokens
            entry["completion_tokens"] += record.usage.completion_tokens
            entry["total_tokens"] += record.usage.total_tokens
            entry["prompt_chars"] += record.prompt_chars
            entry["duration_ms"] += record.duration_ms

        for entry in groups.values():
            calls = entry["calls"] or 1
            entry["avg_prompt_chars"] = entry.pop("prompt_chars") // calls
            total_ms = entry.pop("duration_ms")
            entry["avg_duration_ms"] = total_ms / calls

        return groups

    # ---------- Etapa 4: agregados por componente (chamadas/contexto) ----------

    def component_stats(self) -> dict[str, dict[str, Any]]:
        """Agregados por componente LLM (Etapa 4, só observa).

        Para cada componente retorna: calls, prompt_chars (sum/avg/max),
        completion_chars (sum), prompt/completion/total_tokens REAIS do
        provider (sum; avg de prompt), latency_ms (sum/avg/max),
        retries (attempts > 1), empty_responses e failures. Tokens reais
        vêm de record.usage (resposta do provider); quando o provider
        não informa, valem 0 — nunca estimativa silenciosa.
        """

        groups: dict[str, dict[str, Any]] = {}
        for record in self._records:
            key = record.component or "unknown"
            entry = groups.setdefault(key, {
                "calls": 0,
                "prompt_chars_sum": 0,
                "prompt_chars_max": 0,
                "completion_chars_sum": 0,
                "prompt_tokens": 0,
                "prompt_tokens_max": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms_sum": 0.0,
                "latency_ms_max": 0.0,
                "retries": 0,
                "empty_responses": 0,
                "failures": 0,
            })
            entry["calls"] += 1
            entry["prompt_chars_sum"] += record.prompt_chars or 0
            entry["prompt_chars_max"] = max(
                entry["prompt_chars_max"], record.prompt_chars or 0)
            entry["completion_chars_sum"] += record.completion_chars or 0
            ptokens = record.usage.prompt_tokens if record.usage else 0
            entry["prompt_tokens"] += ptokens or 0
            entry["prompt_tokens_max"] = max(
                entry["prompt_tokens_max"], ptokens or 0)
            entry["completion_tokens"] += (
                record.usage.completion_tokens if record.usage else 0) or 0
            entry["total_tokens"] += (
                record.usage.total_tokens if record.usage else 0) or 0
            entry["latency_ms_sum"] += record.duration_ms or 0.0
            entry["latency_ms_max"] = max(
                entry["latency_ms_max"], record.duration_ms or 0.0)
            if (record.attempt or 1) > 1:
                entry["retries"] += 1
            if record.empty_response:
                entry["empty_responses"] += 1
            if not record.success:
                entry["failures"] += 1

        for entry in groups.values():
            calls = entry["calls"] or 1
            entry["prompt_chars_avg"] = entry["prompt_chars_sum"] // calls
            entry["prompt_tokens_avg"] = entry["prompt_tokens"] // calls
            entry["latency_ms_avg"] = entry["latency_ms_sum"] / calls
            entry["latency_ms_total"] = entry.pop("latency_ms_sum")

        return groups

    # ---------- Etapa 2E: métricas do ProjectSummaryUpdater ----------

    def _updater_records(self) -> list[CallRecord]:
        return [
            r for r in self._records if r.component == "ProjectSummaryUpdater"
        ]

    def project_summary_stats(self) -> dict[str, Any]:
        """Agregados do ProjectSummaryUpdater (Etapa 2E).

        Invocações são delimitadas por attempt==1 (retries compartilham
        a invocação). Sucesso/falha = estado do último attempt.
        """

        records = self._updater_records()
        if not records:
            return {
                "invocations": 0,
                "attempts": 0,
                "retries": 0,
                "empty_responses": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "avg_prompt_chars": 0,
                "max_prompt_chars": 0,
                "avg_prompt_tokens": 0,
                "max_prompt_tokens": 0,
                "avg_completion_tokens": 0.0,
                "max_completion_tokens": 0,
                "by_iteration": {},
            }

        invocations: list[list[CallRecord]] = []
        for record in records:
            if record.attempt <= 1 or not invocations:
                invocations.append([])
            invocations[-1].append(record)

        attempts = len(records)
        n_inv = len(invocations)
        empties = sum(1 for r in records if r.empty_response)

        def _invocation_ok(group: list[CallRecord]) -> bool:
            last = group[-1]
            return last.success and not last.empty_response

        successes = sum(1 for group in invocations if _invocation_ok(group))
        durations = [r.duration_ms for r in records]
        prompt_chars = [r.prompt_chars for r in records]
        prompt_tokens = [
            _estimated_tokens_for_chars(r.prompt_chars) for r in records
        ]
        completions = [r.usage.completion_tokens for r in records]
        by_iteration: dict[Any, int] = {}
        for r in records:
            by_iteration[r.iteration] = by_iteration.get(r.iteration, 0) + 1

        return {
            "invocations": n_inv,
            "attempts": attempts,
            "retries": attempts - n_inv,
            "empty_responses": empties,
            "successes": successes,
            "failures": n_inv - successes,
            "avg_duration_ms": sum(durations) / attempts,
            "max_duration_ms": max(durations),
            "avg_prompt_chars": sum(prompt_chars) // attempts,
            "max_prompt_chars": max(prompt_chars),
            "avg_prompt_tokens": sum(prompt_tokens) // attempts,
            "max_prompt_tokens": max(prompt_tokens),
            "avg_completion_tokens": sum(completions) / attempts,
            "max_completion_tokens": max(completions),
            "by_iteration": by_iteration,
        }

    def project_summary_section(self) -> str | None:
        """Seção compacta p/ CLI; None quando sem chamadas do componente."""

        stats = self.project_summary_stats()
        if not stats["invocations"]:
            return None

        lines = [
            "ProjectSummaryUpdater:",
            f"  calls: {stats['invocations']}",
            f"  retries: {stats['retries']}",
            f"  empty responses: {stats['empty_responses']}",
            f"  failures: {stats['failures']}",
            f"  avg duration: {stats['avg_duration_ms'] / 1000:.2f}s",
            f"  max duration: {stats['max_duration_ms'] / 1000:.2f}s",
            f"  avg prompt: {stats['avg_prompt_tokens']} tokens",
            f"  max prompt: {stats['max_prompt_tokens']} tokens",
        ]
        return "\n".join(lines)

    # ---------- Etapa Fase 3: métricas de ferramentas ----------

    def record_tool(
        self,
        tool: str,
        duration_ms: float,
        success: bool,
        error: str | None = None,
        timeout: bool = False,
        exit_code: int | None = None,
        command: str | None = None,
    ) -> None:
        self._tool_records.append(
            ToolCallRecord(
                tool=tool,
                duration_ms=duration_ms,
                success=success,
                timestamp=datetime.now().isoformat(),
                error=error,
                timeout=timeout,
                exit_code=exit_code,
                command=command,
            )
        )

    def get_tool_records(self) -> list["ToolCallRecord"]:
        return list(self._tool_records)

    def tool_stats(self) -> dict[str, Any]:
        """Agregados por tool: chamadas, tempos, falhas."""

        records = self._tool_records
        by_tool: dict[str, dict[str, Any]] = {}

        for record in records:
            entry = by_tool.setdefault(
                record.tool,
                {"calls": 0, "total_ms": 0.0, "max_ms": 0.0,
                 "failures": 0, "timeouts": 0},
            )
            entry["calls"] += 1
            entry["total_ms"] += record.duration_ms
            entry["max_ms"] = max(entry["max_ms"], record.duration_ms)
            if not record.success:
                entry["failures"] += 1
            if record.timeout:
                entry["timeouts"] += 1

        for entry in by_tool.values():
            entry["avg_ms"] = (
                entry["total_ms"] / entry["calls"] if entry["calls"] else 0.0
            )

        total_ms = sum(r.duration_ms for r in records)

        return {
            "calls": len(records),
            "total_ms": total_ms,
            "failures": sum(1 for r in records if not r.success),
            "timeouts": sum(1 for r in records if r.timeout),
            "by_tool": by_tool,
        }

    # ---------- Etapa 2B: instrumentação do contexto do Planner ----------

    def get_planner_context_history(self) -> list[dict[str, Any]]:
        """Histórico por iteração da decomposição do prompt do Planner.

        Retorna uma lista (ordem de chamada) com:
        {"iteration", "prompt_chars", "provider_prompt_tokens",
         "estimated_total_tokens", "sections": {component: {...}}}.
        Chamadas sem breakdown (não-Planner ou anteriores à Etapa 2B)
        são ignoradas.
        """
        history: list[dict[str, Any]] = []
        for record in self._records:
            if record.component != "Planner":
                continue
            if not record.context_breakdown:
                continue
            estimated_total = sum(
                entry.get("estimated_tokens", 0)
                for entry in record.context_breakdown.values()
            )
            history.append(
                {
                    "iteration": record.iteration,
                    "prompt_chars": record.prompt_chars,
                    "provider_prompt_tokens": record.usage.prompt_tokens,
                    "estimated_total_tokens": estimated_total,
                    "sections": dict(record.context_breakdown),
                }
            )
        return history

    def planner_context_stats(self) -> dict[str, Any]:
        """Agregados por componente (total/avg/max estimados) + provider.

        Retorna {"calls", "components": {name: {avg, total, max, avg_chars,
        total_chars, max_chars}}, "estimated_total_tokens",
        "provider_prompt_tokens_total", ...}. Vazio (calls=0) se sem dados.
        """
        history = self.get_planner_context_history()
        calls = len(history)
        if not calls:
            return {
                "calls": 0,
                "components": {},
                "estimated_total_tokens": 0,
                "provider_prompt_tokens_total": 0,
            }

        totals: dict[str, int] = {}
        maxes: dict[str, int] = {}
        total_chars: dict[str, int] = {}
        max_chars: dict[str, int] = {}
        provider_total = 0
        estimated_grand = 0
        for entry in history:
            provider_total += entry["provider_prompt_tokens"] or 0
            estimated_grand += entry["estimated_total_tokens"] or 0
            for name, sec in entry["sections"].items():
                tok = sec.get("estimated_tokens", 0)
                ch = sec.get("chars", 0)
                totals[name] = totals.get(name, 0) + tok
                total_chars[name] = total_chars.get(name, 0) + ch
                if name not in maxes or tok > maxes[name]:
                    maxes[name] = tok
                if name not in max_chars or ch > max_chars[name]:
                    max_chars[name] = ch

        components: dict[str, dict[str, int]] = {}
        for name, total in totals.items():
            components[name] = {
                "total_tokens": total,
                "avg_tokens": total // calls,
                "max_tokens": maxes.get(name, 0),
                "total_chars": total_chars.get(name, 0),
                "avg_chars": total_chars.get(name, 0) // calls,
                "max_chars": max_chars.get(name, 0),
            }

        return {
            "calls": calls,
            "components": components,
            "estimated_total_tokens": estimated_grand,
            "provider_prompt_tokens_total": provider_total,
        }

    def planner_context_breakdown(self) -> str:
        """Relatório observacional do contexto do Planner (Etapa 2B).

        Compatível e separado de breakdown(): não altera o formato
        existente. Mostra média/total/máximo ESTIMADOS por componente,
        compara com os tokens de prompt reais do provider e lista o
        crescimento por iteração. Puro relatório — sem otimização.
        """
        stats = self.planner_context_stats()
        calls = stats["calls"]
        if not calls:
            return "=== Planner Context Breakdown ===\n\n(no Planner calls with context breakdown recorded)"

        try:
            from app.agent.planning.prompt_sections import (
                PLANNER_COMPONENTS,
            )
        except Exception:
            PLANNER_COMPONENTS = []

        components: dict[str, dict[str, int]] = stats["components"]
        ordered = [c for c in (PLANNER_COMPONENTS or []) if c in components]
        ordered += sorted([c for c in components if c not in ordered])

        header = (
            f"{'Component':<22} | {'Avg Tokens':>10} | "
            f"{'Total Tokens':>12} | {'Max Tokens':>10}"
        )
        separator = "-" * len(header)
        lines = [
            "=== Planner Context Breakdown ===",
            "",
            f"Calls: {calls}",
            f"Provider prompt tokens (real, sum): {stats['provider_prompt_tokens_total']:,}",
            f"Estimated tokens (sum of sections): {stats['estimated_total_tokens']:,}",
            "",
            header,
            separator,
        ]
        for name in ordered:
            comp = components[name]
            lines.append(
                f"{name:<22} | {comp['avg_tokens']:>10,} | "
                f"{comp['total_tokens']:>12,} | {comp['max_tokens']:>10,}"
            )
        lines.append(separator)
        avg_total = stats["estimated_total_tokens"] // calls
        max_total = max(
            (e["estimated_total_tokens"] for e in self.get_planner_context_history()),
            default=0,
        )
        lines.append(
            f"{'TOTAL (estimated)':<22} | {avg_total:>10,} | "
            f"{stats['estimated_total_tokens']:>12,} | {max_total:>10,}"
        )
        provider_avg = stats["provider_prompt_tokens_total"] // calls
        provider_max = max(
            (
                e["provider_prompt_tokens"]
                for e in self.get_planner_context_history()
            ),
            default=0,
        )
        lines.append(
            f"{'TOTAL (provider)':<22} | {provider_avg:>10,} | "
            f"{stats['provider_prompt_tokens_total']:>12,} | {provider_max:>10,}"
        )
        lines.append("")
        lines.append(
            "Note: estimated tokens use estimate_tokens() (len//4, gross "
            "approximation for instrumentation). Provider tokens use the "
            "real tokenizer/billing of the model and include message "
            "framing overhead (roles, JSON envelope). Small differences "
            "are expected; large gaps indicate dynamic context growth."
        )
        lines.append("")
        lines.append("--- Per-iteration estimated totals ---")
        for entry in self.get_planner_context_history():
            lines.append(
                f"iteration {entry['iteration']}: "
                f"estimated={entry['estimated_total_tokens']:,} "
                f"provider_prompt={entry['provider_prompt_tokens']:,} "
                f"chars={entry['prompt_chars']:,}"
            )
        lines.append("")
        return "\n".join(lines)
