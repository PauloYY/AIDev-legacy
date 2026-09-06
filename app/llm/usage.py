from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.llm.models import Usage


@dataclass
class CallRecord:
    component: str
    provider: str | None
    usage: Usage
    iteration: int | None
    timestamp: str
    prompt_chars: int = 0
    completion_chars: int = 0

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
