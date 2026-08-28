from dataclasses import dataclass, field

from app.llm.models import Usage


@dataclass
class UsageTracker:
    """Acumula o consumo de tokens ao longo de uma execução do agente.

    Isso resolve a falta de qualquer telemetria de custo/uso: sem isso,
    não havia como saber quantos tokens (e, por extensão, quanto custo)
    uma execução do agente consumiu.
    """

    total: Usage = field(default_factory=Usage)
    by_provider: dict[str, Usage] = field(default_factory=dict)
    calls: int = 0

    def record(self, usage: Usage | None, provider: str | None) -> None:
        if usage is None:
            return

        self.calls += 1
        self.total = self.total + usage

        provider_key = provider or "unknown"
        current = self.by_provider.get(provider_key, Usage())
        self.by_provider[provider_key] = current + usage

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
