class PlannerErrorMemory:
    """Memória determinística dos erros de decisão do Planner nesta run.

    Complementa o mecanismo de retry já existente no Runner (que só
    corrige o erro DENTRO da mesma iteração, com uma janela curta de
    tentativas — `MAX_PLANNER_ATTEMPTS`). Aquele mecanismo não impede
    o Planner de cometer o MESMO erro de novo numa iteração futura,
    já com o "orçamento de retries" resetado.

    Esta classe guarda, em Python puro (sem LLM nenhuma envolvida),
    toda mensagem de erro distinta já vista nesta run — mesmo entre
    iterações diferentes. Isso permite duas coisas:

    1. Uma lista permanente "isto já foi tentado e é proibido",
       injetada no contexto a cada iteração (prevenção proativa —
       ver `render()`).
    2. Detectar quando o Planner repete um erro que ele mesmo já
       cometeu antes (mesmo texto de erro exato) — nesse caso, o
       Runner trata isso como violação de uma proibição, não como
       "mais uma tentativa inválida igual às outras", e para de
       insistir no retry normal em vez de gastar todo o orçamento de
       novo com algo que já provou não se corrigir sozinho.
    """

    MAX_ERRORS_RENDERED = 20

    def __init__(self):
        self._seen: list[str] = []

    def reset(self) -> None:
        self._seen = []

    def seen(self, signature: str) -> bool:
        return signature in self._seen

    def record(self, signature: str) -> None:
        if signature not in self._seen:
            self._seen.append(signature)

    def render(self) -> str:
        if not self._seen:
            return (
                "FORBIDDEN ERRORS (none recorded yet in this run)."
            )

        shown = self._seen[-self.MAX_ERRORS_RENDERED :]

        lines = [
            "FORBIDDEN ERRORS (you already made these mistakes earlier "
            "in this run — DO NOT repeat them, under any circumstances):",
        ]
        lines += [f"- {error}" for error in shown]

        return "\n".join(lines)

    MAX_ERRORS_RENDERED_COMPACT = 5

    def render_compact(self, limit: int | None = None) -> str:
        """Últimos N erros proibidos. `render()` intacto."""
        if not self._seen:
            return (
                "FORBIDDEN ERRORS (none recorded yet in this run)."
            )
        count = limit or self.MAX_ERRORS_RENDERED_COMPACT
        shown = self._seen[-count:]
        omitted = len(self._seen) - len(shown)
        lines = [
            "FORBIDDEN ERRORS (you already made these mistakes earlier "
            "in this run — DO NOT repeat them, under any circumstances):",
        ]
        if omitted:
            lines.append(
                f"... [+{omitted} earlier error(s) omitted]"
            )
        lines += [f"- {error}" for error in shown]
        return "\n".join(lines)
