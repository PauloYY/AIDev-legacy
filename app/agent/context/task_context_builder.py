from typing import Any

from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task


class TaskContextBuilder:
    # Etapa 4: teto por resultado de dependency no contexto do Executor.
    # Mesma família do Runner.MAX_CONTEXT_CHARS (4000): o Executor já
    # recebe vereditos "verdict-first" (STATUS na 1ª linha do
    # run_command) e detalhes de falha também via error checklist
    # (gerado do resultado INTEGRAL, separadamente). Preserva o início
    # e marca a omissão — nunca trunca silenciosamente.
    DEFAULT_MAX_RESULT_CHARS = 4000

    # Etapa 6: teto compacto por resultado (só quando AIDEV_COMPACT_
    # EXECUTOR=1 E já havia um limite configurado). Arquivos pequenos
    # (o caso comum: < 2000 chars) passam byte-idênticos; só outputs
    # grandes encolhem, com o total omitido marcado. None (integral
    # explícito) continua integral — limite explícito não é adivinhado.
    COMPACT_MAX_RESULT_CHARS = 2000

    def __init__(self, max_result_chars: int | None = None):
        self.max_result_chars = max_result_chars

    def build(
        self,
        task: Task,
        dependency_results: list[Any],
    ) -> str:
        if len(task.dependencies) != len(dependency_results):
            raise ValueError(
                "A quantidade de dependencies e resultados deve ser igual."
            )

        context = [
            "PARENT TASK:",
            f"Tool: {task.tool}",
            f"Arguments: {task.arguments}",
            "",
            "DEPENDENCIES:",
        ]

        if not task.dependencies:
            context.append("None.")

        for index, (dependency, result) in enumerate(
            zip(task.dependencies, dependency_results),
            start=1,
        ):
            context.extend(
                [
                    "",
                    f"DEPENDENCY {index}:",
                    f"Tool: {dependency.tool}",
                    f"Arguments: {dependency.arguments}",
                    "",
                    "RESULT:",
                    self._format_result(result),
                ]
            )

        return "\n".join(context)

    def _effective_limit(self) -> int | None:
        """Limite vigente p/ UM resultado (Etapas 4+6)."""
        limit = self.max_result_chars
        if limit is None:
            return None
        try:
            from app.config import Config

            if bool(getattr(Config, "compact_executor", False)):
                return min(limit, self.COMPACT_MAX_RESULT_CHARS)
        except Exception:
            pass
        return limit

    def _format_result(self, result: Any) -> str:
        """Representação limitada de UM resultado (Etapas 4+6).

        Sem limite configurado (None): comportamento legado integral.
        Com limite: preserva o início (veredito STATUS, STDOUT inicial)
        e indica o total omitido. Nunca muta o resultado real.
        """

        text = result if isinstance(result, str) else str(result)
        limit = self._effective_limit()
        if limit is None or len(text) <= limit:
            return text
        omitted = len(text) - limit
        return (
            f"[truncated result: {len(text)} chars in total]\n"
            f"{text[:limit]}"
            f"\n... [+{omitted} chars omitted]"
        )