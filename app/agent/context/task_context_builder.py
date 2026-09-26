from typing import Any

from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task


class TaskContextBuilder:
    DEFAULT_MAX_RESULT_CHARS = 4000

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
        """Limite vigente p/ UM resultado."""
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
        """Representação limitada de UM resultado.

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
