from typing import Any

from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task


class TaskContextBuilder:
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
            "TASK PAI:",
            f"Tool: {task.tool}",
            f"Arguments: {task.arguments}",
            "",
            "DEPENDÊNCIAS:",
        ]

        if not task.dependencies:
            context.append("Nenhuma.")

        for index, (dependency, result) in enumerate(
            zip(task.dependencies, dependency_results),
            start=1,
        ):
            context.extend(
                [
                    "",
                    f"DEPENDÊNCIA {index}:",
                    f"Tool: {dependency.tool}",
                    f"Arguments: {dependency.arguments}",
                    "",
                    "RESULTADO:",
                    str(result),
                ]
            )

        return "\n".join(context)