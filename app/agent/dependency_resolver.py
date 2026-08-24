from typing import Any

from app.agent.dependency import Dependency
from app.tools.registry import ToolRegistry


class DependencyResolver:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def resolve(
        self,
        dependencies: list[Dependency],
    ) -> list[Any]:
        results = []

        for dependency in dependencies:
            result = self.tools.execute(
                dependency.tool,
                dependency.arguments,
            )

            results.append(result)

        return results