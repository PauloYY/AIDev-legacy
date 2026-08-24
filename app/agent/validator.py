from app.agent.dependency import Dependency
from app.agent.task import Task
from app.tools.base import ToolType
from app.tools.registry import ToolRegistry


class TaskValidator:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def validate(self, task: Task) -> None:
        self._validate_tool(task.tool)
        self._validate_arguments(task.arguments)

        for dependency in task.dependencies:
            self._validate_dependency(dependency)

    def _validate_dependency(
        self,
        dependency: Dependency,
    ) -> None:
        tool = self.tools.get(dependency.tool)

        if tool.type != ToolType.ANALYSIS:
            raise ValueError(
                f"A tool '{dependency.tool}' "
                "não pode ser usada como dependency."
            )

        self._validate_arguments(dependency.arguments)

    def _validate_tool(self, name: str) -> None:
        self.tools.get(name)

    def _validate_arguments(
        self,
        arguments: dict,
    ) -> None:
        if not isinstance(arguments, dict):
            raise ValueError(
                "Os argumentos da task devem ser um objeto."
            )