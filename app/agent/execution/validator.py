from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task
from app.tools.base import Tool, ToolType
from app.tools.registry import ToolRegistry
from app.tools.schema_validator import SchemaValidator


class TaskValidator:
    def __init__(
        self,
        tools: ToolRegistry,
        schema_validator: SchemaValidator | None = None,
    ):
        self.tools = tools
        self.schema_validator = schema_validator or SchemaValidator()

    def validate(self, task: Task) -> None:
        self.validate_arguments(task.tool, task.arguments)

        for dependency in task.dependencies:
            self._validate_dependency(dependency)

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict,
    ) -> None:
        """Valida tool + argumentos contra o schema real da tool.

        Usado tanto para a task planejada quanto, separadamente, para a
        decisão final do executor — é nesse segundo ponto que erros como
        um parâmetro inventado (`path` em vez de `file_path`) mais
        aparecem, já que é a LLM executora quem monta os argumentos
        finais antes da tool ser chamada de verdade.
        """

        tool = self._validate_tool(tool_name)
        self.schema_validator.validate(tool, arguments)

    def _validate_dependency(
        self,
        dependency: Dependency,
    ) -> None:
        tool = self._validate_tool(dependency.tool)

        if tool.type != ToolType.ANALYSIS:
            raise ValueError(
                f"A tool '{dependency.tool}' "
                "não pode ser usada como dependency."
            )

        self.schema_validator.validate(tool, dependency.arguments)

    def _validate_tool(self, name: str) -> Tool:
        return self.tools.get(name)