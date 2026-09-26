from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task
from app.tools.base import Tool, ToolType
from app.tools.registry import ToolRegistry
from app.tools.schema_validator import SchemaValidator


class TaskValidator:

    DEPENDENCY_ONLY_TOOLS = {"read_file", "list_files", "find_references"}

    def __init__(
        self,
        tools: ToolRegistry,
        schema_validator: SchemaValidator | None = None,
    ):
        self.tools = tools
        self.schema_validator = schema_validator or SchemaValidator()

    def validate(self, task: Task, allow_investigation: bool = False) -> None:
        self.validate_arguments(
            task.tool,
            task.arguments,
            allow_investigation=allow_investigation or task.investigation,
        )

        for dependency in task.dependencies:
            self._validate_dependency(dependency)

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict,
        allow_investigation: bool = False,
    ) -> None:
        """Valida tool + argumentos contra o schema real da tool.

        Usado tanto para a task planejada quanto, separadamente, para a
        decisão final do executor — é nesse segundo ponto que erros como
        um parâmetro inventado (`path` em vez de `file_path`) mais
        aparecem, já que é a LLM executora quem monta os argumentos
        finais antes da tool ser chamada de verdade.

        `allow_investigation=True` libera read_file/list_files/
        find_references como task principal — usado pelo Runner
        especificamente logo após um run_command de teste/build que
        falhou, quando o Planner precisa investigar o que quebrou
        antes de conseguir corrigir. Fora desse caso, a proibição
        normal se aplica.
        """

        tool = self._validate_tool(tool_name)

        if (
            tool_name in self.DEPENDENCY_ONLY_TOOLS
            and not allow_investigation
        ):
            raise ValueError(
                f"The tool '{tool_name}' can only be used as a "
                "dependency, not as the main task. It gathers "
                "information before a real action — attach it in "
                '"dependencies" of the task that really needs that '
                "information (e.g. a write_file that edits the file "
                "you read), instead of creating a task just to "
                "investigate. "
                "Exception: right after a failed test/build run_command, "
                "you may use it alone to investigate what broke. "
                "Another exception: if your objective is to "
                "analyze or investigate an existing project (without "
                "creating or modifying anything), mark the task with "
                '"investigation": true'
                " — this explicitly allows read_file/list_files/"
                "find_references as the main action."
            )

        self.schema_validator.validate(tool, arguments)

    def _validate_dependency(
        self,
        dependency: Dependency,
    ) -> None:
        tool = self._validate_tool(dependency.tool)

        if tool.type != ToolType.ANALYSIS:
            raise ValueError(
                f"The tool '{dependency.tool}' "
                "cannot be used as a dependency."
            )

        self.schema_validator.validate(tool, dependency.arguments)

    def _validate_tool(self, name: str) -> Tool:
        return self.tools.get(name)
