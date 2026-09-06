from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task
from app.tools.base import Tool, ToolType
from app.tools.registry import ToolRegistry
from app.tools.schema_validator import SchemaValidator


class TaskValidator:

    # Ferramentas de pura investigação: fazem sentido como dependency
    # (reunir informação antes de agir), mas por padrão não como task
    # principal — deixar o Planner usá-las sozinhas livremente tende
    # a abrir ciclos de investigação sem progresso real. A exceção é
    # controlada pelo Runner via `allow_investigation` (ver abaixo).
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
                f"A tool '{tool_name}' só pode ser usada como "
                "dependency, não como task principal. Ela serve para "
                "reunir informação antes de uma ação real — anexe-a em "
                '"dependencies" da task que realmente precisa dessa '
                "informação (ex.: um write_file que edita o arquivo "
                "lido), em vez de criar uma task só para investigar. "
                "Exceção: logo após um run_command de teste/build que "
                "falhou, você pode usá-la sozinha para investigar o "
                "que quebrou. Outra exceção: se o seu objetivo for "
                "analisar ou investigar um projeto existente (sem "
                "criar ou modificar nada), marque a task com "
                '"investigation": true'
                " — isso permite usar read_file/list_files/find_references "
                "como ação principal de forma explícita."
            )

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