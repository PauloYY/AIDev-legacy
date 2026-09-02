import json

from app.agent.planning.decision import Decision, DecisionAction
from app.agent.planning.dependency import Dependency
from app.agent.execution.task import Task
from app.llm.json_extraction import parse_json_object
from app.tools.registry import ToolRegistry


class DecisionParser:
    def __init__(self, tools: ToolRegistry | None = None):
        # Opcional para não quebrar quem instancia sem tools (ex.:
        # testes existentes que só testam o parsing "normal"). Sem
        # tools, a auto-recuperação abaixo fica desativada.
        self.tools = tools

    def parse(self, content: str) -> Decision:
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError as exc:
            raise ValueError("A LLM retornou um JSON inválido.") from exc

        action = data.get("action")

        if action == "task":
            return self._parse_task_decision(data)

        if action == "finish":
            return Decision(
                action=DecisionAction.FINISH,
                content=data.get("content"),
            )

        if action == "fail":
            return Decision(
                action=DecisionAction.FAIL,
                reason=data.get("reason"),
            )

        recovered = self._recover_task_from_tool_action(action, data)

        if recovered is not None:
            return recovered

        raise ValueError(
            f"Ação de decisão inválida: {action!r}"
        )

    def _recover_task_from_tool_action(
        self,
        action,
        data: dict,
    ) -> Decision | None:
        """Recupera decisões malformadas onde a LLM colocou o nome de
        uma tool diretamente em 'action' (ex.: {"action": "run_command",
        ...}) em vez de aninhar corretamente em 'task.tool'.

        Esse padrão específico de erro tende a se repetir de forma
        consistente quando acontece (a LLM "trava" na mesma confusão
        por várias tentativas seguidas), o que já causou o agente
        estourar MAX_PLANNER_ATTEMPTS e crashar. Em vez de tratar como
        erro fatal, interpretamos a intenção óbvia e seguimos.
        """

        if not self.tools or not isinstance(action, str):
            return None

        if not self.tools.exists(action):
            return None

        arguments = data.get("arguments")

        if not isinstance(arguments, dict):
            task_data = data.get("task")
            arguments = (
                task_data.get("arguments", {})
                if isinstance(task_data, dict)
                else {}
            )

        return Decision(
            action=DecisionAction.TASK,
            task=Task(
                tool=action,
                arguments=arguments,
                dependencies=[],
            ),
        )

    def _parse_task_decision(self, data: dict) -> Decision:
        task_data = data.get("task")

        if not isinstance(task_data, dict):
            raise ValueError(
                "Uma decisão 'task' deve possuir um objeto 'task'."
            )

        dependencies = [
            Dependency(
                tool=dependency["tool"],
                arguments=dependency.get("arguments", {}),
            )
            for dependency in task_data.get("dependencies", [])
        ]

        task = Task(
            tool=task_data["tool"],
            arguments=task_data.get("arguments", {}),
            dependencies=dependencies,
        )

        return Decision(
            action=DecisionAction.TASK,
            task=task,
        )