import json

from app.agent.decision import Decision, DecisionAction
from app.agent.dependency import Dependency
from app.agent.task import Task


class DecisionParser:
    def parse(self, content: str) -> Decision:
        try:
            data = json.loads(content)
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

        raise ValueError(
            f"Ação de decisão inválida: {action!r}"
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