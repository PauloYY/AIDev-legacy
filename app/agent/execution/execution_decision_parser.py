import json

from app.agent.execution.execution_decision import ExecutionDecision
from app.llm.json_extraction import parse_json_object


class ExecutionDecisionParser:

    def parse(
        self,
        content: str,
    ) -> ExecutionDecision:
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The LLM returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "The decision must be a JSON object."
            )

        tool = data.get("tool")
        arguments = data.get("arguments")

        if not isinstance(tool, str):
            raise ValueError(
                "The decision must contain a valid tool."
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                "The decision must contain arguments as an object."
            )

        return ExecutionDecision(
            tool=tool,
            arguments=arguments,
        )