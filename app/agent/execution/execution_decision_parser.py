import json
from app.llm.json_extraction import extract_json_object

from app.agent.execution.execution_decision import ExecutionDecision


class ExecutionDecisionParser:

    def parse(
        self,
        content: str,
    ) -> ExecutionDecision:
        try:
            data = json.loads(extract_json_object(content))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "A LLM retornou JSON inválido."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "A decisão deve ser um objeto JSON."
            )

        tool = data.get("tool")
        arguments = data.get("arguments")

        if not isinstance(tool, str):
            raise ValueError(
                "A decisão deve conter uma tool válida."
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                "A decisão deve conter arguments como objeto."
            )

        return ExecutionDecision(
            tool=tool,
            arguments=arguments,
        )