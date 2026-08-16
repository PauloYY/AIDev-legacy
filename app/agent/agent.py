from app.llm.client import LLMClient
from app.llm.models import Message
from app.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
    ):
        self.llm = llm
        self.tools = tools

    def run(self, prompt: str):
        messages = [
            Message(
                role="user",
                content=prompt,
            )
        ]

        response = self.llm.generate(
            messages,
            tools=self.tools.definitions,
        )

        for tool_call in response.tool_calls:
            result = self.tools.execute(
                tool_call.name,
                tool_call.arguments,
            )

            print("Tool result:", result)

        return response