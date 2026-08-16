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

    def run(self, prompt: str) -> str | None:
        messages = [
            Message(
                role="user",
                content=prompt,
            )
        ]

        while True:
            response = self.llm.generate(
                messages,
                tools=self.tools.definitions,
            )

            if not response.tool_calls:
                return response.content

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls
                )
            )

            for tool_call in response.tool_calls:
                result = self.tools.execute(
                    tool_call.name,
                    tool_call.arguments,
                )

                messages.append(
                    Message(
                        role="tool",
                        content=str(result),
                        tool_call_id=tool_call.id
                    )
                )