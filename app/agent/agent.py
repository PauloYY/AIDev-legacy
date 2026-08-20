from app.agent.events import AgentEvent
from app.llm.client import LLMClient
from app.llm.models import Message
from app.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        on_event=None,
    ):
        self.llm = llm
        self.tools = tools
        self.on_event = on_event

    def _emit(self, event_type: str, **data):
        if self.on_event:
            self.on_event(
                AgentEvent(
                    type=event_type,
                    data=data,
                )
            )

    def run(self, prompt: str) -> str | None:
        self._emit("agent_start")
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

            self._emit(
                "llm_response",
                tool_calls=len(response.tool_calls),
            )

            if not response.tool_calls:
                self._emit("agent_done")
                return response.content

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                self._emit(
                    "tool_start",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )

                result = self.tools.execute(
                    tool_call.name,
                    tool_call.arguments,
                )

                self._emit(
                    "tool_end",
                    name=tool_call.name,
                )

                messages.append(
                    Message(
                        role="tool",
                        content=str(result),
                        tool_call_id=tool_call.id,
                    )
                )