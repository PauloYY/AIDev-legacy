from app.llm.models import LLMResponse, Message
from app.llm.providers.base import LLMProvider


class LLMClient:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        return self.provider.generate(
            messages,
            tools,
        )