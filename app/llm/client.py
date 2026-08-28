from app.llm.models import LLMResponse, Message
from app.llm.providers.base import LLMProvider
from app.llm.usage import UsageTracker


class LLMClient:

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.usage = UsageTracker()

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        response = self.provider.generate(
            messages,
            tools,
        )

        self.usage.record(response.usage, response.provider)

        return response
