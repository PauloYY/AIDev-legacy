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
        component: str | None = None,
        iteration: int | None = None,
    ) -> LLMResponse:

        response = self.provider.generate(
            messages,
            tools,
        )

        prompt_text = ""
        for msg in messages:
            if msg.content:
                prompt_text += msg.content

        self.usage.record(
            response.usage,
            response.provider,
            component=component,
            iteration=iteration,
            prompt_chars=len(prompt_text),
            completion_chars=len(response.content or ""),
        )

        return response
