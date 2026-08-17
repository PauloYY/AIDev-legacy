from app.exceptions import LLMRateLimitError
from app.llm.models import LLMResponse, Message
from app.llm.providers.base import LLMProvider


class LLMRouter:

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        last_error = None

        for provider in self.providers:
            try:
                return provider.generate(
                    messages,
                    tools,
                )

            except LLMRateLimitError as error:
                last_error = error
                continue

        if last_error:
            raise last_error

        raise RuntimeError("Nenhum provider disponível.")