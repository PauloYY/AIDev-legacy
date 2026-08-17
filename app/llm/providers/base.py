from abc import ABC, abstractmethod

from app.llm.models import LLMResponse, Message


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        pass