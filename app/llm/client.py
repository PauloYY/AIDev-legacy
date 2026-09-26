import time

from app.llm.models import LLMResponse, Message, Usage
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
        context_breakdown: dict[str, dict[str, int]] | None = None,
        attempt: int = 1,
        request_type: str = "normal",
    ) -> LLMResponse:

        prompt_text = ""
        for msg in messages:
            if msg.content:
                prompt_text += msg.content

        start = time.monotonic()
        try:
            response = self.provider.generate(
                messages,
                tools,
            )
        except Exception as error:
            duration_ms = (time.monotonic() - start) * 1000
            self.usage.record(
                Usage(),
                None,
                component=component,
                iteration=iteration,
                prompt_chars=len(prompt_text),
                completion_chars=0,
                context_breakdown=context_breakdown,
                duration_ms=duration_ms,
                attempt=attempt,
                success=False,
                error=type(error).__name__,
                request_type=request_type,
            )
            raise

        duration_ms = (time.monotonic() - start) * 1000

        self.usage.record(
            response.usage,
            response.provider,
            component=component,
            iteration=iteration,
            prompt_chars=len(prompt_text),
            completion_chars=len(response.content or ""),
            context_breakdown=context_breakdown,
            duration_ms=duration_ms,
            attempt=attempt,
            empty_response=not (
                response.content and response.content.strip()
            ),
            request_type=request_type,
        )

        return response

    async def generate_async(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        component: str | None = None,
        iteration: int | None = None,
        context_breakdown: dict[str, dict[str, int]] | None = None,
        attempt: int = 1,
        request_type: str = "normal",
    ) -> LLMResponse:
        """Variante assíncrona do `generate`.

        Espelho exato: mesma medição de tempo, mesmo registro no
        UsageTracker (incluindo `request_type`, breakdown e registro
        em caso de exceção) e mesma propagação de erros. O `generate`
        síncrono segue intacto e é o usado pelo agente por padrão."""

        prompt_text = ""
        for msg in messages:
            if msg.content:
                prompt_text += msg.content

        start = time.monotonic()
        try:
            response = await self.provider.generate_async(
                messages,
                tools,
            )
        except Exception as error:
            duration_ms = (time.monotonic() - start) * 1000
            self.usage.record(
                Usage(),
                None,
                component=component,
                iteration=iteration,
                prompt_chars=len(prompt_text),
                completion_chars=0,
                context_breakdown=context_breakdown,
                duration_ms=duration_ms,
                attempt=attempt,
                success=False,
                error=type(error).__name__,
                request_type=request_type,
            )
            raise

        duration_ms = (time.monotonic() - start) * 1000

        self.usage.record(
            response.usage,
            response.provider,
            component=component,
            iteration=iteration,
            prompt_chars=len(prompt_text),
            completion_chars=len(response.content or ""),
            context_breakdown=context_breakdown,
            duration_ms=duration_ms,
            attempt=attempt,
            empty_response=not (
                response.content and response.content.strip()
            ),
            request_type=request_type,
        )

        return response
