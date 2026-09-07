import asyncio
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

    async def generate_async(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Variante assíncrona (Etapa 3). Default: roda o `generate`
        síncrono numa thread, sem bloquear o event loop. Providers com
        I/O HTTP nativamente assíncrono (ex.: httpx.AsyncClient)
        sobrescrevem com implementação real. O caminho síncrono segue
        intacto e é o usado pelo agente por padrão."""

        return await asyncio.to_thread(self.generate, messages, tools)