import json
import logging
import time

import httpx

from app.config import Config
from app.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMInvalidResponseError,
)
from app.llm.models import LLMResponse, Message, ToolCall, Usage
from app.llm.providers.base import LLMProvider


logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):

    MAX_RETRIES = 3
    BACKOFF_SECONDS = 1.5
    TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        name: str | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.name = name or self.__class__.__name__

    def _serialize_message(self, message: Message) -> dict:
        data = {
            "role": message.role,
            "content": message.content,
        }

        if message.role == "tool":
            data["tool_call_id"] = message.tool_call_id

        if message.role == "assistant" and message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tool.id,
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "arguments": json.dumps(tool.arguments),
                    },
                }
                for tool in message.tool_calls
            ]

        return data

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                self._serialize_message(message)
                for message in messages
            ],
            "max_tokens": Config.llm_max_output_tokens,
        }

        if tools:
            payload["tools"] = tools

        response = self._post_with_retry(headers, payload)

        if response.is_error:
            self._raise_for_error(response)

        data = response.json()
        message_data = data["choices"][0]["message"]

        tool_calls = []

        for tool_call in message_data.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=json.loads(
                        tool_call["function"]["arguments"]
                    ),
                )
            )

        usage_data = data.get("usage") or {}

        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=message_data.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            provider=self.name,
        )

    def _post_with_retry(
        self,
        headers: dict,
        payload: dict,
    ) -> httpx.Response:
        """Faz POST com retry/backoff para falhas de rede (timeout, conexão).

        Erros de aplicação (4xx/5xx retornados pela API) NÃO são
        re-tentados aqui — isso é responsabilidade do LLMRouter, que
        decide se vale a pena trocar de provider ou esperar.
        """

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return httpx.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.TIMEOUT_SECONDS,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as error:
                last_error = error

                if attempt >= self.MAX_RETRIES:
                    break

                wait = self.BACKOFF_SECONDS * (2 ** (attempt - 1))

                logger.warning(
                    "Falha de rede ao chamar %s (tentativa %d/%d): %s. "
                    "Tentando novamente em %.1fs...",
                    self.name,
                    attempt,
                    self.MAX_RETRIES,
                    error,
                    wait,
                )

                time.sleep(wait)

        raise LLMConnectionError(
            f"Falha de conexão com {self.name} após "
            f"{self.MAX_RETRIES} tentativas: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Lê o header padrão HTTP 'Retry-After' (em segundos), se presente."""

        header = response.headers.get("Retry-After")

        if not header:
            return None

        try:
            return float(header)
        except ValueError:
            return None

    def _raise_for_error(self, response: httpx.Response) -> None:
        try:
            error = response.json().get("error", {})
            message = error.get("message", response.text)
            code = error.get("code", response.status_code)

            metadata = error.get("metadata", {})
            raw = metadata.get("raw")

            if raw:
                try:
                    raw_error = json.loads(raw)
                    message = raw_error.get("message", message)
                except (json.JSONDecodeError, TypeError):
                    pass

        except (ValueError, AttributeError):
            message = response.text
            code = response.status_code

        if response.status_code == 429 or code == 429:
            retry_after = self._parse_retry_after(response)
            raise LLMRateLimitError(message, retry_after=retry_after)

        if "rate_limit" in str(message).lower():
            # Alguns providers (ex.: Agnes) devolvem rate limit como
            # HTTP 500 genérico em vez de 429, com o motivo real só no
            # corpo da mensagem (ex.: "rate_limit_check_failed"). Sem
            # essa checagem, isso seria tratado como "erro transitório
            # genérico" — funcionalmente similar (também cai no
            # fallback), mas com um log confuso e sem chance de usar
            # um eventual header Retry-After.
            retry_after = self._parse_retry_after(response)
            raise LLMRateLimitError(message, retry_after=retry_after)

        if "tool choice is none" in str(message).lower():
            raise LLMInvalidResponseError(message)

        raise LLMAPIError(message, status_code=response.status_code)