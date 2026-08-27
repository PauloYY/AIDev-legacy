import httpx
import json

from app.exceptions import (
    LLMAPIError,
    LLMRateLimitError,
    LLMInvalidResponseError,
)
from app.llm.models import LLMResponse, Message, ToolCall
from app.llm.providers.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

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
        }

        if tools:
            payload["tools"] = tools

        response = httpx.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60.0,
        )

        if response.is_error:
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
                raise LLMRateLimitError(message)

            if "tool choice is none" in message.lower():
                raise LLMInvalidResponseError(message)

            raise LLMAPIError(message)

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

        return LLMResponse(
            content=message_data.get("content"),
            tool_calls=tool_calls,
        )