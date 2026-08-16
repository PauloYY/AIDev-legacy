import httpx
import json

from app.config import Config
from app.llm.models import LLMResponse, Message, ToolCall

class LLMClient:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.api_key = Config.llm_api_key
        self.model = Config.llm_model

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        }
        
        if tools:
            payload["tools"] = tools

        response = httpx.post(
            self.BASE_URL,
            headers=headers,
            json=payload,
            timeout=60.0,
        )

        response.raise_for_status()
        data = response.json()
        message_data = data["choices"][0]["message"]

        tool_calls = []

        for tool_call in message_data.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=json.loads(tool_call["function"]["arguments"])
                )
            )

        return LLMResponse(
            content=message_data.get("content"),
            tool_calls=tool_calls
        )
