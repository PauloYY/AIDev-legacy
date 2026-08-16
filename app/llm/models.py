from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    
@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]