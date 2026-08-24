from typing import Any, Callable
from enum import Enum


class ToolType(Enum):
    ANALYSIS = "analysis"
    EXECUTION = "execution"


class Tool:
    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        definition: dict,
        type: ToolType
    ):
        self.name = name
        self.function = function
        self.definition = definition
        self.type = type

    def execute(self, arguments: dict[str, Any]) -> Any:
        return self.function(**arguments)