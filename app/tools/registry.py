from typing import Any

from app.tools.base import Tool
from app.tools.analysis import TOOLS as ANALYSIS_TOOLS
from app.tools.execution import TOOLS as EXECUTION_TOOLS
from app.tools.filesystem import TOOLS as FILESYSTEM_TOOLS


DEFAULT_TOOLS = [
    *FILESYSTEM_TOOLS,
    *EXECUTION_TOOLS,
    *ANALYSIS_TOOLS,
]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def load_defaults(self):
        for tool in DEFAULT_TOOLS:
            self.register(tool)

    @property
    def definitions(self) -> list[dict]:
        return [
            tool.definition
            for tool in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(
                f"Tool não encontrada: {name}"
            )

        return self._tools[name]

    def exists(self, name: str) -> bool:
        return name in self._tools

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if name not in self._tools:
            raise ValueError(
                f"Tool não encontrada: {name}"
            )

        return self._tools[name].execute(arguments)