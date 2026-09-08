import re
import time
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

# Métricas (Fase 3): limites para não registrar conteúdo sensível/grande.
_MAX_COMMAND_CHARS = 200
_EXIT_CODE_RE = re.compile(r"exit code (-?\d+)")


def _summarize_command(arguments: dict[str, Any]) -> str | None:
    command = arguments.get("command")

    if not isinstance(command, str):
        return None

    if len(command) <= _MAX_COMMAND_CHARS:
        return command

    omitted = len(command) - _MAX_COMMAND_CHARS
    return f"{command[:_MAX_COMMAND_CHARS]}... [+{omitted} chars]"


class ToolRegistry:
    def __init__(self, tracker=None):
        self._tools: dict[str, Tool] = {}
        # Fase 3: tracker opcional (UsageTracker). None = sem overhead de
        # métricas além do cronômetro; comportamento inalterado.
        self._tracker = tracker

    def set_tracker(self, tracker) -> None:
        self._tracker = tracker

    def tool_stats(self) -> dict[str, Any]:
        if self._tracker is None:
            return {"calls": 0, "total_ms": 0.0, "failures": 0,
                    "timeouts": 0, "by_tool": {}}
        return self._tracker.tool_stats()

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
                f"Tool not found: {name}"
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
            if self._tracker is not None:
                self._tracker.record_tool(
                    name, 0.0, False, error="ValueError",
                )
            raise ValueError(
                f"Tool not found: {name}"
            )

        start = time.monotonic()
        try:
            result = self._tools[name].execute(arguments)
        except Exception as error:
            if self._tracker is not None:
                self._tracker.record_tool(
                    name,
                    (time.monotonic() - start) * 1000,
                    False,
                    error=type(error).__name__,
                    command=_summarize_command(arguments)
                    if name == "run_command"
                    else None,
                )
            raise

        if self._tracker is not None:
            timeout, exit_code, command = False, None, None
            if name == "run_command" and isinstance(result, str):
                timeout = result.startswith("TIMEOUT:")
                match = _EXIT_CODE_RE.search(result)
                exit_code = int(match.group(1)) if match else None
                command = _summarize_command(arguments)

            self._tracker.record_tool(
                name,
                (time.monotonic() - start) * 1000,
                True,
                timeout=timeout,
                exit_code=exit_code,
                command=command,
            )

        return result