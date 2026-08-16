from typing import Any, Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}
        self._definitions: list[dict] = []

    def register(
        self,
        name: str,
        function: Callable[..., Any],
        definition: dict,
    ):
        self._tools[name] = function
        self._definitions.append(definition)

    @property
    def definitions(self) -> list[dict]:
        return self._definitions

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if name not in self._tools:
            raise ValueError(
                f"Tool não encontrada: {name}"
            )

        return self._tools[name](**arguments)