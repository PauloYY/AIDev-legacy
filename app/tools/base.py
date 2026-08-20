from typing import Any, Callable


class Tool:
    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        definition: dict,
    ):
        self.name = name
        self.function = function
        self.definition = definition

    def execute(self, arguments: dict[str, Any]) -> Any:
        return self.function(**arguments)