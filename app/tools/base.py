from typing import Any, Callable
from enum import Enum


class ToolType(Enum):
    # ATENÇÃO: ANALYSIS **não** significa "sem efeitos colaterais".
    # Significa apenas "pode aparecer como dependency" (ver
    # TaskValidator._validate_dependency). Ex.: run_command é ANALYSIS
    # mas pode mutar arquivos via shell — por isso a pureza para o
    # executor paralelo é um atributo separado (`Tool.pure`).
    ANALYSIS = "analysis"
    EXECUTION = "execution"


class Tool:
    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        definition: dict,
        type: ToolType,
        pure: bool = False,
    ):
        self.name = name
        self.function = function
        self.definition = definition
        self.type = type
        # Pureza p/ paralelização (P4): True somente se a tool é
        # comprovadamente livre de efeitos colaterais relevantes,
        # independentemente dos argumentos. Default False
        # (conservador): em caso de dúvida, sequencial. NUNCA inferir
        # pureza do conteúdo de um comando shell — run_command é
        # sempre não-puro, mesmo para comandos com cara de leitura.
        self.pure = pure

    def execute(self, arguments: dict[str, Any]) -> Any:
        return self.function(**arguments)