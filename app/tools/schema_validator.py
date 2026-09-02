import difflib
from typing import Any

from app.tools.base import Tool


# Mapeia os tipos do JSON Schema (usados nas `definition` das tools)
# para os tipos Python equivalentes, para permitir checagem de tipo.
JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}

# Abaixo desse score (0-1), uma sugestão de correção não é exibida
# por ser pouco confiável (ex.: nomes completamente diferentes).
SUGGESTION_CUTOFF = 0.5


class SchemaValidationError(ValueError):
    """Argumentos de uma tool não batem com o schema esperado.

    Levantada em Python puro, sem depender de LLM, para pegar erros
    comuns cometidos pelo modelo ao montar a chamada de uma tool —
    principalmente nomes de parâmetro inventados (ex.: `path` no lugar
    de `file_path`) — antes que a tool seja executada de fato.
    """


class SchemaValidator:
    """Valida argumentos de uma tool contra o JSON Schema da sua definition.

    Cada `Tool` já carrega sua própria definition (o mesmo JSON Schema
    exposto para a LLM em `tools.definitions`). Este validador reusa
    essa definition como fonte da verdade, então não há necessidade de
    manter um schema separado por tool.
    """

    def validate(self, tool: Tool, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise SchemaValidationError(
                f"Os argumentos da tool '{tool.name}' devem ser um "
                "objeto JSON (dict)."
            )

        schema = tool.definition.get("function", {}).get("parameters", {})
        properties: dict[str, Any] = schema.get("properties", {}) or {}
        required: list[str] = schema.get("required", []) or []
        known_names = list(properties.keys())

        self._check_unknown_arguments(tool.name, arguments, known_names)
        self._check_required_arguments(tool.name, arguments, required)
        self._check_types(tool.name, arguments, properties)

    def _check_unknown_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        known_names: list[str],
    ) -> None:
        for name in arguments:
            if name in known_names:
                continue

            suggestion = self._suggest(name, known_names)
            suggestion_text = (
                f" Você quis dizer '{suggestion}'?"
                if suggestion
                else ""
            )
            valid_list = ", ".join(known_names) if known_names else "(nenhum)"

            raise SchemaValidationError(
                f"Argumento desconhecido '{name}' para a tool "
                f"'{tool_name}'.{suggestion_text} "
                f"Argumentos válidos: {valid_list}."
            )

    def _check_required_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        required: list[str],
    ) -> None:
        missing = [name for name in required if name not in arguments]

        if missing:
            raise SchemaValidationError(
                f"Faltam argumentos obrigatórios para a tool "
                f"'{tool_name}': {', '.join(missing)}."
            )

    def _check_types(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        properties: dict[str, Any],
    ) -> None:
        for name, value in arguments.items():
            prop = properties.get(name)

            if not prop:
                continue

            expected_type = prop.get("type")
            python_type = JSON_TYPE_MAP.get(expected_type)

            if python_type is None:
                continue

            # bool é subclasse de int em Python; sem esse caso especial
            # um bool passaria despercebido como "integer"/"number".
            if isinstance(value, bool) and python_type is not bool:
                valid = False
            else:
                valid = isinstance(value, python_type)

            if not valid:
                raise SchemaValidationError(
                    f"Argumento '{name}' da tool '{tool_name}' deveria "
                    f"ser do tipo '{expected_type}', mas recebeu "
                    f"{type(value).__name__} ({value!r})."
                )

    def _suggest(
        self,
        name: str,
        known_names: list[str],
    ) -> str | None:
        matches = difflib.get_close_matches(
            name,
            known_names,
            n=1,
            cutoff=SUGGESTION_CUTOFF,
        )
        return matches[0] if matches else None