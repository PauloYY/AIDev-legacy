import pytest

from app.tools.base import Tool, ToolType
from app.tools.schema_validator import SchemaValidator, SchemaValidationError


DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Lê um arquivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "file_path": {"type": "string"},
            },
            "required": ["project_name", "file_path"],
        },
    },
}


@pytest.fixture
def tool():
    return Tool(
        name="read_file",
        function=lambda **kwargs: kwargs,
        definition=DEFINITION,
        type=ToolType.ANALYSIS,
    )


@pytest.fixture
def validator():
    return SchemaValidator()


def test_accepts_valid_arguments(validator, tool):
    validator.validate(
        tool,
        {"project_name": "p", "file_path": "a.py"},
    )  # não deve lançar


def test_rejects_invented_argument_with_suggestion(validator, tool):
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate(
            tool,
            {"project_name": "p", "path": "a.py"},
        )

    message = str(exc_info.value)
    assert "path" in message
    assert "file_path" in message


def test_rejects_missing_required_argument(validator, tool):
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate(tool, {"project_name": "p"})

    assert "file_path" in str(exc_info.value)


def test_rejects_wrong_type(validator, tool):
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate(
            tool,
            {"project_name": "p", "file_path": 123},
        )

    assert "file_path" in str(exc_info.value)


def test_rejects_non_dict_arguments(validator, tool):
    with pytest.raises(SchemaValidationError):
        validator.validate(tool, ["not", "a", "dict"])