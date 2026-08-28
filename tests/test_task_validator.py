import pytest

from app.agent.execution.task import Task
from app.agent.planning.dependency import Dependency
from app.agent.execution.validator import TaskValidator
from app.tools.registry import ToolRegistry


@pytest.fixture
def validator():
    tools = ToolRegistry()
    tools.load_defaults()
    return TaskValidator(tools)


def test_validate_accepts_known_tool(validator):
    task = Task(tool="write_file", arguments={"file_path": "a.py", "content": "x"})

    validator.validate(task)  # não deve lançar


def test_validate_rejects_unknown_tool(validator):
    task = Task(tool="delete_universe", arguments={})

    with pytest.raises(ValueError):
        validator.validate(task)


def test_validate_rejects_non_analysis_tool_as_dependency(validator):
    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        dependencies=[
            Dependency(tool="write_file", arguments={"file_path": "a.py", "content": "x"})
        ],
    )

    with pytest.raises(ValueError):
        validator.validate(task)


def test_validate_accepts_analysis_tool_as_dependency(validator):
    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        dependencies=[
            Dependency(tool="list_files", arguments={"project_name": "p"})
        ],
    )

    validator.validate(task)  # não deve lançar
