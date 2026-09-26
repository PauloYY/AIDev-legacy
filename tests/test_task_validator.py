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
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )

    validator.validate(task)


def test_validate_rejects_unknown_tool(validator):
    task = Task(tool="delete_universe", arguments={})

    with pytest.raises(ValueError):
        validator.validate(task)


def test_validate_rejects_non_analysis_tool_as_dependency(validator):
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
        dependencies=[
            Dependency(tool="write_file", arguments={"file_path": "a.py", "content": "x"})
        ],
    )

    with pytest.raises(ValueError):
        validator.validate(task)


def test_validate_accepts_analysis_tool_as_dependency(validator):
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
        dependencies=[
            Dependency(tool="read_file", arguments={"project_name": "p", "file_path": "a.py"})
        ],
    )

    validator.validate(task)


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("read_file", {"project_name": "p", "file_path": "a.py"}),
        ("list_files", {"project_name": "p"}),
        ("find_references", {"project_name": "p", "symbol": "foo"}),
    ],
)
def test_validate_rejects_investigation_tool_as_main_task(
    validator, tool_name, arguments
):
    task = Task(tool=tool_name, arguments=arguments)

    with pytest.raises(ValueError):
        validator.validate(task)


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("read_file", {"project_name": "p", "file_path": "a.py"}),
        ("list_files", {"project_name": "p"}),
        ("find_references", {"project_name": "p", "symbol": "foo"}),
    ],
)
def test_validate_accepts_investigation_tool_when_allowed(
    validator, tool_name, arguments
):
    task = Task(tool=tool_name, arguments=arguments)

    validator.validate(task, allow_investigation=True)


def test_validate_accepts_check_project_and_run_command_as_main_task(validator):
    validator.validate(
        Task(tool="check_project", arguments={"project_name": "p"})
    )
    validator.validate(
        Task(
            tool="run_command",
            arguments={"project_name": "p", "command": "npm test"},
        )
    )


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("read_file", {"project_name": "p", "file_path": "a.py"}),
        ("list_files", {"project_name": "p"}),
        ("find_references", {"project_name": "p", "symbol": "foo"}),
    ],
)
def test_validate_accepts_investigation_tool_with_investigation_flag(
    validator, tool_name, arguments
):
    task = Task(tool=tool_name, arguments=arguments, investigation=True)
    validator.validate(task)


def test_validate_rejects_investigation_tool_without_flag():
    """Uma task com read_file e investigation=False (ou omitido) deve
    continuar sendo rejeitada — a proteção existente não é removida."""
    tools = ToolRegistry()
    tools.load_defaults()
    validator = TaskValidator(tools)

    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        investigation=False,
    )

    with pytest.raises(ValueError):
        validator.validate(task)


def test_validate_investigation_flag_does_not_affect_non_investigation_tools(
    validator,
):
    """O flag investigation só libera DEPENDENCY_ONLY_TOOLS. Tools de
    execução (write_file, run_command) continuam funcionando normalmente
    mesmo sem o flag."""
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
        investigation=True,
    )
    validator.validate(task)
