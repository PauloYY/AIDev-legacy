"""P4 — classificação do run_command (pureza p/ paralelização).

Cobre: run_command de leitura/teste/mutante/ambíguo sempre não-puro,
tools puras inalteradas, gate do batch, e coerência entre a declaração
(`Tool.pure`), `PURE_READ_TOOLS` e `Runner.READ_ONLY_TOOLS`.
"""

import pytest

from app.agent.parallel import (
    PURE_READ_TOOLS,
    all_pure_read,
    declared_pure_tool_names,
    is_pure_read_tool,
)
from app.agent.runner import Runner
from app.tools.base import ToolType
from app.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


# run_command claramente de leitura — AINDA ASSIM não-puro ---------------
@pytest.mark.parametrize("command", [
    "python3 --version",
    "ls",
    "cat notes.py",
    "node --check app.js",
])
def test_read_like_run_command_is_not_pure(registry, command):
    """Comandos com cara de leitura não entram em batch (sem sniffing)."""
    tool = registry.get("run_command")
    assert tool.pure is False
    assert tool.type == ToolType.ANALYSIS  # ainda pode ser dependency
    assert is_pure_read_tool("run_command") is False
    assert all_pure_read(["read_file", "run_command"]) is False


# run_command de teste/build — conservador --------------------------------
@pytest.mark.parametrize("command", [
    "python -m pytest -q",
    "npm test",
    "npm run build",
    "go test ./...",
])
def test_test_build_run_command_is_not_pure(command):
    assert is_pure_read_tool("run_command") is False
    assert all_pure_read(["run_command"]) is False


# run_command que modifica arquivos ----------------------------------------
@pytest.mark.parametrize("command", [
    "cat > out.txt <<'EOF'\nhi\nEOF",
    "echo x >> notes.txt",
    "mkdir -p data",
    "rm -rf build",
    "touch a.py",
])
def test_mutating_run_command_is_not_pure(command):
    assert is_pure_read_tool("run_command") is False


# comando ambíguo — em dúvida, sequencial ----------------------------------
@pytest.mark.parametrize("command", [
    "echo hello",
    "grep -r foo .",
    "python3 script.py",
    "",
])
def test_ambiguous_run_command_is_not_pure(command):
    assert is_pure_read_tool("run_command") is False
    assert all_pure_read(["read_file", "run_command"]) is False


# tools puras inalteradas ----------------------------------------------------
@pytest.mark.parametrize("name", [
    "read_file", "list_files", "find_references", "list_symbols",
])
def test_pure_tools_stay_pure(registry, name):
    tool = registry.get(name)
    assert tool.pure is True
    assert tool.type == ToolType.ANALYSIS
    assert is_pure_read_tool(name) is True


def test_non_pure_tools(registry):
    for name in ("write_file", "run_command", "check_project"):
        assert registry.get(name).pure is False
        assert is_pure_read_tool(name) is False


def test_tool_pure_defaults_to_false():
    # Construtor sem `pure` mantém o default conservador.
    from app.tools.base import Tool, ToolType
    legacy = Tool(name="x", function=lambda: None, definition={},
                  type=ToolType.EXECUTION)
    assert legacy.pure is False


# gate do batch ---------------------------------------------------------------
def test_batch_gate():
    assert all_pure_read(["read_file", "list_files"]) is True
    assert all_pure_read(["read_file"]) is True
    assert all_pure_read(["read_file", "run_command"]) is False
    assert all_pure_read(["run_command", "run_command"]) is False
    assert all_pure_read(["read_file", "check_project"]) is False
    assert all_pure_read([]) is False
    assert all_pure_read(["unknown_tool"]) is False
    assert all_pure_read([None, 123]) is False


# coerência --------------------------------------------------------------------
def test_coherence_between_declaration_and_parallel_gate(registry):
    """PURE_READ_TOOLS espelha exatamente as tools com pure=True."""
    assert declared_pure_tool_names(registry) == set(PURE_READ_TOOLS)


def test_coherence_with_runner_read_only_tools():
    """Runner.READ_ONLY_TOOLS cobre as mesmas leituras (só leitura)."""
    assert set(Runner.READ_ONLY_TOOLS) == set(PURE_READ_TOOLS)


def test_run_command_still_allowed_as_dependency(registry):
    """P4 não muda semântica de dependency: ANALYSIS continua valendo."""
    from app.agent.execution.validator import TaskValidator
    from app.agent.planning.dependency import Dependency
    validator = TaskValidator(registry)
    validator._validate_dependency(Dependency(
        tool="run_command",
        arguments={"project_name": "p", "command": "python -m pytest -q"},
    ))
