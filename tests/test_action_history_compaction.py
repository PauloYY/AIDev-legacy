"""Compactação do action_history."""

import pytest

from app.agent.context.operational_memory import OperationalMemory
from app.llm.utils import estimate_tokens
from app.tools.registry import ToolRegistry


@pytest.fixture
def memory(projects_root):
    tools = ToolRegistry()
    tools.load_defaults()
    return OperationalMemory(tools)


def _old_style_history_line(iteration, tool, arguments, success, result_summary):
    """Formatação ANTES da 2C (args integrais) — só p/ medir redução."""
    status = "OK" if success else "FALHOU"
    return f"[{iteration}] (task, {status}) {tool}({arguments!r}) -> {result_summary}"


def test_write_file_hides_full_content(memory):
    big = "SECRET-CONTENT-" * 500
    memory.record(
        iteration=1,
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "models/user.py",
            "content": big,
        },
        result="File written successfully",
        success=True,
    )
    history = memory.render_history()
    assert "models/user.py" in history
    assert "OK" in history
    assert big not in history
    assert "SECRET-CONTENT" not in history
    assert "omitted" in history.lower()


def test_small_content_is_also_omitted_consistently(memory):
    memory.record(
        iteration=1,
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "a.py",
            "content": "x",
        },
        result="ok",
        success=True,
    )
    history = memory.render_history()
    assert "a.py" in history
    assert "'content': '<omitted: 1 chars>'" in history


def test_large_arguments_are_capped(memory):
    huge_command = "echo " + "y" * 50_000
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": huge_command},
        result="ok",
        success=True,
    )
    history = memory.render_history()
    assert len(history) < 5_000
    assert huge_command not in history
    assert "chars]" in history

    memory.reset()
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={
            "project_name": "p",
            "command": "python main.py",
            "stdin": "z" * 20_000,
        },
        result="ok",
        success=True,
    )
    assert len(memory.render_history()) < 5_000


def test_other_tools_preserve_semantics(memory):
    memory.record(
        iteration=1,
        tool="read_file",
        arguments={"project_name": "p", "file_path": "src/app.py"},
        result="conteúdo...",
        success=True,
    )
    memory.record(
        iteration=2,
        tool="run_command",
        arguments={"project_name": "p", "command": "python -m pytest -q"},
        result="STATUS: success",
        success=True,
    )
    memory.record(
        iteration=3,
        tool="list_files",
        arguments={"project_name": "p"},
        result="['a.py']",
        success=True,
    )
    memory.record(
        iteration=4,
        tool="find_references",
        arguments={"project_name": "p", "symbol": "UserService"},
        result="['a.py:1']",
        success=True,
    )
    memory.record(
        iteration=5,
        tool="list_symbols",
        arguments={"project_name": "p", "file_path": "a.py"},
        result="['Foo']",
        success=True,
    )
    history = memory.render_history()
    assert "src/app.py" in history
    assert "python -m pytest -q" in history
    assert "list_files" in history
    assert "UserService" in history
    assert "list_symbols" in history


def test_result_truncation_still_works(memory):
    big_result = "R" * 5_000
    memory.record(
        iteration=1,
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        result=big_result,
        success=True,
    )
    history = memory.render_history()
    assert big_result not in history
    assert "[+4700 chars]" in history
    assert "R" * 300 in history


def test_history_preserves_order(memory):
    for i in range(1, 4):
        memory.record(
            iteration=i,
            tool="read_file",
            arguments={"project_name": "p", "file_path": f"f{i}.py"},
            result="ok",
            success=True,
        )
    lines = memory.render_history().splitlines()
    assert len(lines) == 3
    assert "f1.py" in lines[0]
    assert "f2.py" in lines[1]
    assert "f3.py" in lines[2]
    assert lines[0].startswith("[1]")
    assert lines[2].startswith("[3]")


def test_semantic_info_preserved(memory):
    memory.record(
        iteration=2,
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "app/services/user_service.py",
            "content": "C" * 10_000,
        },
        result="ok",
        success=False,
    )
    memory.record(
        iteration=3,
        tool="run_command",
        arguments={"project_name": "p", "command": "python -m pytest"},
        result="fail",
        success=False,
    )
    history = memory.render_history()
    assert "write_file" in history
    assert "app/services/user_service.py" in history
    assert "FAILED" in history
    assert "run_command" in history
    assert "python -m pytest" in history


def test_real_arguments_not_mutated(memory):
    args = {
        "project_name": "p",
        "file_path": "a.py",
        "content": "ORIGINAL-" * 1000,
    }
    snapshot = dict(args)
    memory.record(
        iteration=1, tool="write_file", arguments=args,
        result="ok", success=True,
    )
    assert args == snapshot
    assert len(args["content"]) == len(snapshot["content"])
    stored = memory._actions[0].arguments
    assert stored["content"] == snapshot["content"]
    assert snapshot["content"] not in memory.render_history()


def test_context_breakdown_still_measures_action_history(memory):
    from unittest.mock import MagicMock
    from app.agent.planning.decision_parser import DecisionParser
    from app.agent.planning.planner import Planner
    from app.llm.client import LLMClient
    from app.llm.models import LLMResponse, Usage

    tools = ToolRegistry()
    tools.load_defaults()

    class Prov:
        name = "fake"

        def generate(self, messages, tools=None):
            return LLMResponse(
                content='{"action": "finish", "content": "done"}',
                tool_calls=[],
                usage=Usage(10, 1, 11),
                provider="fake",
            )

    llm = LLMClient(Prov())
    planner = Planner(
        llm=llm, parser=DecisionParser(tools=tools), tools=tools
    )
    memory.record(
        iteration=1,
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "a.py",
            "content": "B" * 5000,
        },
        result="ok",
        success=True,
    )
    ctx = (
        "PROJECT SUMMARY:\ns\n\n"
        "ACTION HISTORY (memory):\n"
        f"{memory.render_history()}\n"
    )
    planner.plan(objective="o", context=ctx, iteration=1)
    history = llm.usage.get_planner_context_history()
    assert len(history) == 1
    assert "action_history" in history[0]["sections"]
    assert history[0]["sections"]["action_history"]["chars"] < 1000


def test_growth_scales_with_metadata_not_file_size(memory):
    n = 15
    file_kb = 10_000
    for i in range(1, n + 1):
        memory.record(
            iteration=i,
            tool="write_file",
            arguments={
                "project_name": "p",
                "file_path": f"src/file_{i:02d}.py",
                "content": "x" * file_kb,
            },
            result="STATUS: success",
            success=True,
        )
    after = memory.render_history()
    after_chars = len(after)
    after_tokens = estimate_tokens(after)

    before_lines = [
        _old_style_history_line(
            i, "write_file",
            {
                "project_name": "p",
                "file_path": f"src/file_{i:02d}.py",
                "content": "x" * file_kb,
            },
            True, "STATUS: success",
        )
        for i in range(1, n + 1)
    ]
    before_chars = sum(len(line) + 1 for line in before_lines)
    before_tokens = estimate_tokens("\n".join(before_lines))

    assert after_chars < before_chars // 10
    assert after_tokens < before_tokens // 10
    assert after_chars < n * 500
    for i in range(1, n + 1):
        assert f"file_{i:02d}.py" in after
