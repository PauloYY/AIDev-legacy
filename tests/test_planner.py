import json

import pytest

from app.agent.planning.planner import Planner
from app.tools.base import Tool, ToolType
from app.tools.registry import ToolRegistry


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


@pytest.fixture
def planner(tools):
    from unittest.mock import MagicMock
    llm = MagicMock()
    from app.agent.planning.decision_parser import DecisionParser
    parser = DecisionParser(tools=tools)
    return Planner(llm=llm, parser=parser, tools=tools)


def test_tools_context_is_serialized_json(planner, tools):
    context = planner._build_tools_context()
    parsed = json.loads(context)
    assert isinstance(parsed, list)
    assert len(parsed) == len(tools.definitions)


def test_tools_context_contains_expected_tools(planner, tools):
    context = planner._build_tools_context()
    parsed = json.loads(context)
    names = {d["function"]["name"] for d in parsed}
    assert names == {
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "delete_file",
        "find_references",
        "run_command",
        "check_project",
        "list_symbols",
    }


def test_tools_context_cached_on_repeated_calls(planner):
    context1 = planner._build_tools_context()
    context2 = planner._build_tools_context()

    # Same object returned from cache
    assert context1 is context2


def test_tools_context_invalidated_when_tool_added(planner, tools):
    context1 = planner._build_tools_context()

    new_tool = Tool(
        name="new_tool",
        function=lambda: "ok",
        definition={
            "type": "function",
            "function": {
                "name": "new_tool",
                "description": "A new tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        type=ToolType.ANALYSIS,
    )
    tools.register(new_tool)

    context2 = planner._build_tools_context()

    assert context1 != context2
    parsed = json.loads(context2)
    names = {d["function"]["name"] for d in parsed}
    assert "new_tool" in names


def test_tools_context_invalidated_when_tool_removed(tools):
    from unittest.mock import MagicMock
    from app.agent.planning.decision_parser import DecisionParser

    planner = Planner(
        llm=MagicMock(),
        parser=DecisionParser(tools=tools),
        tools=tools,
    )

    context1 = planner._build_tools_context()
    initial_count = len(json.loads(context1))

    # Remove a tool by unregistering (internally via _tools.pop)
    if "list_symbols" in tools._tools:
        del tools._tools["list_symbols"]

    context2 = planner._build_tools_context()
    final_count = len(json.loads(context2))

    assert context1 != context2
    assert final_count == initial_count - 1


def test_tools_context_empty_when_no_tools():
    from unittest.mock import MagicMock
    from app.agent.planning.decision_parser import DecisionParser

    empty_registry = ToolRegistry()
    planner = Planner(
        llm=MagicMock(),
        parser=DecisionParser(tools=empty_registry),
        tools=empty_registry,
    )

    context = planner._build_tools_context()
    parsed = json.loads(context)
    assert parsed == []


def test_tools_context_uses_sorted_keys_for_stable_cache(tools):
    from unittest.mock import MagicMock
    from app.agent.planning.decision_parser import DecisionParser

    planner = Planner(
        llm=MagicMock(),
        parser=DecisionParser(tools=tools),
        tools=tools,
    )

    context1 = planner._build_tools_context()
    context2 = planner._build_tools_context()

    # Even though internal dict order may vary, the cached names
    # should be sorted so repeated calls always match
    assert context1 == context2
    assert planner._cached_tools_names == tuple(sorted(tools._tools.keys()))
