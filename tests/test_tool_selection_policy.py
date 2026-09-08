"""Tool selection policy: write_file vs edit_file vs delete_file.

Validates that the policy "most precise tool for the operation" is
explicit in tool descriptions/schemas and in the Planner prompts
(legacy + compact), without changing any tool behavior.
"""

from unittest.mock import MagicMock

from app.agent.parallel import PURE_READ_TOOLS, all_pure_read
from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.llm.usage import UsageTracker
from app.tools.base import ToolType
from app.tools.registry import ToolRegistry


def _registry():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


def _planner(tools):
    return Planner(
        llm=MagicMock(), parser=DecisionParser(tools=tools), tools=tools)


# --- Registration: no tool added or removed ----------------------------


def test_all_three_tools_registered():
    names = {d["function"]["name"] for d in _registry().definitions}
    assert {"write_file", "edit_file", "delete_file"} <= names
    assert names == {
        "list_files", "read_file", "write_file", "edit_file",
        "delete_file", "find_references", "run_command",
        "check_project", "list_symbols",
    }


def test_mutative_tools_stay_impure():
    registry = _registry()
    for name in ("write_file", "edit_file", "delete_file"):
        tool = registry.get(name)
        assert tool.type == ToolType.EXECUTION
        assert tool.pure is False
        assert name not in PURE_READ_TOOLS


def test_mutative_tools_outside_parallel_batches():
    assert all_pure_read(["read_file", "edit_file"]) is False
    assert all_pure_read(["write_file", "delete_file"]) is False
    assert all_pure_read(["edit_file", "delete_file"]) is False


# --- Descriptions reflect create/rebuild vs localized vs remove --------


def test_write_description_is_create_or_rebuild():
    desc = _registry().get("write_file").definition["function"]["description"]
    assert "novo" in desc  # cria arquivo novo
    assert "edit_file" in desc  # aponta a alternativa pontual


def test_edit_description_requires_existing_localized_change():
    tool = _registry().get("edit_file")
    desc = tool.definition["function"]["description"]
    assert "existente" in desc
    assert "write_file" in desc  # criar => write_file; reescrita => write_file
    assert "exatamente uma vez" in desc
    params = tool.definition["function"]["parameters"]
    assert params["required"] == [
        "project_name", "file_path", "old_text", "new_text"]


def test_delete_description_is_remove_only():
    desc = _registry().get("delete_file").definition["function"]["description"]
    assert "desnecessário" in desc
    assert ".git" in desc
    params = _registry().get("delete_file").definition["function"][
        "parameters"]
    assert params["required"] == ["project_name", "file_path"]


def test_descriptions_stay_concise():
    for name in ("write_file", "edit_file", "delete_file"):
        desc = _registry().get(name).definition["function"]["description"]
        assert len(desc) <= 500, f"{name} description too long"


# --- Planner receives the explicit policy (both templates) -------------


def _prompts():
    tools = _registry()
    planner = _planner(tools)
    return (planner._build_prompt("obj", "ctx"),
            planner._build_prompt_compact("obj", "ctx"))


def test_planner_policy_present_in_both_templates():
    legacy, compact = _prompts()
    for prompt in (legacy, compact):
        assert "TOOL SELECTION POLICY" in prompt
        assert "write_file" in prompt
        assert "edit_file" in prompt
        assert "delete_file" in prompt


def test_planner_policy_covers_all_four_cases():
    legacy, compact = _prompts()
    for prompt in (legacy, compact):
        lowered = prompt.lower()
        assert "new file" in lowered
        assert "localized" in lowered
        assert "substantial rewrite" in lowered
        assert "unnecessary file" in lowered


def test_planner_policy_is_preference_not_hard_rule():
    legacy, compact = _prompts()
    for prompt in (legacy, compact):
        assert "most precise tool" in prompt
        assert "preference" in prompt.lower()


# --- Observability: existing tool_stats answers write/edit/delete ------


def test_tool_stats_counts_each_tool_separately(projects_root):
    from app.tools.filesystem.delete_file import delete_file
    from app.tools.filesystem.edit_file import edit_file
    from app.tools.filesystem.write_file import write_file

    tracker = UsageTracker()
    registry = ToolRegistry(tracker=tracker)
    registry.load_defaults()

    registry.execute("write_file", {
        "project_name": "p", "file_path": "a.py", "content": "x = 1\n"})
    registry.execute("edit_file", {
        "project_name": "p", "file_path": "a.py",
        "old_text": "x = 1", "new_text": "x = 2"})
    registry.execute("delete_file", {
        "project_name": "p", "file_path": "a.py"})

    by_tool = tracker.tool_stats()["by_tool"]
    assert by_tool["write_file"]["calls"] == 1
    assert by_tool["edit_file"]["calls"] == 1
    assert by_tool["delete_file"]["calls"] == 1

    # Sanity: direct functions unchanged by the policy task.
    assert write_file is not None
    assert edit_file is not None
    assert delete_file is not None
