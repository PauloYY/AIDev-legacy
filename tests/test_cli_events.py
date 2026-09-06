from app.agent.events import AgentEvent
from app.cli.events import handle_agent_event


def test_tool_end_prints_success_by_default(capsys):
    handle_agent_event(AgentEvent(type="tool_end", data={"name": "read_file"}))

    out = capsys.readouterr().out
    assert "✓ concluído" in out


def test_tool_end_prints_success_when_explicit_true(capsys):
    handle_agent_event(
        AgentEvent(type="tool_end", data={"name": "run_command", "success": True})
    )

    out = capsys.readouterr().out
    assert "✓ concluído" in out


def test_tool_end_prints_failure_when_command_failed(capsys):
    handle_agent_event(
        AgentEvent(type="tool_end", data={"name": "run_command", "success": False})
    )

    out = capsys.readouterr().out
    assert "✗" in out
    assert "erro" in out.lower()