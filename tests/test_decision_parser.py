import json

import pytest

from app.agent.planning.decision import DecisionAction
from app.agent.planning.decision_parser import DecisionParser


@pytest.fixture
def parser():
    return DecisionParser()


def test_parse_task_decision(parser):
    content = json.dumps(
        {
            "action": "task",
            "task": {
                "tool": "write_file",
                "arguments": {"file_path": "a.py", "content": "x"},
                "dependencies": [
                    {"tool": "list_files", "arguments": {"project_name": "p"}}
                ],
            },
        }
    )

    decision = parser.parse(content)

    assert decision.action == DecisionAction.TASK
    assert decision.task.tool == "write_file"
    assert decision.task.arguments == {"file_path": "a.py", "content": "x"}
    assert len(decision.task.dependencies) == 1
    assert decision.task.dependencies[0].tool == "list_files"


def test_parse_finish_decision(parser):
    content = json.dumps({"action": "finish", "content": "Concluído."})

    decision = parser.parse(content)

    assert decision.action == DecisionAction.FINISH
    assert decision.content == "Concluído."


def test_parse_fail_decision(parser):
    content = json.dumps({"action": "fail", "reason": "Impossível continuar."})

    decision = parser.parse(content)

    assert decision.action == DecisionAction.FAIL
    assert decision.reason == "Impossível continuar."


def test_parse_invalid_json_raises(parser):
    with pytest.raises(ValueError):
        parser.parse("isso não é json")


def test_parse_unknown_action_raises(parser):
    with pytest.raises(ValueError):
        parser.parse(json.dumps({"action": "dance"}))


def test_parse_task_without_task_object_raises(parser):
    with pytest.raises(ValueError):
        parser.parse(json.dumps({"action": "task"}))
