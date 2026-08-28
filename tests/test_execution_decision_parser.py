import json

import pytest

from app.agent.execution.execution_decision_parser import ExecutionDecisionParser


@pytest.fixture
def parser():
    return ExecutionDecisionParser()


def test_parse_valid_decision(parser):
    content = json.dumps(
        {
            "tool": "write_file",
            "arguments": {"file_path": "a.py", "content": "x"},
        }
    )

    decision = parser.parse(content)

    assert decision.tool == "write_file"
    assert decision.arguments == {"file_path": "a.py", "content": "x"}


def test_parse_invalid_json_raises(parser):
    with pytest.raises(ValueError):
        parser.parse("não é json")


def test_parse_missing_tool_raises(parser):
    with pytest.raises(ValueError):
        parser.parse(json.dumps({"arguments": {}}))


def test_parse_arguments_not_dict_raises(parser):
    with pytest.raises(ValueError):
        parser.parse(json.dumps({"tool": "write_file", "arguments": "x"}))
