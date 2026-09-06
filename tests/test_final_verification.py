import json
from unittest.mock import MagicMock

import pytest

from app.agent.context.final_verification import FinalVerification, FinalVerificationResult


class FakeLLM:
    def __init__(self, response_content=None, raise_on_generate=False):
        self.response_content = response_content
        self._raise = raise_on_generate
        self.call_count = 0

    def generate(self, messages, tools=None, component=None, iteration=None):
        self.call_count += 1
        if self._raise:
            raise RuntimeError("LLM indisponível")
        result = MagicMock()
        result.content = self.response_content
        return result


def _make_verification(llm=None):
    if llm is None:
        llm = FakeLLM(response_content=json.dumps({"passed": True, "problems": []}))
    return FinalVerification(llm)


def test_result_ok():
    result = FinalVerificationResult(FinalVerificationResult.OK)
    assert result.is_ok is True
    assert result.is_problems is False
    assert result.is_unavailable is False


def test_result_problems():
    result = FinalVerificationResult(
        FinalVerificationResult.PROBLEMS_FOUND, "problema 1"
    )
    assert result.is_problems is True
    assert result.is_ok is False
    assert result.report == "problema 1"


def test_result_unavailable():
    result = FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)
    assert result.is_unavailable is True
    assert result.is_ok is False


def test_verify_returns_ok_when_llm_says_passed_and_empty_problems():
    verification = _make_verification(
        FakeLLM(response_content=json.dumps({"passed": True, "problems": []}))
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_ok is True
    assert result.report == ""


def test_verify_returns_problems_when_llm_detects_issues():
    verification = _make_verification(
        FakeLLM(
            response_content=json.dumps(
                {"passed": False, "problems": ["import errado", "falta função"]}
            )
        )
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_problems is True
    assert "import errado" in result.report
    assert "falta função" in result.report


def test_verify_returns_unavailable_on_llm_exception():
    verification = _make_verification(
        FakeLLM(raise_on_generate=True)
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_unavailable is True


def test_verify_returns_unavailable_on_none_content():
    verification = _make_verification(FakeLLM(response_content=None))
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_unavailable is True


def test_verify_returns_unavailable_on_invalid_json():
    verification = _make_verification(FakeLLM(response_content="não é json"))
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_unavailable is True


def test_verify_returns_unavailable_when_missing_problems_field():
    verification = _make_verification(
        FakeLLM(response_content=json.dumps({"passed": False}))
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_unavailable is True


def test_verify_returns_unavailable_when_passed_is_not_bool():
    verification = _make_verification(
        FakeLLM(response_content=json.dumps({"problems": [], "passed": "sim"}))
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_unavailable is True


def test_verify_reports_passing_but_no_problems_detail():
    verification = _make_verification(
        FakeLLM(
            response_content=json.dumps({"passed": False, "problems": []})
        )
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_problems is True
    assert "passou" not in result.report.lower() or "não foi plenamente atingido" in result.report


def test_gather_project_info_returns_none_when_list_files_fails():
    verification = _make_verification()
    result = verification._gather_project_info(
        "proj", lambda tool, args: (_ for _ in ()).throw(RuntimeError("falha"))
    )
    assert result is None


def test_gather_project_info_handles_empty_files():
    verification = _make_verification()
    result = verification._gather_project_info(
        "proj", lambda tool, args: []
    )
    assert result is not None
    assert result["files"] == []


def test_gather_project_info_reads_files_and_symbols():
    verification = _make_verification()
    call_log = []

    def mock_execute(tool, args):
        call_log.append((tool, args))
        if tool == "list_files":
            return ["main.py", "utils.py"]
        if tool == "read_file":
            return "def foo(): pass"
        if tool == "list_symbols":
            return "- foo  (def)"
        return ""

    result = verification._gather_project_info("proj", mock_execute)

    assert result is not None
    assert result["files"] == ["main.py", "utils.py"]
    assert "main.py" in result["file_contents"]
    assert "main.py" in result["symbols"]


def test_gather_project_info_respects_max_files_limit():
    verification = _make_verification()
    many_files = [f"file_{i}.py" for i in range(30)]

    def mock_execute(tool, args):
        if tool == "list_files":
            return many_files
        return "content"

    result = verification._gather_project_info("proj", mock_execute)

    assert result["files_shown"] == 20
    assert result["files_omitted"] == 10


def test_build_prompt_contains_objective_and_summary():
    verification = _make_verification()
    info = {
        "files": [],
        "files_shown": 0,
        "files_omitted": 0,
        "file_contents": {},
        "symbols": {},
    }
    prompt = verification._build_prompt("meu objetivo", "meu resumo", info)
    assert "meu objetivo" in prompt
    assert "meu resumo" in prompt


def test_verify_cleanes_whitespace_from_problems():
    verification = _make_verification(
        FakeLLM(
            response_content=json.dumps(
                {
                    "passed": False,
                    "problems": ["  problema com espaços  ", "", "  outro "],
                }
            )
        )
    )
    result = verification.verify(
        objective="Criar um programa",
        project_name="proj",
        summary="resumo",
        tools_execute=lambda tool, args: [],
    )
    assert result.is_problems is True
    assert "  problema com espaços  " not in result.report
    assert "problema com espaços" in result.report
