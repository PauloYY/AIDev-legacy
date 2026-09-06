import json
from unittest.mock import MagicMock

import pytest

from app.agent.context.error_checklist import ErrorChecklist


@pytest.fixture
def llm():
    return MagicMock()


@pytest.fixture
def checklist(llm):
    return ErrorChecklist(llm)


def _llm_returns(llm, items):
    llm.generate.return_value = MagicMock(
        content=json.dumps({"items": items})
    )


def test_render_is_none_when_empty(checklist):
    assert checklist.render() is None


def test_pending_count_is_zero_when_empty(checklist):
    assert checklist.pending_count == 0


def test_generate_creates_items_from_llm_response(checklist, llm):
    _llm_returns(
        llm,
        [
            "RideService.test.js: 'deve criar corrida' falhou",
            "RideService.test.js: 'deve cancelar corrida' falhou",
        ],
    )

    checklist.generate("npm test", "STATUS: falha (exit code 1)\n...")

    assert checklist.pending_count == 2
    assert "deve criar corrida" in checklist.render()
    assert "deve cancelar corrida" in checklist.render()


def test_generate_with_empty_items_list_clears_checklist(checklist, llm):
    _llm_returns(llm, ["falha 1"])
    checklist.generate("npm test", "saida")
    assert checklist.pending_count == 1

    _llm_returns(llm, [])
    checklist.generate("npm test", "saida nova, sem falhas reais")

    assert checklist.pending_count == 0
    assert checklist.render() is None


def test_generate_raises_on_invalid_llm_format(checklist, llm):
    llm.generate.return_value = MagicMock(
        content=json.dumps({"not_items": []})
    )

    with pytest.raises(ValueError):
        checklist.generate("npm test", "saida")


def test_generate_truncates_huge_input(checklist, llm):
    _llm_returns(llm, ["falha"])

    huge_output = "x" * (ErrorChecklist.MAX_INPUT_CHARS * 3)
    checklist.generate("npm test", huge_output)

    prompt_sent = llm.generate.call_args.kwargs["messages"][0].content
    assert "truncado" in prompt_sent
    assert len(prompt_sent) < len(huge_output)


def test_clear_removes_all_items(checklist, llm):
    _llm_returns(llm, ["falha 1", "falha 2"])
    checklist.generate("npm test", "saida")
    assert checklist.pending_count == 2

    checklist.clear()

    assert checklist.pending_count == 0
    assert checklist.render() is None


def test_reset_removes_all_items(checklist, llm):
    _llm_returns(llm, ["falha 1"])
    checklist.generate("npm test", "saida")

    checklist.reset()

    assert checklist.pending_count == 0
    assert checklist.render() is None


def test_generate_regenerates_from_scratch_discarding_old_items(
    checklist, llm
):
    # Regra central do design: cada generate() substitui os itens
    # anteriores por completo — não acumula entre rodadas de falha.
    _llm_returns(llm, ["falha antiga 1", "falha antiga 2"])
    checklist.generate("npm test", "primeira saida")
    assert checklist.pending_count == 2

    _llm_returns(llm, ["falha nova"])
    checklist.generate("npm test", "segunda saida")

    assert checklist.pending_count == 1
    rendered = checklist.render()
    assert "falha nova" in rendered
    assert "falha antiga" not in rendered


def test_render_includes_command_context_marker(checklist, llm):
    _llm_returns(llm, ["alguma falha"])
    checklist.generate("npm test", "saida")

    rendered = checklist.render()

    assert "CHECKLIST DE ERROS" in rendered
    assert "some sozinho" in rendered