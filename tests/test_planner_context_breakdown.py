"""Etapa 2B — instrumentação observacional do contexto do Planner.

Verifica medição sem otimização: cada componente é medido via
estimate_tokens(), totais batem, vazios/None não quebram, múltiplas
iterações são preservadas e prompt/decisão/component/iteration seguem
inalterados. Também cobre compatibilidade do UsageTracker.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.agent.planning.prompt_sections import (
    PLANNER_COMPONENTS,
    measure_sections,
    split_planner_context,
    summarize_measurements,
)
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Message, Usage
from app.llm.usage import UsageTracker
from app.llm.utils import estimate_tokens
from app.tools.registry import ToolRegistry


REQUIRED_COMPONENTS = [
    "static_template",
    "project_summary",
    "objective_checklist",
    "error_checklist",
    "planner_error_memory",
    "action_history",
    "file_list",
    "task_context",
    "execution_result",
    "error_blocks",
    "objective",
]


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


@pytest.fixture
def parser(tools):
    return DecisionParser(tools=tools)


def _make_planner_with_mock_llm(tools, parser):
    llm = MagicMock()
    return Planner(llm=llm, parser=parser, tools=tools), llm


def _full_context() -> str:
    return (
        "RESUMO DO PROJETO:\n"
        "summary text here\n\n"
        "CHECKLIST DO OBJETIVO (definido uma única vez):\n"
        "[x] 1. item one\n[ ] 2. item two\n\n"
        "CHECKLIST DE ERROS ATUAIS (extraído automaticamente):\n"
        "- 1. test foo failed\n\n"
        "ERROS PROIBIDOS (você já cometeu estes erros):\n"
        "- ValueError: bad\n\n"
        "HISTÓRICO DE AÇÕES (memória determinística):\n"
        "[1] (task, OK) write_file -> ok\n\n"
        "COMANDOS CONHECIDOS QUE JÁ FUNCIONARAM (reutilize):\n"
        "- Teste: ainda não descoberto\n- Build: ainda não descoberto\n\n"
        "ARQUIVOS ATUAIS DO PROJETO (lista real do disco):\n"
        "- main.py\n- other.py\n\n"
        "TASK PAI:\nTool: write_file\nArguments: x\n\nDEPENDÊNCIAS:\nNenhuma.\n\n"
        "RESULTADO DA EXECUÇÃO:\nSTATUS: sucesso\n\n"
        "ERRO DE VALIDAÇÃO ANTES DO FINISH:\nbloqueado"
    )


class _FakeProvider:
    """Provider fake que captura messages e retorna finish válido."""

    def __init__(self, content='{"action": "finish", "content": "done"}'):
        self.content = content
        self.seen_messages = []
        self.name = "fake"

    def generate(self, messages, tools=None):
        self.seen_messages.append(messages)
        return LLMResponse(
            content=self.content,
            tool_calls=[],
            usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
            provider="fake",
        )


# 1. Cada componente é medido.
def test_each_required_component_is_measured(tools, parser):
    planner, _ = _make_planner_with_mock_llm(tools, parser)
    sections = planner.build_prompt_sections("obj", _full_context())
    for name in REQUIRED_COMPONENTS:
        assert name in sections, f"missing section {name}"
    measured = planner.measure_prompt_sections(sections)
    for name in REQUIRED_COMPONENTS:
        assert name in measured
        assert "chars" in measured[name]
        assert "estimated_tokens" in measured[name]


# 2. estimate_tokens() é usado para cada componente.
def test_estimate_tokens_used_for_each_component(tools, parser):
    planner, _ = _make_planner_with_mock_llm(tools, parser)
    sections = planner.build_prompt_sections("obj", _full_context())
    with patch(
        "app.agent.planning.prompt_sections.estimate_tokens",
        side_effect=lambda t: estimate_tokens(t),
    ) as mocked:
        measured = planner.measure_prompt_sections(sections)
    # Chamado uma vez por componente presente no dict.
    assert mocked.call_count == len(sections)
    for name, text in sections.items():
        assert measured[name]["estimated_tokens"] == estimate_tokens(text)


# 3. O total é calculado corretamente.
def test_total_is_sum_of_components(tools, parser):
    planner, _ = _make_planner_with_mock_llm(tools, parser)
    sections = planner.build_prompt_sections("objective text", _full_context())
    measured = planner.measure_prompt_sections(sections)
    totals = summarize_measurements(measured)
    assert totals["chars"] == sum(v["chars"] for v in measured.values())
    assert totals["estimated_tokens"] == sum(
        v["estimated_tokens"] for v in measured.values()
    )
    # Soma dos componentes equivale ao prompt final (a menos de
    # arredondamento de tokens; chars deve ser exato).
    prompt = planner._build_prompt("objective text", _full_context())
    assert totals["chars"] == len(prompt)


# 4. Componentes vazios não causam erro.
def test_empty_components_do_not_error():
    sections = split_planner_context("")
    assert all(v == "" for v in sections.values())
    measured = measure_sections({"a": "", "b": ""})
    assert measured["a"] == {"chars": 0, "estimated_tokens": 0}
    assert summarize_measurements(measured) == {"chars": 0, "estimated_tokens": 0}


# 5. Medição funciona quando contexto é None/ausente.
def test_none_objective_and_context_handled(tools, parser):
    planner, _ = _make_planner_with_mock_llm(tools, parser)
    sections = planner.build_prompt_sections(None, None)
    assert sections["objective"] == ""
    measured = planner.measure_prompt_sections(sections)
    # static_template sempre existe; demais dinâmicos vazios exceto headers?
    # O importante é não levantar e zerar o que falta.
    assert measured["objective"] == {"chars": 0, "estimated_tokens": 0}
    assert measured["static_template"]["chars"] > 0

    sections2 = planner.build_prompt_sections("obj", None)
    assert sections2["objective"] == "obj"

    # measure_sections aceita None diretamente.
    measured2 = measure_sections({"x": None, "y": "abc"})
    assert measured2["x"] == {"chars": 0, "estimated_tokens": 0}
    assert measured2["y"]["chars"] == 3


# 6. O breakdown preserva múltiplas iterações.
def test_breakdown_preserves_multiple_iterations(tools, parser):
    provider = _FakeProvider()
    llm = LLMClient(provider)
    planner = Planner(llm=llm, parser=parser, tools=tools)

    planner.plan(objective="obj1", context="RESUMO DO PROJETO:\nshort", iteration=1)
    planner.plan(objective="obj2", context=_full_context(), iteration=2)

    history = llm.usage.get_planner_context_history()
    assert len(history) == 2
    assert history[0]["iteration"] == 1
    assert history[1]["iteration"] == 2
    # Segunda iteração tem contexto maior que a primeira.
    assert history[1]["estimated_total_tokens"] > history[0]["estimated_total_tokens"]

    stats = llm.usage.planner_context_stats()
    assert stats["calls"] == 2
    assert stats["components"]["static_template"]["total_tokens"] == (
        stats["components"]["static_template"]["avg_tokens"] * 2
    )


# 7. A medição não altera o prompt final.
def test_measurement_does_not_alter_final_prompt(tools, parser):
    provider = _FakeProvider()
    llm = LLMClient(provider)
    planner = Planner(llm=llm, parser=parser, tools=tools)

    objective = "my objective"
    context = _full_context()
    expected = planner._build_prompt(objective, context)
    planner.plan(objective=objective, context=context, iteration=3)

    sent = provider.seen_messages[0][0].content
    assert sent == expected


# 8. A medição não altera a decisão do Planner.
def test_measurement_does_not_alter_decision(tools, parser):
    provider = _FakeProvider(
        content='{"action": "finish", "content": "resultado final"}'
    )
    llm = LLMClient(provider)
    planner = Planner(llm=llm, parser=parser, tools=tools)

    decision = planner.plan(objective="o", context=_full_context(), iteration=1)
    assert decision.content == "resultado final"

    # Com contexto vazio a decisão (mockada) também passa intacta.
    provider2 = _FakeProvider(
        content='{"action": "fail", "reason": "impossivel"}'
    )
    planner2 = Planner(llm=LLMClient(provider2), parser=parser, tools=tools)
    decision2 = planner2.plan(objective="o", context="", iteration=1)
    assert decision2.reason == "impossivel"


# 9. Comportamento anterior do UsageTracker continua funcionando.
def test_usagetracker_backward_compatibility():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq", component="Planner")
    # Sem breakdown: histórico de contexto vazio, mas breakdown antigo ok.
    assert tracker.get_planner_context_history() == []
    assert "Planner" in tracker.breakdown()
    assert "Chamadas à LLM" in tracker.summary()

    stats = tracker.planner_context_stats()
    assert stats["calls"] == 0

    empty_report = tracker.planner_context_breakdown()
    assert "Planner Context Breakdown" in empty_report


# 10. Planner atribui component="Planner" e iteration.
def test_planner_records_component_and_iteration(tools, parser):
    provider = _FakeProvider()
    llm = LLMClient(provider)
    planner = Planner(llm=llm, parser=parser, tools=tools)

    planner.plan(objective="o", context="ctx", iteration=7)
    records = llm.usage.get_records()
    assert len(records) == 1
    assert records[0].component == "Planner"
    assert records[0].iteration == 7
    assert records[0].context_breakdown is not None
    assert "static_template" in records[0].context_breakdown


def test_report_contains_avg_total_max_and_iterations(tools, parser):
    provider = _FakeProvider()
    llm = LLMClient(provider)
    planner = Planner(llm=llm, parser=parser, tools=tools)
    planner.plan(objective="o", context="RESUMO DO PROJETO:\na", iteration=1)
    planner.plan(objective="o", context=_full_context(), iteration=2)

    report = llm.usage.planner_context_breakdown()
    assert "Planner Context Breakdown" in report
    assert "Calls: 2" in report
    assert "Avg Tokens" in report
    assert "Total Tokens" in report
    assert "Max Tokens" in report
    assert "static_template" in report
    assert "TOTAL (estimated)" in report
    assert "TOTAL (provider)" in report
    assert "iteration 1" in report
    assert "iteration 2" in report


def test_error_blocks_capture_retry_markers(tools, parser):
    planner, _ = _make_planner_with_mock_llm(tools, parser)
    ctx = (
        "RESUMO DO PROJETO:\ns\n\n"
        "ARQUIVOS ATUAIS DO PROJETO (lista):\n- a.py\n\n"
        "RESULTADO DA EXECUÇÃO:\nok\n\n"
        "CORREÇÃO DA TENTATIVA ANTERIOR:\nValueError: bad\n\n"
        "Não execute ferramentas."
    )
    sections = planner.build_prompt_sections("o", ctx)
    assert "CORREÇÃO DA TENTATIVA ANTERIOR" in sections["error_blocks"]

    ctx2 = "RESUMO DO PROJETO:\ns\n\nERRO REPETIDO — LEIA COM ATENÇÃO:\nrepetiu"
    sections2 = planner.build_prompt_sections("o", ctx2)
    assert "ERRO REPETIDO" in sections2["error_blocks"]


def test_unclassified_context_goes_to_other_context():
    sections = split_planner_context("apenas um texto livre sem marcadores")
    assert sections["other_context"] == "apenas um texto livre sem marcadores"
    assert sections["project_summary"] == ""


def test_llmclient_forwards_breakdown_without_breaking_old_callers():
    provider = _FakeProvider()
    client = LLMClient(provider)
    # Chamada antiga sem breakdown continua funcionando.
    client.generate(
        messages=[Message(role="user", content="hi")],
        component="Planner",
        iteration=1,
    )
    assert client.usage.calls == 1
    assert client.usage.get_planner_context_history() == []
