"""Observabilidade e retry do ProjectSummaryUpdater."""

import logging

import pytest

from app.agent.context.project_summary import ProjectSummary
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.agent.execution.task import Task
from app.exceptions import LLMAPIError, LLMConnectionError
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage
from app.llm.usage import UsageTracker


class _ScriptedLLM:
    """Fake com roteiro de respostas/erros; captura kwargs e sleeps."""

    def __init__(self, script, sleep_log=None):
        self.script = list(script)
        self.calls: list[dict] = []
        self.sleep_log = sleep_log if sleep_log is not None else []
        self.usage = UsageTracker()

    def generate(self, messages, tools=None, component=None,
                 iteration=None, attempt=1):
        from app.llm.utils import estimate_tokens
        import time

        self.calls.append({
            "component": component,
            "iteration": iteration,
            "attempt": attempt,
            "prompt_chars": len(messages[0].content),
        })
        start = time.monotonic()
        item = self.script.pop(0) if self.script else "resumo ok"
        duration_ms = (time.monotonic() - start) * 1000
        if isinstance(item, Exception):
            self.usage.record(
                Usage(), None, component=component, iteration=iteration,
                prompt_chars=len(messages[0].content),
                duration_ms=duration_ms, attempt=attempt,
                success=False, error=type(item).__name__,
            )
            raise item
        self.usage.record(
            Usage(10, 2, 12), "fake", component=component,
            iteration=iteration,
            prompt_chars=len(messages[0].content),
            completion_chars=len(item or ""),
            duration_ms=duration_ms, attempt=attempt,
            empty_response=not (item and item.strip()),
        )
        return LLMResponse(
            content=item, tool_calls=[],
            usage=Usage(10, 2, 12), provider="fake",
        )


def _make_updater(script):
    sleeps: list[float] = []
    llm = _ScriptedLLM(script)
    updater = ProjectSummaryUpdater(
        llm=llm, summary=ProjectSummary(), sleep_fn=sleeps.append,
    )
    return updater, llm, sleeps


def _task():
    return Task(
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "app.py",
            "content": "x" * 5000,
        },
    )


def test_registra_component_iteration_prompt_completion_duracao(projects_root):
    updater, llm, _ = _make_updater(["resumo válido"])
    updater.update(
        objective="obj", project_name="p", task=_task(),
        result="ok", iteration=12,
    )
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["component"] == "ProjectSummaryUpdater"
    assert call["iteration"] == 12
    assert call["attempt"] == 1
    assert call["prompt_chars"] > 0
    records = llm.usage.get_records()
    assert len(records) == 1
    assert records[0].prompt_chars == call["prompt_chars"]
    assert records[0].completion_chars == len("resumo válido")
    assert records[0].duration_ms >= 0.0


def test_resposta_valida_nao_faz_retry(projects_root):
    updater, llm, sleeps = _make_updater(["resumo ok"])
    out = updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=1,
    )
    assert out == "resumo ok"
    assert len(llm.calls) == 1
    assert sleeps == []


def test_vazio_depois_valido(projects_root):
    updater, llm, sleeps = _make_updater(["", "resumo recuperado"])
    out = updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=2,
    )
    assert out == "resumo recuperado"
    assert len(llm.calls) == 2
    assert [c["attempt"] for c in llm.calls] == [1, 2]
    assert sleeps == [1.0]
    stats = llm.usage.project_summary_stats()
    assert stats["invocations"] == 1
    assert stats["attempts"] == 2
    assert stats["retries"] == 1
    assert stats["empty_responses"] == 1
    assert stats["successes"] == 1
    assert stats["failures"] == 0


def test_dois_vazios_depois_valido(projects_root):
    updater, llm, sleeps = _make_updater(["  ", "", "resumo final"])
    out = updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=3,
    )
    assert out == "resumo final"
    assert len(llm.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_todas_vazias_falha_explicita_com_3_chamadas(projects_root):
    updater, llm, sleeps = _make_updater(["", " ", ""])
    with pytest.raises(ValueError, match="após 3 tentativas"):
        updater.update(
            objective="o", project_name="p", task=_task(), result="ok",
            iteration=4,
        )
    assert len(llm.calls) == 3
    assert sleeps == [1.0, 2.0]
    stats = llm.usage.project_summary_stats()
    assert stats["failures"] == 1
    assert stats["empty_responses"] == 3


def test_erro_transitorio_faz_retry(projects_root):
    updater, llm, sleeps = _make_updater(
        [LLMConnectionError("read timed out"), "resumo ok"]
    )
    out = updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=5,
    )
    assert out == "resumo ok"
    assert len(llm.calls) == 2
    assert sleeps == [1.0]
    records = llm.usage.get_records()
    assert records[0].success is False
    assert records[0].error == "LLMConnectionError"


def test_erro_permanente_nao_faz_retry(projects_root):
    updater, llm, sleeps = _make_updater(
        [LLMAPIError("bad request", status_code=400)]
    )
    with pytest.raises(LLMAPIError):
        updater.update(
            objective="o", project_name="p", task=_task(), result="ok",
            iteration=6,
        )
    assert len(llm.calls) == 1
    assert sleeps == []


def test_backoff_valores_esperados():
    assert ProjectSummaryUpdater.RETRY_BACKOFF_SECONDS == (1.0, 2.0)
    assert ProjectSummaryUpdater.MAX_SUMMARY_ATTEMPTS == 3


def test_runner_passa_iteration_ao_updater():
    from tests.test_runner import (
        FakePlanner, FakeTaskDecisionMaker, FakeToolRegistry,
        FakeValidator, FakeProjectContext, FakeSummary,
        FakeTaskContextBuilder, FakeOperationalMemory, FakeChecklist,
        FakeErrorChecklist, FakeFinalVerification,
        _make_runner,
    )
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.execution.execution_decision import ExecutionDecision
    from app.agent.execution.task import Task as TaskData
    from app.agent.planning.decision import Decision, DecisionAction

    seen: dict = {}

    class CapturingUpdater:
        def update(self, objective, project_name, task, result,
                   iteration=None):
            seen["iteration"] = iteration
            return "resumo"

    task = TaskData(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    runner = _make_runner(
        [Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")],
        [ExecutionDecision(tool="write_file", arguments=task.arguments)],
        final_verification=FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)
        ),
    )
    runner.project_summary_updater = CapturingUpdater()
    runner.run(objective="obj", project_name="p")
    assert seen["iteration"] == 1


def test_logs_e_erro_sem_conteudo_sensivel(projects_root, caplog):
    secret = "SUPER-SECRETO-" * 500
    updater, llm, _ = _make_updater(["", "", ""])
    task = Task(
        tool="write_file",
        arguments={
            "project_name": "p", "file_path": "app.py", "content": secret,
        },
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError) as exc_info:
            updater.update(
                objective="o", project_name="p", task=task, result="ok",
                iteration=7,
            )
    assert secret not in caplog.text
    assert "SUPER-SECRETO" not in caplog.text
    assert secret not in str(exc_info.value)
    assert "iteration=7" in str(exc_info.value)


def test_stats_agregadas_e_secao_cli(projects_root):
    updater, llm, _ = _make_updater(["resumo um", "", "resumo dois"])
    updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=1,
    )
    updater.update(
        objective="o", project_name="p", task=_task(), result="ok",
        iteration=2,
    )
    stats = llm.usage.project_summary_stats()
    assert stats["invocations"] == 2
    assert stats["attempts"] == 3
    assert stats["retries"] == 1
    assert stats["empty_responses"] == 1
    assert stats["successes"] == 2
    assert stats["by_iteration"] == {1: 1, 2: 2}
    assert stats["max_duration_ms"] >= stats["avg_duration_ms"] >= 0
    assert stats["max_prompt_chars"] > 0
    assert stats["avg_completion_tokens"] > 0

    section = llm.usage.project_summary_section()
    assert section is not None
    assert "calls: 2" in section
    assert "retries: 1" in section


def test_secao_cli_ausente_sem_chamadas():
    assert UsageTracker().project_summary_section() is None
    assert UsageTracker().project_summary_stats()["invocations"] == 0


def test_breakdown_e_summary_preservados():
    tracker = UsageTracker()
    from app.llm.models import Usage
    tracker.record(
        Usage(100, 5, 105), "fake", component="ProjectSummaryUpdater",
        iteration=1, prompt_chars=400, duration_ms=12.5,
    )
    assert "ProjectSummaryUpdater" in tracker.breakdown()
    assert "Chamadas à LLM: 1" in tracker.summary()


def test_client_registra_duracao_em_excecao():
    class Boom:
        def generate(self, messages, tools=None):
            raise LLMConnectionError("down")

    client = LLMClient(Boom())
    from app.llm.models import Message
    with pytest.raises(LLMConnectionError):
        client.generate(
            messages=[Message(role="user", content="oi")],
            component="ProjectSummaryUpdater", iteration=9,
        )
    records = client.usage.get_records()
    assert len(records) == 1
    assert records[0].success is False
    assert records[0].error == "LLMConnectionError"
    assert records[0].duration_ms >= 0.0
    assert records[0].iteration == 9


def test_cli_imprime_secao_quando_presente(capsys):
    from app.agent.events import AgentEvent
    from app.cli.events import handle_agent_event

    handle_agent_event(AgentEvent(
        type="agent_done",
        data={"updater_summary": "ProjectSummaryUpdater:\n  calls: 2"},
    ))
    assert "ProjectSummaryUpdater" in capsys.readouterr().out


def test_cli_omite_secao_ausente(capsys):
    from app.agent.events import AgentEvent
    from app.cli.events import handle_agent_event

    handle_agent_event(AgentEvent(type="agent_done", data={}))
    assert "ProjectSummaryUpdater" not in capsys.readouterr().out


def test_sinteticos_a_ate_f(projects_root):
    cases = {
        "A valido": (["resumo A"], "ok"),
        "B vazio->valido": (["", "resumo B"], "ok"),
        "C 2vazios->valido": (["", "", "resumo C"], "ok"),
        "D tudo vazio": (["", "", ""], "ValueError"),
        "E transitorio->valido": (
            [LLMConnectionError("timeout"), "resumo E"], "ok"),
        "F permanente": (
            [LLMAPIError("unauthorized", status_code=401)],
            "LLMAPIError"),
    }
    print("\nCaso | chamadas | retries | resultado | failures")
    for name, (script, expected) in cases.items():
        updater, llm, _ = _make_updater(list(script))
        try:
            updater.update(
                objective="o", project_name="p", task=_task(),
                result="ok", iteration=1,
            )
            result = "ok"
        except Exception as error:
            result = type(error).__name__
        stats = llm.usage.project_summary_stats()
        print(f"{name} | {stats['attempts']} | {stats['retries']} | "
              f"{result} | failures={stats['failures']}")
        assert result == expected
