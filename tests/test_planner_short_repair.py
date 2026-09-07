"""Fase 3 Etapa 2 — short repair prompt do Planner.

Cobre os 10 casos exigidos: decisão válida sem retry, primeiro retry
curto, curto que resolve (sem fallback), curto que falha (fallback
completo), investigation=true, erro de schema, erro de tool, limites,
trace (short vs full) e UsageTracker.
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import DecisionAction
from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.agent.runner import Runner
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage
from app.tools.registry import ToolRegistry


class ScriptedProvider:
    """Provider fake com roteiro de respostas e captura de prompts."""

    def __init__(self, contents, prompt_tokens=100, completion_tokens=10):
        self._contents = list(contents)
        self.prompts = []
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.name = "scripted"

    def generate(self, messages, tools=None):
        self.prompts.append(messages[0].content)
        item = self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            content=item,
            tool_calls=[],
            usage=Usage(self.prompt_tokens, self.completion_tokens,
                        self.prompt_tokens + self.completion_tokens),
            provider="scripted",
        )


def _task_json(tool, arguments, **extra):
    task = {"tool": tool, "arguments": arguments, "dependencies": []}
    task.update(extra)
    return json.dumps({"action": "task", "task": task})


def _finish_json(content="done"):
    return json.dumps({"action": "finish", "content": content})


def _real_planner(provider):
    tools = ToolRegistry()
    tools.load_defaults()
    return Planner(llm=LLMClient(provider),
                   parser=DecisionParser(tools=tools), tools=tools)


def _runner_with_planner(planner, executions, trace, **overrides):
    tools = ToolRegistry()
    tools.load_defaults()
    return Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=overrides.pop("validator", None) or _validator(tools),
        operational_memory=overrides.pop(
            "operational_memory", None) or T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )


def _validator(tools):
    from app.agent.execution.validator import TaskValidator
    return TaskValidator(tools)


def _trace(tmp_path):
    from app.agent.trace import ExecutionTrace
    return ExecutionTrace(trace_dir=str(tmp_path / "traces"))


def _retry_events(trace):
    return [e for e in trace.read_events() if e["event"] == "planner_retry"]


WRITE_ARGS = {"project_name": "p", "file_path": "a.py", "content": "x"}


# Caso 1 — decisão válida: nenhum retry ---------------------------------------

def test_valid_decision_has_no_retry(tmp_path, projects_root):
    provider = ScriptedProvider([
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(provider.prompts) == 2
    assert _retry_events(trace) == []
    assert all("CURRENT CONTEXT" in p for p in provider.prompts)


# Caso 2 — decisão inválida: primeiro retry usa short repair ------------------

def test_first_retry_uses_short_repair(tmp_path, projects_root):
    provider = ScriptedProvider([
        "isto não é json {{{",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    assert len(provider.prompts) == 3
    full, short, finish = provider.prompts
    assert "CURRENT CONTEXT" in full
    assert "CURRENT CONTEXT" not in short
    assert "PREVIOUS DECISION" in short
    assert "VALIDATION ERROR" in short
    assert "TARGETED FIX" in short
    assert len(short) < len(full) // 2

    retries = _retry_events(trace)
    assert len(retries) == 1
    assert retries[0]["retry_type"] == "short_repair"
    assert retries[0]["planner_attempt"] == 1
    assert retries[0]["prompt_chars"] == len(short)
    assert "reason" in retries[0]


# Caso 3 — short repair funciona: sem retry completo ---------------------------

def test_short_repair_success_skips_full_retry(tmp_path, projects_root):
    provider = ScriptedProvider([
        "lixo {{{",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    runner.run(objective="obj", project_name="p")

    assert len(provider.prompts) == 3
    assert "CORREÇÃO DA TENTATIVA ANTERIOR" not in "".join(
        provider.prompts)
    assert "ERRO REPETIDO" not in "".join(provider.prompts)
    assert [e["retry_type"] for e in _retry_events(trace)] == [
        "short_repair"]


# Caso 4 — short repair falha: fallback para prompt completo -------------------

def test_short_repair_failure_falls_back_to_full(tmp_path, projects_root):
    provider = ScriptedProvider([
        "primeiro lixo {{{",
        "segundo lixo }}}",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    assert len(provider.prompts) == 4
    full1, short, full2, finish = provider.prompts
    assert "CURRENT CONTEXT" in full1
    assert "CURRENT CONTEXT" not in short
    assert "CURRENT CONTEXT" in full2
    assert ("CORREÇÃO DA TENTATIVA ANTERIOR" in full2
            or "ERRO REPETIDO" in full2)

    retries = _retry_events(trace)
    assert [e["retry_type"] for e in retries] == [
        "short_repair", "full_context"]
    assert retries[1]["prompt_chars"] == len(full2)
    assert retries[0]["prompt_chars"] < retries[1]["prompt_chars"]


# Caso 5 — investigation=true via short repair ---------------------------------

def test_short_repair_fixes_investigation_flag(tmp_path, projects_root):
    read_args = {"project_name": "p", "file_path": "a.py"}
    provider = ScriptedProvider([
        _task_json("read_file", read_args),  # sem investigation → erro
        _task_json("read_file", read_args, investigation=True),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="read_file", arguments=read_args)],
        trace,
        operational_memory=T.RecordingOperationalMemory(
            free_pass=False, budget_available=False))

    assert runner.run(objective="obj", project_name="p") == "done"

    assert len(provider.prompts) == 3
    short = provider.prompts[1]
    assert "CURRENT CONTEXT" not in short
    assert '"investigation": true' in short
    assert [e["retry_type"] for e in _retry_events(trace)] == [
        "short_repair"]


# Caso 6 — erro de schema: repair traz o schema da tool -------------------------

def test_short_repair_fixes_schema_error(tmp_path, projects_root):
    bad_args = {"project_name": "p", "path": "a.py", "content": "x"}
    provider = ScriptedProvider([
        _task_json("write_file", bad_args),  # `path` não existe
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    short = provider.prompts[1]
    assert "CURRENT CONTEXT" not in short
    assert "file_path" in short  # schema compacto da write_file
    assert "required" in short


# Caso 7 — erro de tool: repair lista só o necessário ----------------------------

def test_short_repair_fixes_unknown_tool(tmp_path):
    planner = _real_planner(ScriptedProvider([]))
    prompt = planner.build_repair_prompt(
        error="Tool não encontrada: write_files",
        raw_response='{"action": "task", "task": {"tool": "write_files"}}',
        tool_name="write_files",
    )
    assert "write_file" in prompt
    assert "run_command" in prompt
    assert "CURRENT CONTEXT" not in prompt
    assert len(prompt) < 3000


def test_unknown_tool_fixed_end_to_end(tmp_path, projects_root):
    provider = ScriptedProvider([
        _task_json("write_files", WRITE_ARGS),  # tool inexistente
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    assert runner.run(objective="obj", project_name="p") == "done"
    short = provider.prompts[1]
    assert "write_file" in short
    assert "CURRENT CONTEXT" not in short


# Caso 8 — limite de attempts inalterado ------------------------------------------

def test_short_repair_respects_attempt_limit(tmp_path):
    from app.agent.runner import Runner as R
    assert R.MAX_PLANNER_ATTEMPTS == 5

    provider = ScriptedProvider(["lixo {{{"] * 6)
    trace = _trace(tmp_path)
    runner = _runner_with_planner(_real_planner(provider), [], trace)

    with pytest.raises(RuntimeError, match="limite de tentativas"):
        runner.run(objective="obj", project_name="p")

    # 1 full + 1 short + 1 full + 1 short + 1 full = 5 (mesmo orçamento).
    assert len(provider.prompts) == 5
    assert runner._stats.iterations == 0 or runner._stats.planner_calls == 5
    types = [e["retry_type"] for e in _retry_events(trace)]
    assert types == ["short_repair", "full_context",
                     "short_repair", "full_context"]


# Caso 9 — trace diferencia short e full ------------------------------------------

def test_trace_compares_short_vs_full_sizes(tmp_path, projects_root):
    provider = ScriptedProvider([
        "ruim 1 {{{",
        "ruim 2 }}}",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    runner.run(objective="obj", project_name="p")

    retries = _retry_events(trace)
    short_chars = retries[0]["prompt_chars"]
    full_chars = retries[1]["prompt_chars"]
    assert retries[0]["retry_type"] == "short_repair"
    assert retries[1]["retry_type"] == "full_context"
    assert isinstance(short_chars, int) and isinstance(full_chars, int)
    assert short_chars < full_chars
    # Todos os retries do trace têm iteração, attempt e motivo.
    for event in retries:
        assert event["iteration"] == 1
        assert event["planner_attempt"] >= 1
        assert event["reason"]


# Caso 10 — UsageTracker contabiliza e não quebra o formato -------------------------

def test_usage_tracker_counts_request_types(tmp_path, projects_root):
    provider = ScriptedProvider([
        "ruim {{{",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    planner = _real_planner(provider)
    runner = _runner_with_planner(
        planner,
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    runner.run(objective="obj", project_name="p")

    usage = planner.llm.usage
    assert usage.calls == 3
    stats = usage.planner_request_stats()
    assert stats["normal"]["calls"] == 2  # task-1 + finish
    assert stats["short_repair"]["calls"] == 1
    assert "full_retry" not in stats
    total = (stats["normal"]["total_tokens"]
             + stats["short_repair"]["total_tokens"])
    assert total == usage.total.total_tokens

    # Formato público do breakdown inalterado.
    breakdown = usage.breakdown()
    assert "========== TOKEN USAGE BREAKDOWN ==========" in breakdown
    assert "Planner" in breakdown
    assert "TOTAL" in breakdown


# Hints unitários + sem contexto vazado ---------------------------------------------

def test_repair_prompt_hints_and_no_full_context():
    planner = _real_planner(ScriptedProvider([]))

    dep = planner.build_repair_prompt(
        error="A tool 'read_file' só pode ser usada como dependency, "
              "não como task principal.",
        raw_response='{"action":"task"}', tool_name="read_file")
    assert "investigation" in dep

    js = planner.build_repair_prompt(
        error="A LLM retornou um JSON inválido.", raw_response="{{{")
    assert "JSON" in js

    generic = planner.build_repair_prompt(
        error="algo estranho", raw_response="???")
    assert "corrected decision JSON" in generic

    for prompt in (dep, js, generic):
        assert "CURRENT CONTEXT" not in prompt
        assert "RESUMO DO PROJETO" not in prompt
        assert len(prompt) < 4000


def test_no_stale_retry_context_in_next_iteration(tmp_path, projects_root):
    """Correção de uma iteração não vaza para a próxima (regressão)."""
    provider = ScriptedProvider([
        "ruim 1 {{{",
        "ruim 2 }}}",
        _task_json("write_file", WRITE_ARGS),
        _finish_json(),
    ])
    trace = _trace(tmp_path)
    runner = _runner_with_planner(
        _real_planner(provider),
        [ExecutionDecision(tool="write_file", arguments=WRITE_ARGS)],
        trace)

    runner.run(objective="obj", project_name="p")

    full_with_context = provider.prompts[2]
    assert ("CORREÇÃO DA TENTATIVA ANTERIOR" in full_with_context
            or "ERRO REPETIDO" in full_with_context)
    finish_prompt = provider.prompts[3]
    assert "CORREÇÃO DA TENTATIVA ANTERIOR" not in finish_prompt
    assert "ERRO REPETIDO" not in finish_prompt
