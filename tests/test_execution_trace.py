"""Fase 3 — trace persistente e estruturado de execução (observabilidade).

Cobre: criação do trace, escrita de eventos, run_id consistente,
iteração, planner retry, executor error, teste sucesso/falha, finish
block, FinalVerification, não-armazenamento de conteúdo sensível e
resiliência da escrita (nunca quebra a run).
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.agent.trace import (
    ExecutionTrace,
    NullTrace,
    sanitize_arguments,
    sanitize_result,
)
from app.tools.filesystem.write_file import write_file


def _make_trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "traces"))


def _events_by_type(trace):
    by_type = {}
    for event in trace.read_events():
        by_type.setdefault(event["event"], []).append(event)
    return by_type


def _run_with_trace(decisions, executions=None, tmp_path=None,
                    trace=None, **kwargs):
    runner = T._make_runner(decisions, executions, **kwargs)
    runner.execution_trace = trace if trace is not None else _make_trace(
        tmp_path)
    return runner


# 1. criação de um novo trace ------------------------------------------------

def test_trace_creates_unique_file_per_execution(tmp_path):
    first = ExecutionTrace(trace_dir=str(tmp_path))
    second = ExecutionTrace(trace_dir=str(tmp_path))

    assert first.run_id != second.run_id
    assert first.path != second.path
    assert first.path.suffix == ".jsonl"
    assert str(first.run_id) in str(first.path)


# 2. eventos escritos corretamente --------------------------------------------

def test_trace_writes_well_formed_jsonl(tmp_path):
    trace = _make_trace(tmp_path)
    trace.record("iteration_start", iteration=3)

    lines = trace.path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "iteration_start"
    assert payload["iteration"] == 3
    assert "run_id" in payload and "timestamp" in payload


# 3. múltiplos eventos (ordem preservada) -------------------------------------

def test_trace_preserves_order_of_multiple_events(tmp_path):
    trace = _make_trace(tmp_path)
    for name in ("run_start", "iteration_start", "planner_decision",
                 "run_end"):
        trace.record(name, iteration=1)

    events = trace.read_events()
    assert [e["event"] for e in events] == [
        "run_start", "iteration_start", "planner_decision", "run_end"]
    assert trace.event_count == 4


# 4. run_id consistente --------------------------------------------------------

def test_trace_run_id_consistent_across_events(tmp_path):
    trace = _make_trace(tmp_path)
    runner = _run_with_trace(
        [Decision(action=DecisionAction.FINISH, content="done")],
        tmp_path=tmp_path, trace=trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    events = trace.read_events()
    assert events, "trace deve conter eventos"
    assert {e["run_id"] for e in events} == {trace.run_id}


# 5. iteração registrada --------------------------------------------------------

def test_trace_records_iteration_flow(tmp_path):
    trace = _make_trace(tmp_path)
    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": "x"})
    runner = _run_with_trace(
        [Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")],
        [ExecutionDecision(tool="write_file",
                           arguments=task.arguments)],
        tmp_path=tmp_path, trace=trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    by_type = _events_by_type(trace)
    assert [e["iteration"] for e in by_type["iteration_start"]] == [1, 2]
    decision = by_type["planner_decision"][0]
    assert decision["iteration"] == 1
    assert decision["decision"] == "task"
    assert decision["tool"] == "write_file"
    assert by_type["run_end"][0]["outcome"] == "success"


# 6. Planner retry registrado ---------------------------------------------------

def test_trace_records_planner_retry(tmp_path):
    trace = _make_trace(tmp_path)
    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": "x"})
    runner = _run_with_trace(
        [ValueError("JSON inválido de teste"),
         Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")],
        [ExecutionDecision(tool="write_file",
                           arguments=task.arguments)],
        tmp_path=tmp_path, trace=trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    by_type = _events_by_type(trace)
    errors = by_type["planner_error"]
    assert len(errors) == 1
    assert errors[0]["iteration"] == 1
    assert errors[0]["planner_attempt"] == 1
    assert errors[0]["error_type"] == "ValueError"
    assert "JSON inválido" in errors[0]["error_signature"]


# 7. Executor error registrado ---------------------------------------------------

def test_trace_records_executor_error(tmp_path):
    trace = _make_trace(tmp_path)
    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": "x"})

    class FlakyExecutor(T.FakeTaskDecisionMaker):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def decide(self, objective, task, context, iteration=None):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("decisão inválida de teste")
            return ExecutionDecision(tool="write_file",
                                     arguments=task.arguments)

    runner = T._make_runner(
        [Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")])
    runner.task_decision_maker = FlakyExecutor()
    runner.execution_trace = trace

    assert runner.run(objective="obj", project_name="p") == "done"

    by_type = _events_by_type(trace)
    errors = by_type["executor_error"]
    assert len(errors) == 1
    assert errors[0]["iteration"] == 1
    assert errors[0]["executor_attempt"] == 1
    assert errors[0]["error_type"] == "ValueError"
    decisions = by_type["executor_decision"]
    assert len(decisions) == 1
    assert decisions[0]["executor_attempt"] == 2


def _run_command_runner(tmp_path, trace, test_file_content):
    """Runner com tools reais que roda `python3 -m pytest -q` de verdade."""
    from app.agent.context.operational_memory import OperationalMemory
    from app.tools.registry import ToolRegistry

    project = "traceproj"
    write_file(project, "test_sample.py", test_file_content)

    tools = ToolRegistry()
    tools.load_defaults()
    task = Task(tool="run_command",
                arguments={"project_name": project,
                           "command": "python3 -m pytest -q"})
    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             Decision(action=DecisionAction.FINISH, content="done")]),
        task_decision_maker=T.FakeTaskDecisionMaker(
            [ExecutionDecision(tool="run_command",
                               arguments=task.arguments)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    return runner


# 8a. teste com sucesso registrado ------------------------------------------------

def test_trace_records_passing_test(projects_root, tmp_path):
    trace = _make_trace(tmp_path)
    runner = _run_command_runner(
        tmp_path, trace, "def test_ok():\n    assert 1 + 1 == 2\n")

    runner.run(objective="obj", project_name="traceproj")

    by_type = _events_by_type(trace)
    results = by_type["test_result"]
    assert len(results) == 1
    assert results[0]["success"] is True
    assert results[0]["exit_code"] == 0
    assert "pytest" in results[0]["command"]
    assert results[0]["iteration"] == 1


# 8b. teste com falha registrado ---------------------------------------------------

def test_trace_records_failing_test(projects_root, tmp_path):
    trace = _make_trace(tmp_path)
    runner = _run_command_runner(
        tmp_path, trace, "def test_broken():\n    assert 1 + 1 == 3\n")

    runner.run(objective="obj", project_name="traceproj")

    by_type = _events_by_type(trace)
    results = by_type["test_result"]
    assert len(results) == 1
    assert results[0]["success"] is False
    assert results[0]["exit_code"] != 0


# 9. finish block registrado ---------------------------------------------------------

def test_trace_records_finish_block(tmp_path):
    trace = _make_trace(tmp_path)

    class OnceProblems(T.FakeFinalVerification):
        def __init__(self):
            super().__init__(
                result=FinalVerificationResult(
                    FinalVerificationResult.OK))

        def verify(self, objective, project_name, summary,
                   tools_execute):
            if not self.verify_calls:
                self.verify_calls.append({})
                return FinalVerificationResult(
                    FinalVerificationResult.PROBLEMS_FOUND,
                    "símbolo X ausente")
            self.verify_calls.append({})
            return FinalVerificationResult(FinalVerificationResult.OK)

    runner = _run_with_trace(
        [Decision(action=DecisionAction.FINISH, content="a"),
         Decision(action=DecisionAction.FINISH, content="b")],
        tmp_path=tmp_path, trace=trace,
        final_verification=OnceProblems())

    assert runner.run(objective="obj", project_name="p") == "b"

    by_type = _events_by_type(trace)
    assert len(by_type["finish_requested"]) == 2
    blocks = by_type["finish_block"]
    assert len(blocks) == 1
    assert blocks[0]["gate"] == "final_verification"
    assert blocks[0]["iteration"] == 1
    gates = [e for e in by_type["finish_gate"]
             if e["gate"] == "final_verification"]
    assert [g["passed"] for g in gates] == [False, True]


# 10. FinalVerification registrada --------------------------------------------------------

def test_trace_records_final_verification_statuses(tmp_path):
    trace = _make_trace(tmp_path)
    runner = _run_with_trace(
        [Decision(action=DecisionAction.FINISH, content="done")],
        tmp_path=tmp_path, trace=trace)

    assert runner.run(objective="obj", project_name="p") == "done"

    by_type = _events_by_type(trace)
    verifications = by_type["final_verification"]
    assert len(verifications) == 1
    assert verifications[0]["status"] == "ok"
    assert verifications[0]["iteration"] == 1


# 11. conteúdo sensível não é armazenado ------------------------------------------

def test_trace_never_stores_write_file_content(tmp_path):
    secret = "SUPER-SECRETO-" * 500
    args = {"project_name": "p", "file_path": "a.py", "content": secret}

    sanitized = sanitize_arguments("write_file", args)
    assert sanitized["content"] == f"<omitted: {len(secret)} chars>"
    # dict original intacto (nunca muta).
    assert args["content"] == secret

    trace = _make_trace(tmp_path)
    trace.record("tool_result", iteration=1, tool="write_file",
                 arguments=sanitized, result_summary="ok")
    raw = trace.path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "<omitted:" in raw


def test_trace_redacts_secret_like_keys_and_truncates_output(tmp_path):
    sanitized = sanitize_arguments(
        "run_command",
        {"project_name": "p", "command": "deploy",
         "api_key": "CHAVE-ULTRA-SECRETA-123"})
    assert sanitized["api_key"] == "<redacted>"

    big = "L" * 100000
    assert len(sanitize_result(big)) < len(big)
    assert "CHAVE-ULTRA-SECRETA-123" not in sanitize_result("x")

    trace = _make_trace(tmp_path)
    runner = _run_with_trace(
        [Decision(action=DecisionAction.FINISH, content="done")],
        tmp_path=tmp_path, trace=trace)
    runner.run(objective="obj", project_name="p")
    assert "CHAVE-ULTRA-SECRETA-123" not in trace.path.read_text(
        encoding="utf-8")


def test_trace_end_to_end_omits_secret_file_content(tmp_path):
    secret = "SEGREDO-E2E-" * 300
    trace = _make_trace(tmp_path)
    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": secret})
    runner = _run_with_trace(
        [Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")],
        [ExecutionDecision(tool="write_file",
                           arguments=task.arguments)],
        tmp_path=tmp_path, trace=trace)

    assert runner.run(objective="obj", project_name="p") == "done"
    assert secret not in trace.path.read_text(encoding="utf-8")


# 12. falha na escrita não interrompe a run --------------------------------------------

def test_trace_record_never_raises(tmp_path):
    trace = _make_trace(tmp_path)
    trace.trace_dir = tmp_path / "arquivo"
    trace.trace_dir.write_text("sou um arquivo, não um diretório")
    trace.path = trace.trace_dir / "impossivel.jsonl"

    # Não levanta, mesmo com disco/paths quebrados.
    trace.record("iteration_start", iteration=1)
    assert trace.event_count == 0


def test_runner_completes_when_trace_is_broken(tmp_path):
    trace = _make_trace(tmp_path)
    trace.trace_dir = tmp_path / "arquivo"
    trace.trace_dir.write_text("sou um arquivo, não um diretório")
    trace.path = trace.trace_dir / "impossivel.jsonl"

    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": "x"})
    runner = _run_with_trace(
        [ValueError("retry com trace quebrado"),
         Decision(action=DecisionAction.TASK, task=task),
         Decision(action=DecisionAction.FINISH, content="done")],
        [ExecutionDecision(tool="write_file",
                           arguments=task.arguments)],
        tmp_path=tmp_path, trace=trace)

    # A run completa normalmente apesar do trace inoperante.
    assert runner.run(objective="obj", project_name="p") == "done"
    assert runner._stats.planner_calls == 3
    assert runner._stats.corrections == 1


def test_null_trace_disables_persistence():
    trace = NullTrace()
    trace.record("anything", iteration=1)
    assert trace.read_events() == []

    runner = T._make_runner(
        [Decision(action=DecisionAction.FINISH, content="done")])
    runner.execution_trace = trace
    assert runner.run(objective="obj", project_name="p") == "done"
