"""Unified Error Analysis (1 análise → N consumidores)."""

import json

import pytest

import tests.test_runner as T
from app.agent.context.error_checklist import ErrorChecklist
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.context.operational_memory import OperationalMemory
from app.agent.errors.error_analyzer import (
    UnifiedErrorAnalyzer,
    extract_failure_facts,
)
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.agent.taskstate.task_state import TaskState
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage


@pytest.fixture(autouse=True)
def _no_disk_persistence(monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", False)


FAIL_2 = (
    "STATUS: falha (exit code 1)\n"
    "FAILED test_cart.py::test_total - assert 10 == 12\n"
    "test_cart.py:42 AssertionError\n"
    "FAILED test_user.py::test_login - KeyError: 'token'\n"
    "src/user.py:7 KeyError\n"
)
PASS = "STATUS: sucesso (exit code 0)\nSTDOUT:\n2 passed\nSTDERR:\n(vazio)"


class _ScriptedLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.usage = T.FakeLLM.usage

    def generate(self, messages, component=None, iteration=None,
                 **kwargs):
        self.calls.append({"component": component,
                           "prompt": messages[0].content})
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content, tool_calls=[],
                           usage=Usage(10, 5, 15), provider="scripted")


def _analysis_json():
    return json.dumps({"problems": [
        {"error": "FAILED test_cart.py::test_total - assert 10 == 12",
         "test": "test_cart.py::test_total",
         "probable_cause": "total sums price without discount",
         "suggested_solution": "review Cart.total in test_cart.py",
         "affected_files": ["test_cart.py"]},
        {"error": "FAILED test_user.py::test_login - KeyError",
         "test": "test_user.py::test_login",
         "probable_cause": "unknown",
         "suggested_solution": "investigate",
         "affected_files": ["src/user.py", "ghost.py"]},
    ]})


def _task(tool, arguments, dependencies=None):
    return Task(tool=tool, arguments=arguments,
                dependencies=dependencies or [])


def _run_task(command):
    return _task("run_command", {"project_name": "p",
                                 "command": command})


def _write_task(path="a.py", content="x = 1\n"):
    return _task("write_file", {"project_name": "p", "file_path": path,
                                "content": content})


def _finish(content="done"):
    return Decision(action=DecisionAction.FINISH, content=content)


def _task_decision(task):
    return Decision(action=DecisionAction.TASK, task=task)


def _exec_for(task):
    return ExecutionDecision(tool=task.tool, arguments=task.arguments)


class _ScriptedTools(T.FakeToolRegistry):
    def __init__(self, run_outputs):
        self._run_outputs = list(run_outputs)
        self.ran_commands = []

    def execute(self, tool, arguments):
        if tool == "run_command":
            self.ran_commands.append(
                str((arguments or {}).get("command", "")))
            if self._run_outputs:
                return self._run_outputs.pop(0)
            return PASS
        return super().execute(tool, arguments)


def _runner(decisions, executions, trace, tools=None, **overrides):
    return Runner(
        planner=overrides.pop("planner", None)
        or T.FakePlanner(decisions),
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools or T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=overrides.pop(
            "operational_memory", None) or OperationalMemory(
                tools or T.FakeToolRegistry()),
        checklist=T.FakeChecklist(),
        error_checklist=overrides.pop(
            "error_checklist", None) or T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )


def _trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "trace6"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


def test_zero_errors_empty_no_call():
    unified = UnifiedErrorAnalyzer(
        llm=_ScriptedLLM(content="{}")).analyze("pytest", PASS)
    assert unified.problems == []
    assert unified.failures == []
    assert unified.llm_calls == 0
    assert unified.checklist_items == []


def test_single_failure_one_call():
    llm = _ScriptedLLM(content=_analysis_json())
    unified = UnifiedErrorAnalyzer(llm=llm).analyze(
        "python -m pytest -q", FAIL_2)
    assert len(llm.calls) == 1
    assert unified.llm_calls == 1
    assert unified.fallback_used is False
    assert len(unified.problems) == 2
    first = unified.problems[0]
    assert first.test == "test_cart.py::test_total"
    assert "discount" in first.probable_cause
    assert "Cart.total" in first.suggested_solution
    assert first.affected_files == ["test_cart.py"]
    assert len(unified.failures) == 2
    assert unified.failures[0].test == "test_cart.py::test_total"
    assert unified.exit_code == 1
    assert len(unified.checklist_items) == 2
    assert "test_total" in unified.checklist_items[0]


def test_ungrounded_file_dropped_but_problem_kept():
    llm = _ScriptedLLM(content=_analysis_json())
    unified = UnifiedErrorAnalyzer(llm=llm).analyze("pytest", FAIL_2)
    second = unified.problems[1]
    assert "ghost.py" not in second.affected_files
    assert "src/user.py" in second.affected_files


def test_four_failures_single_call():
    lines = "\n".join(f"FAILED t{i}.py::test_{i} boom" for i in range(4))
    files = " ".join(f"t{i}.py" for i in range(4))
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": f"FAILED t{i}.py::test_{i} boom"} for i in range(4)]}))
    unified = UnifiedErrorAnalyzer(llm=llm).analyze(
        "pytest", f"STATUS: falha (exit code 1)\n{lines}\n{files}")
    assert len(llm.calls) == 1
    assert len(unified.problems) == 4
    assert len(unified.failures) == 4


def test_extract_failure_facts_unit():
    facts = extract_failure_facts(FAIL_2)
    assert len(facts) == 2
    assert facts[0].test == "test_cart.py::test_total"
    assert "test_cart.py" in facts[0].files
    assert extract_failure_facts(PASS) == []
    assert extract_failure_facts(None) == []
    many = "\n".join(f"FAILED t{i}.py::x" for i in range(30))
    assert len(extract_failure_facts(f"STATUS: falha\n{many}")) == 20


@pytest.mark.parametrize("bad,reason", [
    ("not json {{{", "invalid_response"),
    (json.dumps({"nope": []}), "invalid_response"),
    (json.dumps({"problems": "x"}), "invalid_response"),
    ("", "invalid_response"),
    (None, "invalid_response"),
])
def test_invalid_responses_fall_back(bad, reason):
    llm = _ScriptedLLM(content=bad)
    unified = UnifiedErrorAnalyzer(llm=llm).analyze("pytest", FAIL_2)
    assert unified.fallback_used is True
    assert unified.error == reason
    assert len(unified.problems) == 2
    assert unified.problems[0].probable_cause == "unknown"
    assert len(unified.checklist_items) == 2
    assert len(unified.failures) == 2


def test_llm_error_falls_back_without_invention():
    llm = _ScriptedLLM(error=TimeoutError("slow"))
    unified = UnifiedErrorAnalyzer(llm=llm).analyze("pytest", FAIL_2)
    assert unified.fallback_used is True
    assert unified.error == "llm_error: TimeoutError"
    assert unified.llm_calls == 1
    assert len(unified.problems) == 2


def test_no_evidence_no_problems_no_invention():
    unified = UnifiedErrorAnalyzer(llm=None).analyze(
        "pytest", "STATUS: falha (exit code 1)\nblabla sem padrão")
    assert unified.problems == []
    assert unified.failures == []
    assert unified.checklist_items == []
    assert unified.llm_calls == 0


def test_llm_none_deterministic_only():
    unified = UnifiedErrorAnalyzer(llm=None).analyze("pytest", FAIL_2)
    assert unified.llm_calls == 0
    assert unified.fallback_used is True
    assert len(unified.problems) == 2
    assert len(unified.checklist_items) == 2


def test_apply_unified_from_problems():
    from app.agent.context.error_checklist import ErrorChecklist
    llm = _ScriptedLLM(content=_analysis_json())
    unified = UnifiedErrorAnalyzer(llm=llm).analyze("pytest", FAIL_2)
    checklist = ErrorChecklist(llm)
    assert checklist.apply_unified(unified) == 2
    assert checklist.pending_count == 2
    rendered = checklist.render()
    assert "test_total" in rendered
    assert "test_cart.py" in rendered
    assert [c for c in llm.calls
            if c["component"] == "ErrorChecklist"] == []
    assert len(llm.calls) == 1


def test_apply_unified_from_failures_only():
    from app.agent.context.error_checklist import ErrorChecklist
    unified = UnifiedErrorAnalyzer(llm=None).analyze("pytest", FAIL_2)
    checklist = ErrorChecklist(None)
    assert checklist.apply_unified(unified) == 2
    assert "test_total" in checklist.render()


def test_apply_unified_empty_clears():
    from app.agent.context.error_checklist import ErrorChecklist
    from app.agent.errors.error_analyzer import UnifiedErrorAnalyzer
    checklist = ErrorChecklist(None)
    full = UnifiedErrorAnalyzer(llm=None).analyze("pytest", FAIL_2)
    assert checklist.apply_unified(full) == 2
    empty = UnifiedErrorAnalyzer(llm=None).analyze("pytest", PASS)
    assert checklist.apply_unified(empty) == 0
    assert checklist.render() is None


def test_deterministic_projection_no_llm():
    from app.agent.context.error_checklist import ErrorChecklist
    from app.agent.errors.error_analyzer import UnifiedErrorAnalyzer
    llm = _ScriptedLLM(content="UNUSED")
    unified = UnifiedErrorAnalyzer(llm=None).analyze("pytest", FAIL_2)
    checklist = ErrorChecklist(llm)
    assert checklist.apply_unified(unified) == 2
    assert llm.calls == []
    assert "test_login" in checklist.render()
    assert checklist._items[0].command == "pytest"


def test_four_errors_cost_one_analysis_call():
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": f"FAILED t{i}.py::test_{i}"} for i in range(4)]}))
    lines = "\n".join(f"FAILED t{i}.py::test_{i}" for i in range(4))
    unified = UnifiedErrorAnalyzer(llm=llm).analyze(
        "pytest", f"STATUS: falha\n{lines}")
    total_analysis_calls = len(llm.calls)
    assert total_analysis_calls == 1
    assert len(unified.problems) == 4


def test_runner_level_single_analysis_call(tmp_path):
    from app.agent.context.error_checklist import ErrorChecklist

    class _Provider:
        name = "s"

        def __init__(self, contents):
            self._contents = list(contents)

        def generate(self, messages, tools=None):
            item = self._contents.pop(0)
            return LLMResponse(content=item, tool_calls=[],
                               usage=Usage(10, 5, 15), provider="s")

    provider = _Provider([_analysis_json()])
    llm = LLMClient(provider)
    fail = _run_task("python -m pytest -q")
    fix = _write_task(path="test_cart.py")
    cart_only_fail = (
        "STATUS: falha (exit code 1)\n"
        "FAILED test_cart.py::test_total - assert 10 == 12\n"
        "test_cart.py:42 AssertionError\n"
        "FAILED test_cart.py::test_add - assert 1 == 2\n")
    scripted_tools = _ScriptedTools([cart_only_fail, PASS])
    trace = _trace(tmp_path)
    planner = T.FakePlanner(
        [_task_decision(fail), _task_decision(fix),
         _task_decision(fail), _finish()])
    planner.llm = llm
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker(
            [_exec_for(fail), _exec_for(fix), _exec_for(fail)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=scripted_tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(scripted_tools),
        checklist=T.FakeChecklist(),
        error_checklist=ErrorChecklist(_ScriptedLLM()),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    assert runner.run(objective="obj", project_name="p") == "done"
    assert provider._contents == []
    components = [r.component for r in llm.usage.get_records()]
    assert components == ["ErrorAnalyzer"]
    assert len(runner.task_state.problems) == 2


def test_unified_feeds_finish_gate(tmp_path):
    from app.agent.context.error_checklist import ErrorChecklist
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    fix_cart = _write_task(path="test_cart.py")
    fix_user = _write_task(path="test_user.py")
    tools = _ScriptedTools([FAIL_2, PASS])
    runner = _runner(
        [_task_decision(fail), _finish("a"), _task_decision(fix_cart),
         _task_decision(fix_user), _task_decision(fail),
         _finish("b")],
        [_exec_for(fail), _exec_for(fix_cart), _exec_for(fix_user),
         _exec_for(fail)],
        trace, tools=tools,
        error_checklist=ErrorChecklist(_ScriptedLLM()))
    assert runner.run(objective="obj", project_name="p") == "b"
    assert runner._stats.finish_blocks == 1


def test_planner_receives_structured_problems():
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append(context)
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_2])
    runner = _runner(
        [_task_decision(fail), _finish()], [_exec_for(fail)],
        NullTrace(), tools=tools,
        planner=_CatchingPlanner(
            [_task_decision(fail), _finish()]))
    runner.run(objective="obj", project_name="p")
    context = seen[1]
    assert "test_cart.py::test_total" in context
    assert "files: test_cart.py" in context
    assert "TASK STATE:" in context


def test_planner_context_stays_compact():
    state = TaskState(objective="o")
    for i in range(10):
        state.problems.append(__import__(
            "app.agent.taskstate.task_state",
            fromlist=["Problem"]).Problem(
                description=f"problem {i} failed",
                problem_id=f"p{i:03d}",
                error=f"error {i}",
                probable_cause="some cause text here",
                suggested_solution="some fix text here",
                affected_files=[f"f{i}.py", f"g{i}.py"],
                status="pending", test=f"t{i}.py::x"))
    block = state.render_compact(include_objective=False,
                                 original_ref_chars=0)
    assert len(block) < 6000
    assert block.count("cause:") == 8
    assert "[+2 more]" in block


def test_executor_gets_detail_without_raw_output():
    state = TaskState(objective="o")
    state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a",
                        "test": "a.py::t",
                        "probable_cause": "c",
                        "suggested_solution": "s",
                        "affected_files": ["a.py"]})()], source="pytest")
    slim = state.render_compact(include_objective=False,
                                problem_detail=False)
    assert "cause:" not in slim and "fix:" not in slim
    assert "PROBLEMS: 1 open" in slim
    detail = state.render_known_problems()
    assert "Probable cause: c" in detail
    assert "Suggested solution: s" in detail
    assert "STDOUT" not in detail and "Traceback" not in detail


@pytest.mark.parametrize("bad", [
    "not json", "", None,
    json.dumps({"problems": []}),
    json.dumps({"wrong": 1}),
])
def test_runner_fallback_variants(tmp_path, bad):
    llm = _ScriptedLLM(content=bad)
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_2])
    runner = _runner([_task_decision(fail), _finish()],
                     [_exec_for(fail)], trace, tools=tools)
    runner.planner.llm = llm
    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(runner.task_state.problems) == 2
    assert all(p.probable_cause == "unknown"
               for p in runner.task_state.problems)
    completed = _events(trace, "error_analysis_completed")
    assert len(completed) == 1
    assert completed[0]["fallback"] is True


def test_runner_timeout_fallback(tmp_path):
    llm = _ScriptedLLM(error=TimeoutError("slow"))
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_2])
    runner = _runner([_task_decision(fail), _finish()],
                     [_exec_for(fail)], trace, tools=tools)
    runner.planner.llm = llm
    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(runner.task_state.problems) == 2
    completed = _events(trace, "error_analysis_completed")
    assert completed[0]["fallback"] is True
    assert "TimeoutError" in completed[0]["error"]


def test_structured_persist_roundtrip(projects_root, tmp_path,
                                      monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", True)
    from app.agent.taskstate.persistence import (
        load_task_state,
        save_task_state,
    )
    state = TaskState(task_id="t6", objective="o")
    state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a", "test": "a.py::t",
                        "probable_cause": "c", "suggested_solution": "s",
                        "affected_files": ["a.py"]})()], source="pytest")
    state.apply_correction("fix", ["a.py"], 1)
    assert save_task_state(state, "p")["ok"] is True
    loaded = load_task_state("p")
    assert loaded == state
    assert loaded.problems[0].probable_cause == "c"
    assert loaded.problems[0].affected_files == ["a.py"]
    assert loaded.corrections[0].files_changed == ["a.py"]


def test_unified_trace_fields(tmp_path):
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_2])
    runner = _runner([_task_decision(fail), _finish()],
                     [_exec_for(fail)], trace, tools=tools)
    runner.run(objective="obj", project_name="p")
    completed = _events(trace, "error_analysis_completed")
    assert len(completed) == 1
    event = completed[0]
    assert event["problems"] == 2
    assert event["checklist_items"] == 2
    assert event["exit_code"] == 1
    assert len(event["created"]) == 2
    assert "llm_calls" in event
