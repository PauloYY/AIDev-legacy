"""Error Analyzer + ciclo de correção + test gate."""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.errors.error_analyzer import (
    ErrorAnalyzer,
    normalize_error,
    normalize_test_id,
)
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.agent.taskstate.task_state import (
    PROBLEM_STATUS_BLOCKED,
    PROBLEM_STATUS_IN_PROGRESS,
    PROBLEM_STATUS_INVALIDATED,
    PROBLEM_STATUS_PENDING,
    PROBLEM_STATUS_PENDING_VERIFICATION,
    PROBLEM_STATUS_RESOLVED,
    TaskState,
)
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage


@pytest.fixture(autouse=True)
def _no_disk_persistence(monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", False)


FAIL_3 = (
    "STATUS: failure (exit code 1)\n"
    "FAILED test_cart.py::test_total - assert 10 == 12\n"
    "test_cart.py:42 AssertionError\n"
    "FAILED test_cart.py::test_add - assert 1 == 2\n"
    "FAILED test_user.py::test_login - KeyError: 'token'\n"
    "src/cart.py:10 KeyError\n"
)

PASS = "STATUS: success (exit code 0)\nSTDOUT:\n3 passed\nSTDERR:\n(empty)"


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
    """Outputs roteirizados por chamada run_command."""

    def __init__(self, run_outputs, write_errors=0):
        self._run_outputs = list(run_outputs)
        self._write_errors = write_errors
        self.ran_commands = []

    def execute(self, tool, arguments):
        if tool == "run_command":
            self.ran_commands.append(
                str((arguments or {}).get("command", "")))
            if self._run_outputs:
                return self._run_outputs.pop(0)
            return PASS
        if tool == "write_file" and self._write_errors > 0:
            self._write_errors -= 1
            raise RuntimeError("disk exploded")
        return super().execute(tool, arguments)


def _runner(decisions, executions, trace, tools=None, **overrides):
    from app.agent.context.operational_memory import OperationalMemory
    return Runner(
        planner=T.FakePlanner(decisions),
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
    return ExecutionTrace(trace_dir=str(tmp_path / "trace5"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


def test_no_error_no_problems_no_call():
    analysis = ErrorAnalyzer(llm=_ScriptedLLM(content="{}")).analyze(
        "python -m pytest -q", PASS)
    assert analysis.problems == []
    assert analysis.llm_calls == 0


def test_single_failure_fallback():
    analysis = ErrorAnalyzer().analyze(
        "python -m pytest -q",
        "STATUS: failure (exit code 1)\nFAILED test_x.py::test_a\nboom")
    assert len(analysis.problems) == 1
    problem = analysis.problems[0]
    assert problem.test == "test_x.py::test_a"
    assert "FAILED" in problem.error
    assert problem.probable_cause == "unknown"
    assert problem.suggested_solution == "investigate"
    assert analysis.fallback_used is True


def test_multiple_failures_single_call():
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": "FAILED a", "test": "a.py::t1",
         "probable_cause": "c1", "suggested_solution": "s1",
         "affected_files": ["a.py"]},
        {"error": "FAILED b", "test": "b.py::t2",
         "probable_cause": "unknown", "suggested_solution": "s2",
         "affected_files": []},
        {"error": "FAILED c", "test": "", "affected_files": []},
    ]}))
    out = ("STATUS: failure (exit code 1)\nFAILED a\nFAILED b\nFAILED c\n"
           "a.py b.py")
    analysis = ErrorAnalyzer(llm=llm).analyze("pytest", out)
    assert len(llm.calls) == 1
    assert llm.calls[0]["component"] == "ErrorAnalyzer"
    assert len(analysis.problems) == 3
    assert analysis.problems[0].probable_cause == "c1"
    assert analysis.problems[0].affected_files == ["a.py"]
    assert analysis.problems[2].probable_cause == "unknown"
    assert analysis.fallback_used is False


def test_five_errors_one_call():
    lines = "\n".join(f"FAILED t{i}.py::test_{i} boom" for i in range(5))
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": f"FAILED t{i}.py::test_{i} boom"} for i in range(5)]}
    ))
    analysis = ErrorAnalyzer(llm=llm).analyze(
        "pytest", f"STATUS: failure (exit code 1)\n{lines}\n" + " ".join(
            f"t{i}.py" for i in range(5)))
    assert len(llm.calls) == 1
    assert len(analysis.problems) == 5


def test_ungrounded_files_dropped():
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": "FAILED a", "test": "a.py::t",
         "affected_files": ["a.py", "invented_service.py"]}]}))
    analysis = ErrorAnalyzer(llm=llm).analyze(
        "pytest", "STATUS: failure\nFAILED a\na.py")
    assert analysis.problems[0].affected_files == ["a.py"]


def test_invalid_response_falls_back():
    for bad in ("not json {{{", json.dumps({"nope": 1}),
                json.dumps({"problems": "x"}),
                json.dumps({"problems": []}), "", None):
        llm = _ScriptedLLM(content=bad)
        analysis = ErrorAnalyzer(llm=llm).analyze(
            "pytest", "STATUS: failure (exit code 1)\nFAILED t.py::x\n"
                      "t.py boom")
        assert analysis.fallback_used is True
        assert len(analysis.problems) == 1
        assert analysis.problems[0].test == "t.py::x"


def test_entries_without_evidence_skipped():
    llm = _ScriptedLLM(content=json.dumps({"problems": [
        {"error": "  ", "test": "x"}, "string", {"no_error": 1},
        {"error": "FAILED real", "test": "r.py::t"}]}))
    analysis = ErrorAnalyzer(llm=llm).analyze(
        "pytest", "STATUS: failure\nFAILED real\nr.py")
    assert [p.error for p in analysis.problems] == ["FAILED real"]


def test_llm_error_falls_back():
    llm = _ScriptedLLM(error=TimeoutError("slow"))
    analysis = ErrorAnalyzer(llm=llm).analyze(
        "pytest", "STATUS: failure (exit code 1)\nFAILED t.py::x")
    assert analysis.fallback_used is True
    assert analysis.error == "llm_error: TimeoutError"
    assert len(analysis.problems) == 1


def test_no_evidence_no_problems():
    analysis = ErrorAnalyzer().analyze(
        "pytest", "STATUS: failure (exit code 1)\nblablabla sem padrão")
    assert analysis.problems == []
    assert analysis.fallback_used is True


def test_llm_none_uses_fallback():
    analysis = ErrorAnalyzer(llm=None).analyze(
        "pytest", "STATUS: failure (exit code 1)\nFAILED t.py::x")
    assert analysis.llm_calls == 0
    assert len(analysis.problems) == 1


def test_normalize_helpers():
    assert normalize_test_id("FAILED test_cart.py::test_total - x") == (
        "test_cart.py::test_total")
    assert normalize_test_id("src/a.py::t (extra)") == "a.py::t"
    assert normalize_test_id("no test here") == ""
    assert normalize_test_id(None) == ""
    assert normalize_error("  a   b\nc  ") == "a b c"
    assert len(normalize_error("x" * 500)) == 300


def test_max_problems_cap():
    lines = "\n".join(f"FAILED t{i}.py::test_{i}" for i in range(30))
    analysis = ErrorAnalyzer(max_problems=5).analyze(
        "pytest", f"STATUS: failure\n{lines}")
    assert len(analysis.problems) == 5


def test_analyzer_tracked_as_component():
    from app.llm.usage import UsageTracker
    from app.tools.registry import ToolRegistry

    class _Provider:
        name = "p"

        def generate(self, messages, tools=None):
            return LLMResponse(
                content=json.dumps({"problems": [
                    {"error": "FAILED a"}]}),
                tool_calls=[], usage=Usage(7, 3, 10), provider="p")

    client = LLMClient(_Provider())
    analysis = ErrorAnalyzer(llm=client).analyze(
        "pytest", "STATUS: failure\nFAILED a\na.py")
    assert len(analysis.problems) == 1
    stats = client.usage.component_stats()
    assert stats["ErrorAnalyzer"]["calls"] == 1
    assert stats["ErrorAnalyzer"]["total_tokens"] == 10
    tracker = UsageTracker()
    assert "ErrorAnalyzer" not in tracker.component_stats()


def test_problem_defaults_and_status_flow():
    state = TaskState()
    created, invalidated = state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a", "test": "a.py::t",
                        "probable_cause": "c", "suggested_solution": "s",
                        "affected_files": ["a.py"]})()],
        source="pytest", iteration=1)
    assert created == ["p001"] and invalidated == []
    problem = state.problems[0]
    assert problem.status == PROBLEM_STATUS_PENDING
    assert problem.resolved is False
    assert state.has_blocking_problems() is True
    marked = state.mark_attempt_started(["a.py"])
    assert marked == ["p001"]
    assert problem.status == PROBLEM_STATUS_IN_PROGRESS
    correction = state.apply_correction("fix a", ["a.py"], 2)
    assert correction.correction_id == "c001"
    assert correction.problem_ids == ["p001"]
    assert problem.status == PROBLEM_STATUS_PENDING_VERIFICATION
    assert problem.correction_id == "c001"
    assert state.has_blocking_problems() is False
    assert state.resolve_problems() == 1
    assert problem.status == PROBLEM_STATUS_RESOLVED
    assert problem.resolved is True


def test_attempt_failure_blocks():
    state = TaskState()
    state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a", "test": "",
                        "probable_cause": "u", "suggested_solution": "i",
                        "affected_files": []})()], source="pytest")
    state.mark_attempt_started([])
    assert state.problems[0].status == PROBLEM_STATUS_IN_PROGRESS
    marked = state.mark_attempt_failed([])
    assert marked == ["p001"]
    assert state.problems[0].status == PROBLEM_STATUS_BLOCKED
    assert state.has_blocking_problems() is True


def test_disjoint_files_not_linked():
    state = TaskState()
    from app.agent.errors.error_analyzer import AnalyzedProblem
    state.add_analyzed_problems(
        [AnalyzedProblem(error="FAILED a", test="a.py::t",
                         affected_files=["a.py"]),
         AnalyzedProblem(error="FAILED b", test="b.py::t",
                         affected_files=["b.py"])], source="pytest")
    correction = state.apply_correction("fix a", ["a.py"], 1)
    assert correction.problem_ids == ["p001"]
    assert state.problems[0].status == PROBLEM_STATUS_PENDING_VERIFICATION
    assert state.problems[1].status == PROBLEM_STATUS_PENDING
    assert state.has_blocking_problems() is True


def test_correction_without_open_problems_is_none():
    state = TaskState()
    assert state.apply_correction("fix", ["a.py"], 1) is None
    assert state.corrections == []


def test_exact_rededup_and_invalidation():
    from app.agent.errors.error_analyzer import AnalyzedProblem
    state = TaskState()
    state.add_analyzed_problems(
        [AnalyzedProblem(error="FAILED a", test="a.py::t")],
        source="pytest", iteration=1)
    created, invalidated = state.add_analyzed_problems(
        [AnalyzedProblem(error="FAILED a", test="a.py::t")],
        source="pytest", iteration=2)
    assert created == [] and invalidated == []
    assert len(state.problems) == 1
    created, invalidated = state.add_analyzed_problems(
        [AnalyzedProblem(error="FAILED a differently",
                         test="a.py::t")], source="pytest", iteration=3)
    assert created == ["p002"] and invalidated == ["p001"]
    assert state.problems[0].status == PROBLEM_STATUS_INVALIDATED
    assert state.open_problems == [state.problems[1]]


def test_structured_serialization_roundtrip():
    state = TaskState(task_id="t", objective="o")
    state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a", "test": "a.py::t",
                        "probable_cause": "c", "suggested_solution": "s",
                        "affected_files": ["a.py"]})()], source="pytest")
    state.apply_correction("fix", ["a.py"], 1)
    clone = TaskState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert clone == state
    assert clone.problems[0].affected_files == ["a.py"]
    assert clone.corrections[0].problem_ids == ["p001"]
    assert clone.corrections[0].files_changed == ["a.py"]


def test_legacy_dict_without_new_fields_loads():
    old = {"objective": "o",
           "problems": [{"description": "boom", "kind": "test_failure",
                         "iteration": 1, "resolved": False,
                         "origin": "observed"}]}
    state = TaskState.from_dict(old)
    assert state.problems[0].status == PROBLEM_STATUS_PENDING
    assert state.problems[0].problem_id == ""
    assert state.has_blocking_problems() is True


def test_render_known_problems():
    state = TaskState(objective="o")
    assert state.render_known_problems() == ""
    state.add_analyzed_problems(
        [type("P", (), {"error": "expected 200 got 400",
                        "test": "l.py::t", "probable_cause": "validation",
                        "suggested_solution": "review LoginService",
                        "affected_files": ["LoginService.js"]})()],
        source="pytest")
    block = state.render_known_problems()
    assert "KNOWN PROBLEMS" in block
    assert "expected 200 got 400" in block
    assert "validation" in block
    assert "LoginService" in block
    assert "pending" in block


def _fail_ok_tools(fail_text):
    return _ScriptedTools([fail_text, PASS])


def test_pending_problem_blocks_retest(tmp_path):
    trace = _trace(tmp_path)
    test = _run_task("python -m pytest -q")
    fix_cart = _write_task(path="test_cart.py")
    fix_user = _write_task(path="test_user.py")
    tools = _ScriptedTools([FAIL_3, PASS])
    runner = _runner(
        [_task_decision(test), _task_decision(test),
         _task_decision(fix_cart), _task_decision(test),
         _task_decision(fix_user), _task_decision(test), _finish()],
        [_exec_for(test), _exec_for(fix_cart), _exec_for(fix_user),
         _exec_for(test)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    blocked = _events(trace, "test_blocked_pending_problems")
    assert len(blocked) == 2
    assert blocked[0]["iteration"] == 2
    assert len(blocked[0]["problems"]) == 3
    assert blocked[1]["problems"] == ["p003"]
    contexts = runner.planner.received_contexts
    assert any("TEST_BLOCKED_BY_PENDING_PROBLEMS" in ctx
               for ctx in contexts)
    assert all("TOOL EXECUTION ERROR" not in ctx
               or "TEST_BLOCKED" in ctx for ctx in contexts)


def test_problems_recorded_without_retest(tmp_path):
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_3, PASS])
    runner = _runner(
        [_task_decision(fail), _finish()], [_exec_for(fail)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"]
    assert _events(trace, "test_blocked_pending_problems") == []
    assert len(runner.task_state.problems) == 3


def test_blocked_problem_does_not_deadlock(tmp_path):
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_3, PASS], write_errors=1)
    write_cart = _write_task(path="test_cart.py")
    write_user = _write_task(path="test_user.py")
    runner = _runner(
        [_task_decision(fail), _task_decision(write_cart),
         _task_decision(write_cart), _task_decision(write_user),
         _task_decision(fail), _finish()],
        [_exec_for(fail), _exec_for(write_cart), _exec_for(write_cart),
         _exec_for(write_user), _exec_for(fail)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    assert _events(trace, "test_blocked_pending_problems") == []
    state = runner.task_state
    assert all(p.status == PROBLEM_STATUS_RESOLVED
               for p in state.problems)
    assert len(state.corrections) >= 2


def test_normal_command_not_blocked(tmp_path):
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    ls = _run_task("ls")
    tools = _ScriptedTools([FAIL_3, "arquivos\n"])
    runner = _runner(
        [_task_decision(fail), _task_decision(ls), _finish()],
        [_exec_for(fail), _exec_for(ls)], trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q", "ls"]
    assert _events(trace, "test_blocked_pending_problems") == []


def test_investigation_not_blocked(tmp_path):
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    read = Task(tool="read_file",
                arguments={"project_name": "p",
                           "file_path": "src/cart.py"},
                investigation=True)
    tools = _ScriptedTools([FAIL_3, "conteudo\n"])
    runner = _runner(
        [_task_decision(fail),
         Decision(action=DecisionAction.TASK, task=read), _finish()],
        [_exec_for(fail), ExecutionDecision(
            tool="read_file", arguments=read.arguments)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"]
    assert _events(trace, "test_blocked_pending_problems") == []


def test_test_without_problems_runs(tmp_path):
    trace = _trace(tmp_path)
    ok_task = _run_task("python -m pytest -q")
    tools = _ScriptedTools([PASS])
    runner = _runner([_task_decision(ok_task), _finish()],
                     [_exec_for(ok_task)], trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"]
    assert runner.task_state.problems == []


def test_gate_flag_off_preserves_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "error_test_gate", False)
    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    retest = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_3, FAIL_3])
    runner = _runner(
        [_task_decision(fail), _task_decision(retest), _finish()],
        [_exec_for(fail), _exec_for(retest)], trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    assert _events(trace, "test_blocked_pending_problems") == []
    assert len(runner.task_state.problems) == 3


def test_analyzer_flag_off_uses_legacy_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "error_analyzer", False)
    from app.agent.context.error_checklist import ErrorChecklistItem

    class _ItemsChecklist(T.FakeErrorChecklist):
        def __init__(self):
            self._items = [ErrorChecklistItem(
                id=1, description="falha extraída 1")]
            self._checks = 0

        def render(self):
            return "CURRENT ERROR CHECKLIST\n- 1. falha extraída 1"

        def generate(self, command, output, iteration=None):
            pass

        @property
        def pending_count(self):
            self._checks += 1
            return 1 if self._checks <= 2 else 0

    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    tools = _ScriptedTools([FAIL_3])
    runner = _runner([_task_decision(fail), _finish(), _finish()],
                     [_exec_for(fail)], trace, tools=tools,
                     error_checklist=_ItemsChecklist())
    runner.run(objective="obj", project_name="p")
    assert [p.description for p in runner.task_state.problems] == [
        "falha extraída 1"]
    assert _events(trace, "error_analysis_completed") == []


def _three_fail_tools():
    fail_a = ("STATUS: failure (exit code 1)\nFAILED a.py::test_a\n"
              "a.py:10 AssertionError\n")
    fail_b = ("STATUS: failure (exit code 1)\nFAILED b.py::test_b\n"
              "b.py:20 ValueError\n")
    fail_c = ("STATUS: failure (exit code 1)\nFAILED c.py::test_c\n"
              "c.py:30 KeyError\n")
    return _ScriptedTools([fail_a + fail_b + fail_c, PASS])


def test_three_errors_three_corrections_then_test(tmp_path):
    trace = _trace(tmp_path)
    test = _run_task("python -m pytest -q")
    tools = _three_fail_tools()
    wa = _write_task(path="a.py")
    wb = _write_task(path="b.py")
    wc = _write_task(path="c.py")
    runner = _runner(
        [_task_decision(test), _task_decision(wa), _task_decision(wb),
         _task_decision(wc), _task_decision(test), _finish()],
        [_exec_for(test), _exec_for(wa), _exec_for(wb), _exec_for(wc),
         _exec_for(test)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    state = runner.task_state
    assert len(state.problems) == 3
    assert state.open_problems == []
    assert len(state.corrections) == 4
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    assert _events(trace, "test_blocked_pending_problems") == []
    assert len(_events(trace, "correction_applied")) == 3


def test_partial_fix_keeps_gate_closed(tmp_path):
    trace = _trace(tmp_path)
    test = _run_task("python -m pytest -q")
    tools = _three_fail_tools()
    wa = _write_task(path="a.py")
    runner = _runner(
        [_task_decision(test), _task_decision(wa), _task_decision(test),
         _finish()],
        [_exec_for(test), _exec_for(wa), _exec_for(test)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"]
    blocked = _events(trace, "test_blocked_pending_problems")
    assert len(blocked) == 1
    assert len(blocked[0]["problems"]) == 2


def test_one_correction_covers_related_problems(tmp_path):
    trace = _trace(tmp_path)
    fail_both = ("STATUS: failure (exit code 1)\n"
                 "FAILED u.py::test_a\nFAILED u.py::test_b\n"
                 "u.py:5 ValueError\n")
    tools = _ScriptedTools([fail_both, PASS])
    test = _run_task("python -m pytest -q")
    fix = _write_task(path="u.py")
    runner = _runner(
        [_task_decision(test), _task_decision(fix), _task_decision(test),
         _finish()],
        [_exec_for(test), _exec_for(fix), _exec_for(test)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    assert _events(trace, "test_blocked_pending_problems") == []
    assert len(runner.task_state.corrections) >= 1
    assert all(c.problem_ids == ["p001", "p002"]
               for c in runner.task_state.corrections
               if c.files_changed == ["u.py"])


def test_new_error_creates_new_problem(tmp_path):
    trace = _trace(tmp_path)
    fail_old = ("STATUS: failure (exit code 1)\nFAILED a.py::test_a\n"
                "a.py:1 AssertionError\n")
    fail_new = ("STATUS: failure (exit code 1)\nFAILED b.py::test_b\n"
                "b.py:2 ValueError\n")
    tools = _ScriptedTools([fail_old, fail_new])
    test = _run_task("python -m pytest -q")
    fix = _write_task(path="a.py")
    runner = _runner(
        [_task_decision(test), _task_decision(fix), _task_decision(test),
         _task_decision(test), _finish()],
        [_exec_for(test), _exec_for(fix), _exec_for(test),
         _exec_for(test)],
        trace, tools=tools)
    assert runner.run(objective="obj", project_name="p") == "done"
    state = runner.task_state
    by_test = {p.test: p.status for p in state.problems}
    assert by_test.get("a.py::test_a") == "pending_verification"
    assert by_test.get("b.py::test_b") == "pending"
    assert tools.ran_commands == ["python -m pytest -q"] * 2
    assert len(_events(trace, "test_blocked_pending_problems")) == 1


def test_same_test_new_error_invalidates_old(tmp_path):
    trace = _trace(tmp_path)
    fail_v1 = ("STATUS: failure (exit code 1)\nFAILED a.py::test_a boom1\n")
    fail_v2 = ("STATUS: failure (exit code 1)\nFAILED a.py::test_a boom2\n")
    tools = _ScriptedTools([fail_v1, fail_v2])
    test = _run_task("python -m pytest -q")
    fix = _write_task(path="a.py")
    runner = _runner(
        [_task_decision(test), _task_decision(fix), _task_decision(test),
         _finish()],
        [_exec_for(test), _exec_for(fix), _exec_for(test)],
        trace, tools=tools)
    runner.run(objective="obj", project_name="p")
    invalidated = [p for p in runner.task_state.problems
                   if p.status == "invalidated"]
    assert len(invalidated) == 1
    assert any(e["event"] == "problem_invalidated"
               for e in trace.read_events())


def test_executor_receives_known_problems(tmp_path):
    seen = []

    class _CatchingMaker(T.FakeTaskDecisionMaker):
        def decide(self, objective, task, context, iteration=None):
            seen.append(context)
            return super().decide(objective, task, context,
                                  iteration=iteration)

    trace = _trace(tmp_path)
    fail = _run_task("python -m pytest -q")
    fix = _write_task(path="src/cart.py")
    tools = _ScriptedTools([FAIL_3])
    runner = Runner(
        planner=T.FakePlanner(
            [_task_decision(fail), _task_decision(fix), _finish()]),
        task_decision_maker=_CatchingMaker(
            [_exec_for(fail), _exec_for(fix)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=__import__(
            "app.agent.context.operational_memory",
            fromlist=["OperationalMemory"]).OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    runner.run(objective="obj", project_name="p")
    assert len(seen) == 2
    assert "KNOWN PROBLEMS" in seen[1]
    assert "test_cart.py::test_total" in seen[1]
    assert "Probable cause" in seen[1]
    assert "verify" in seen[1]


def test_short_repair_untouched_by_analyzer(tmp_path):
    from app.agent.execution.validator import TaskValidator
    from app.agent.planning.decision_parser import DecisionParser
    from app.agent.planning.planner import Planner
    from app.llm.client import LLMClient
    from app.tools.registry import ToolRegistry

    class _Provider:
        name = "s"

        def __init__(self, contents):
            self._contents = list(contents)
            self.prompts = []

        def generate(self, messages, tools=None):
            self.prompts.append(messages[0].content)
            item = self._contents.pop(0)
            return LLMResponse(content=item, tool_calls=[],
                               usage=Usage(10, 5, 15), provider="s")

    def _json_task(tool, arguments):
        return json.dumps({"action": "task",
                           "task": {"tool": tool, "arguments": arguments,
                                    "dependencies": []}})

    tools = ToolRegistry()
    tools.load_defaults()
    args = {"project_name": "p", "file_path": "a.py", "content": "x"}
    provider = _Provider([
        "lixo {{{",
        _json_task("write_file", args),
        json.dumps({"action": "finish", "content": "done"}),
    ])
    planner = Planner(llm=LLMClient(provider),
                      parser=DecisionParser(tools=tools), tools=tools)
    trace = _trace(tmp_path)
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker(
            [ExecutionDecision(tool="write_file", arguments=args)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=TaskValidator(tools),
        operational_memory=__import__(
            "app.agent.context.operational_memory",
            fromlist=["OperationalMemory"]).OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == ["short_repair"]
    assert _events(trace, "error_analysis_started") == []


def test_structured_roundtrip_persists(projects_root, tmp_path,
                                       monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", True)
    from app.agent.taskstate.persistence import (
        load_task_state,
        save_task_state,
    )
    state = TaskState(task_id="t9", objective="o")
    state.add_analyzed_problems(
        [type("P", (), {"error": "FAILED a", "test": "a.py::t",
                        "probable_cause": "c", "suggested_solution": "s",
                        "affected_files": ["a.py"]})()], source="pytest")
    state.apply_correction("fix", ["a.py"], 1)
    assert save_task_state(state, "p")["ok"] is True
    loaded = load_task_state("p")
    assert loaded == state
    assert loaded.problems[0].status == "pending_verification"
    assert loaded.corrections[0].files_changed == ["a.py"]
