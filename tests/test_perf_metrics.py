"""Fase 3 — métricas de performance e otimizações.

Cobre: timing/sucesso por tool (incl. timeout/exit code de run_command),
contadores do Runner, formato do resumo, cache do finish-check (OPT-1)
e poda de diretórios de dependência no list_files (OPT-2).
"""

import pytest

from app.agent.perf import AgentStats, format_performance_summary
from app.config import Config
from app.llm.usage import UsageTracker
from app.tools.filesystem.write_file import write_file
from app.tools.registry import ToolRegistry


@pytest.fixture
def tracked_registry():
    tracker = UsageTracker()
    registry = ToolRegistry()
    registry.load_defaults()
    registry.set_tracker(tracker)
    return registry, tracker


def test_tool_records_duration_and_success(projects_root, tracked_registry):
    registry, tracker = tracked_registry
    write_file("proj", "a.py", "1")

    registry.execute("read_file",
                     {"project_name": "proj", "file_path": "a.py"})

    records = tracker.get_tool_records()
    assert len(records) == 1
    assert records[0].tool == "read_file"
    assert records[0].success is True
    assert records[0].duration_ms >= 0.0

    stats = tracker.tool_stats()
    assert stats["calls"] == 1
    assert stats["by_tool"]["read_file"]["calls"] == 1
    assert stats["by_tool"]["read_file"]["failures"] == 0


def test_tool_records_failure_without_swallowing(projects_root,
                                                 tracked_registry):
    registry, tracker = tracked_registry

    with pytest.raises(FileNotFoundError):
        registry.execute("read_file",
                         {"project_name": "proj", "file_path": "nope.py"})

    records = tracker.get_tool_records()
    assert len(records) == 1
    assert records[0].success is False
    assert records[0].error == "FileNotFoundError"
    assert tracker.tool_stats()["failures"] == 1


def test_unknown_tool_records_and_raises(tracked_registry):
    registry, tracker = tracked_registry

    with pytest.raises(ValueError):
        registry.execute("nope", {})

    assert tracker.get_tool_records()[0].success is False


def test_registry_without_tracker_unchanged(projects_root):
    registry = ToolRegistry()
    registry.load_defaults()
    write_file("proj", "a.py", "1")

    assert "1" in registry.execute(
        "read_file", {"project_name": "proj", "file_path": "a.py"})
    assert registry.tool_stats()["calls"] == 0


def test_run_command_records_timeout_and_exit_code(projects_root,
                                                   tracked_registry):
    registry, tracker = tracked_registry
    write_file("proj", "loop.py", "while True:\n    pass\n")
    write_file("proj", "ok.py", "print('hi')")

    assert "TIMEOUT" in registry.execute(
        "run_command",
        {"project_name": "proj", "command": "python3 loop.py",
         "timeout_seconds": 1},
    )
    registry.execute(
        "run_command",
        {"project_name": "proj", "command": "python3 ok.py"},
    )

    by_tool = tracker.tool_stats()["by_tool"]["run_command"]
    assert by_tool["calls"] == 2
    assert by_tool["timeouts"] == 1

    records = [r for r in tracker.get_tool_records()
               if r.tool == "run_command"]
    assert records[0].timeout is True
    assert records[0].exit_code is None
    assert records[0].command == "python3 loop.py"
    assert records[1].timeout is False
    assert records[1].exit_code == 0


def test_run_command_long_command_truncated(projects_root, tracked_registry):
    registry, tracker = tracked_registry
    write_file("proj", "placeholder.txt", "x")
    big = "echo " + "y" * 5000
    registry.execute("run_command", {"project_name": "proj",
                                     "command": big})
    command = tracker.get_tool_records()[0].command
    assert command is not None
    assert len(command) < len(big)
    assert big[:200] in command


def test_write_file_content_never_recorded(projects_root, tracked_registry):
    registry, tracker = tracked_registry
    secret = "SECRETO-" * 1000
    registry.execute("write_file", {"project_name": "proj",
                                    "file_path": "a.py", "content": secret})
    assert tracker.get_tool_records()[0].command is None
    assert secret not in repr(tracker.get_tool_records())


def test_agent_stats_in_scripted_run(projects_root):
    import tests.test_runner as T
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.execution.execution_decision import ExecutionDecision
    from app.agent.execution.task import Task
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.runner import Runner
    from app.agent.context.operational_memory import OperationalMemory

    tracker = UsageTracker()
    tools = ToolRegistry()
    tools.load_defaults()
    tools.set_tracker(tracker)

    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "a.py",
                           "content": "x"})
    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             Decision(action=DecisionAction.FINISH, content="done")]),
        task_decision_maker=T.FakeTaskDecisionMaker(
            [ExecutionDecision(tool="write_file",
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
    )
    events = []
    runner.on_event = lambda e: events.append((e.type, e.data))

    assert runner.run(objective="obj", project_name="p") == "done"
    stats = runner._stats
    assert stats.iterations == 2
    assert stats.planner_calls == 2
    assert stats.test_runs == 0

    done = [data for kind, data in events if kind == "agent_done"][0]
    assert "SUCCESS" in done["perf_summary"]
    assert "Iterations: 2" in done["perf_summary"]


def test_test_runs_counted(projects_root):
    import tests.test_runner as T
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.execution.execution_decision import ExecutionDecision
    from app.agent.execution.task import Task
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.runner import Runner
    from app.agent.context.operational_memory import OperationalMemory

    tools = ToolRegistry()
    tools.load_defaults()
    write_file("p", "main.py", "print('oi')")
    task = Task(tool="run_command",
                arguments={"project_name": "p",
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
    )
    runner.run(objective="obj", project_name="p")
    assert runner._stats.test_runs == 1
    assert (runner._stats.test_passed + runner._stats.test_failed) == 1


def test_finish_check_cached_across_blocked_finishes(projects_root,
                                                     monkeypatch):
    """OPT-1: finish repetido sem escrita reaproveita o check."""
    import tests.test_runner as T
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.runner import Runner
    from app.tools.analysis import check_project as check_module

    monkeypatch.setattr(Config, "sandbox_mode", "none")
    write_file("p", "main.py", "print('oi')")

    tools = ToolRegistry()
    tools.load_defaults()
    calls = []
    real_execute = tools.execute

    def counting_execute(name, arguments):
        if name == "check_project":
            calls.append(arguments)
        return real_execute(name, arguments)

    tools.execute = counting_execute

    class OnceProblems(T.FakeFinalVerification):
        def __init__(self):
            super().__init__(
                result=FinalVerificationResult(FinalVerificationResult.OK))
            self.n = 0

        def verify(self, objective, project_name, summary, tools_execute):
            self.n += 1
            if self.n == 1:
                return FinalVerificationResult(
                    FinalVerificationResult.PROBLEMS_FOUND, "ruim")
            return FinalVerificationResult(FinalVerificationResult.OK)

    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.FINISH, content="a"),
             Decision(action=DecisionAction.FINISH, content="b")]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=OnceProblems(),
    )
    assert runner.run(objective="obj", project_name="p") == "b"
    assert len(calls) == 1  # segundo finish reaproveitou o check
    assert runner._stats.finish_blocks == 1


def test_finish_check_reruns_after_write(projects_root, monkeypatch):
    """OPT-1: escrita entre finishs invalida o cache."""
    import tests.test_runner as T
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.execution.execution_decision import ExecutionDecision
    from app.agent.execution.task import Task
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.runner import Runner

    monkeypatch.setattr(Config, "sandbox_mode", "none")
    write_file("p", "bad.py", "def broken(:\n")
    tools = ToolRegistry()
    tools.load_defaults()
    calls = []
    real_execute = tools.execute

    def counting_execute(name, arguments):
        if name == "check_project":
            calls.append(arguments)
        return real_execute(name, arguments)

    tools.execute = counting_execute

    task = Task(tool="write_file",
                arguments={"project_name": "p", "file_path": "bad.py",
                           "content": "x = 1\n"})
    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.FINISH, content="a"),
             Decision(action=DecisionAction.TASK, task=task),
             Decision(action=DecisionAction.FINISH, content="b")]),
        task_decision_maker=T.FakeTaskDecisionMaker(
            [ExecutionDecision(tool="write_file",
                               arguments=task.arguments)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
    )
    assert runner.run(objective="obj", project_name="p") == "b"
    assert len(calls) == 2  # escrita invalidou: check rodou de novo
    assert runner._stats.finish_blocks == 1


def test_list_files_ignores_dependency_dirs(projects_root):
    from app.tools.filesystem.list_files import list_files

    write_file("proj", "main.py", "1")
    write_file("proj", "node_modules/pkg/index.js", "x")
    write_file("proj", ".venv/lib/a.py", "x")
    write_file("proj", "__pycache__/a.pyc", "x")
    write_file("proj", ".git/objects/x", "x")
    write_file("proj", "src/app.py", "1")

    files = sorted(list_files("proj"))
    assert files == ["main.py", "src/app.py"]


def test_perf_summary_format_with_empty_inputs():
    summary = format_performance_summary("SUCCESS")
    assert "Wall time:" in summary
    assert "Iterations: 0" in summary

    stats = AgentStats(iterations=3, planner_calls=3, test_runs=2,
                       test_passed=2)
    summary = format_performance_summary("SUCCESS", None,
                                         {"calls": 5, "total_ms": 120.0,
                                          "by_tool": {
                                              "write_file": {"calls": 2,
                                                             "total_ms": 4.0,
                                                             "avg_ms": 2.0}}},
                                         stats)
    assert "Iterations: 3" in summary
    assert "write_file: 2" in summary
    assert "Runs: 2" in summary
