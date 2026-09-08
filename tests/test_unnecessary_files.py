"""Tests for unnecessary-file detection, classification and cleanup."""

import pytest

from app.agent.context.unnecessary_files import (
    KEEP,
    SAFE,
    UNCERTAIN,
    analyze_unnecessary_files,
    format_evidence_for_llm,
)
from app.tools.filesystem.write_file import write_file


def _report(projects_root, files: dict[str, str]):
    from app.tools.registry import ToolRegistry

    for path, content in files.items():
        write_file("proj", path, content)
    registry = ToolRegistry()
    registry.load_defaults()
    return analyze_unnecessary_files("proj", registry.execute)


def _by_path(report):
    return {c.path: c for c in report.candidates}


def test_no_candidates_for_referenced_project(projects_root):
    report = _report(projects_root, {
        "main.py": "import svc\nprint(svc.run())\n",
        "svc.py": "def run():\n    return 1\n",
    })

    assert report.files_scanned == 2
    assert report.candidates == []
    assert report.safe_paths == []


def test_referenced_file_is_not_candidate(projects_root):
    report = _report(projects_root, {
        "a.py": "from b import helper\nhelper()\n",
        "b.py": "def helper():\n    return 1\n",
    })

    assert "b.py" not in _by_path(report)


def test_temp_file_is_safe(projects_root):
    report = _report(projects_root, {
        "main.py": "print('oi')\n",
        "notes.tmp": "rascunho\n",
    })

    candidates = _by_path(report)
    assert candidates["notes.tmp"].verdict == SAFE
    assert any("temporary" in e for e in candidates["notes.tmp"].evidences)


def test_superseded_file_is_safe(projects_root):
    report = _report(projects_root, {
        "service.py": "def run():\n    return 1\n",
        "old_service.py": "def run():\n    return 0\n",
    })

    candidates = _by_path(report)
    assert candidates["old_service.py"].verdict == SAFE
    # service.py não tem referências, mas também não tem sinal positivo
    # de inutilidade => UNCERTAIN (conservador), nunca SAFE.
    assert candidates["service.py"].verdict == UNCERTAIN
    assert report.safe_paths == ["old_service.py"]


def test_ambiguous_file_is_uncertain_and_kept(projects_root):
    report = _report(projects_root, {
        "main.py": "print('oi')\n",
        "config.py": "TIMEOUT = 30\n",
    })

    candidates = _by_path(report)
    assert candidates["config.py"].verdict == UNCERTAIN
    assert "config.py" not in report.safe_paths


def test_entrypoint_is_not_candidate(projects_root):
    report = _report(projects_root, {
        "main.py": "print('sozinho')\n",
    })

    assert report.candidates == []


def test_test_file_is_not_candidate(projects_root):
    report = _report(projects_root, {
        "test_svc.py": "def test_x():\n    assert True\n",
    })

    assert report.candidates == []


def test_special_file_is_not_candidate(projects_root):
    report = _report(projects_root, {
        "package.json": '{"name": "x"}\n',
        "README.md": "# x\n",
    })

    assert report.candidates == []


def test_config_reference_keeps_file(projects_root):
    report = _report(projects_root, {
        "package.json": '{"scripts": {"start": "node server.js"}}\n',
        "server.js": "console.log(1)\n",
    })

    assert "server.js" not in _by_path(report)


def test_duplicate_of_kept_file_is_safe(projects_root):
    report = _report(projects_root, {
        "main.py": "import svc\nprint(svc.run())\n",
        "svc.py": "def run():\n    return 1\n",
        "svc_copy_tmp.py": "def run():\n    return 1\n",
    })

    candidates = _by_path(report)
    # svc_copy_tmp.py is an exact duplicate of the needed svc.py
    assert candidates["svc_copy_tmp.py"].verdict == SAFE


def test_duplicate_without_kept_copy_is_uncertain(projects_root):
    report = _report(projects_root, {
        "main.py": "print('oi')\n",
        "dup_a.py": "X = 1\n",
        "dup_b.py": "X = 1\n",
    })

    candidates = _by_path(report)
    assert candidates["dup_a.py"].verdict == UNCERTAIN
    assert candidates["dup_b.py"].verdict == UNCERTAIN
    assert report.safe_paths == []


def test_multiple_candidates_mixed_verdicts(projects_root):
    report = _report(projects_root, {
        "main.py": "print('oi')\n",
        "debug_out.log": "x\n",
        "stray.py": "Y = 2\n",
    })

    candidates = _by_path(report)
    assert candidates["debug_out.log"].verdict == SAFE
    assert candidates["stray.py"].verdict == UNCERTAIN
    assert report.safe_paths == ["debug_out.log"]


def test_empty_project_has_no_candidates(projects_root):
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_defaults()
    (projects_root / "empty").mkdir()
    report = analyze_unnecessary_files("empty", registry.execute)

    assert report.files_scanned == 0
    assert report.candidates == []


def test_missing_project_returns_empty_report():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_defaults()
    report = analyze_unnecessary_files("nao_existe", registry.execute)

    assert report.files_scanned == 0
    assert report.candidates == []


def test_evidence_format_for_llm(projects_root):
    report = _report(projects_root, {
        "main.py": "print('oi')\n",
        "old_x.py": "pass\n",
        "mystery.py": "Z = 3\n",
    })

    text = format_evidence_for_llm(report)

    assert "old_x.py" in text
    assert "SAFE" in text
    assert "no references" in text
    assert "mystery.py" in text
    assert "UNCERTAIN" in text


def test_verdict_constants():
    assert (SAFE, UNCERTAIN, KEEP) == ("safe", "uncertain", "keep")


# --- Registry / parallel-batch exclusion -------------------------------


def test_new_tools_are_mutative_not_parallel():
    from app.agent.parallel import PURE_READ_TOOLS, all_pure_read
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_defaults()

    for name in ("edit_file", "delete_file"):
        assert registry.get(name).pure is False
        assert name not in PURE_READ_TOOLS

    assert all_pure_read(["read_file", "edit_file"]) is False
    assert all_pure_read(["delete_file"]) is False
    assert all_pure_read(["read_file", "delete_file"]) is False
    # read-only batches still work
    assert all_pure_read(["read_file", "list_files"]) is True


def test_final_verification_exposes_detector(projects_root):
    from unittest.mock import MagicMock

    from app.agent.context.final_verification import FinalVerification

    verification = FinalVerification(MagicMock())
    write_file("proj", "main.py", "print('oi')\n")
    write_file("proj", "junk.tmp", "x\n")

    from app.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.load_defaults()
    report = verification.detect_unnecessary_files("proj", registry.execute)

    assert report.safe_paths == ["junk.tmp"]


# --- Cleanup integration (Runner finish flow) ---------------------------


def _cleanup_runner(tools, decisions, final_verification, trace):
    import tests.test_runner as T
    from app.agent.runner import Runner

    return Runner(
        planner=T.FakePlanner(decisions),
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
        final_verification=final_verification,
        execution_trace=trace,
    )


class _DetectingVerification:
    """FinalVerification fake that also runs the real detector."""

    def __init__(self, result):
        from app.agent.context.final_verification import (
            FinalVerificationResult,
        )
        self._result = result

    def verify(self, objective, project_name, summary, tools_execute):
        return self._result

    def detect_unnecessary_files(self, project_name, tools_execute):
        return analyze_unnecessary_files(project_name, tools_execute)


def test_cleanup_removes_only_safe_on_finish(projects_root, tmp_path):
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.trace import ExecutionTrace
    from app.tools.registry import ToolRegistry

    write_file("proj", "main.py", "print('ok')\n")
    write_file("proj", "tmp_helper.py", "X = 1\n")
    write_file("proj", "mystery.py", "Y = 2\n")

    tools = ToolRegistry()
    tools.load_defaults()
    trace = ExecutionTrace(trace_dir=tmp_path)
    runner = _cleanup_runner(
        tools,
        [Decision(action=DecisionAction.FINISH, content="done")],
        _DetectingVerification(
            FinalVerificationResult(FinalVerificationResult.OK)),
        trace,
    )

    assert runner.run(objective="obj", project_name="proj") == "done"

    # SAFE removido; UNCERTAIN e entrypoint preservados.
    assert not (projects_root / "proj" / "tmp_helper.py").exists()
    assert (projects_root / "proj" / "mystery.py").exists()
    assert (projects_root / "proj" / "main.py").exists()

    events = trace.read_events()
    scans = [e for e in events if e["event"] == "unnecessary_files_scan"]
    assert len(scans) == 1
    assert "tmp_helper.py" in scans[0]["safe_paths"]
    removed = [e for e in events
               if e["event"] == "unnecessary_file_removed"]
    assert [e["file_path"] for e in removed] == ["tmp_helper.py"]


def test_cleanup_blocks_finish_when_recheck_fails(
        projects_root, tmp_path):
    import tests.test_runner as T
    from app.agent.context.final_verification import FinalVerificationResult
    from app.agent.execution.execution_decision import ExecutionDecision
    from app.agent.execution.task import Task
    from app.agent.planning.decision import Decision, DecisionAction
    from app.agent.trace import ExecutionTrace
    from app.tools.registry import ToolRegistry

    write_file("proj", "main.py", "print('ok')\n")
    write_file("proj", "stale.tmp", "X = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    trace = ExecutionTrace(trace_dir=tmp_path)

    fix_task = Task(
        tool="write_file",
        arguments={"project_name": "proj", "file_path": "main.py",
                   "content": "print('ok')\n"},
    )
    from app.agent.runner import Runner
    runner = Runner(
        planner=T.FakePlanner([
            Decision(action=DecisionAction.FINISH, content="retry"),
            Decision(action=DecisionAction.TASK, task=fix_task),
            Decision(action=DecisionAction.FINISH, content="done"),
        ]),
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="write_file",
                              arguments=dict(fix_task.arguments)),
        ]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=_DetectingVerification(
            FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )

    calls = {"n": 0}
    real_check = runner._run_finish_check

    def _flaky_check(project_name):
        calls["n"] += 1
        if calls["n"] == 2:
            # revalidação após a remoção falha uma vez
            return "check_project: FAILED\nSTDOUT:\n(empty)"
        return real_check(project_name)

    runner._run_finish_check = _flaky_check

    assert runner.run(objective="obj", project_name="proj") == "done"

    events = trace.read_events()
    blocks = [e for e in events if e["event"] == "finish_block"]
    assert any(e.get("gate") == "unnecessary_files_recheck"
               for e in blocks)
    # Sem loop infinito: segundo finish conclui.
    assert runner._stats.finish_blocks == 1
