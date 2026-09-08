"""Ciclo de identificação e correção de erros.

Cobre o falso positivo do detector de repetição (re-tentativa após
erro/investigação deve passar; repetição sem progresso bloqueia),
a interpretação de resultados de run_command, a preservação do
ErrorChecklist em falhas de infra e a validação fim a fim com erro real.
"""

import pytest

from app.agent.context.error_checklist import ErrorChecklist
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.context.operational_memory import OperationalMemory
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.dependency import Dependency
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.config import Config
from app.llm.models import LLMResponse, Usage
from app.tools.filesystem.write_file import write_file
from app.tools.registry import ToolRegistry

import tests.test_runner as T


FAIL_PYTEST = "python3 -m pytest /nonexistent_path_xyz"
PASS_PYTEST = "python3 -m pytest --version"


class StubLLM:
    """Fake de LLMClient com roteiro por componente e captura de chamadas."""

    def __init__(self, checklist_items=None):
        self.checklist_items = checklist_items or []
        self.calls = []

    def generate(self, messages, tools=None, component=None,
                 iteration=None, attempt=1):
        self.calls.append({
            "component": component, "iteration": iteration,
            "prompt": messages[0].content,
        })
        if component == "ErrorChecklist":
            import json
            content = json.dumps({"items": self.checklist_items})
        else:
            content = "resumo"
        return LLMResponse(content=content, tool_calls=[],
                           usage=Usage(1, 1, 2), provider="stub")


def _write_task(path="app.py", content="x = 1\n"):
    return Task(tool="write_file",
                arguments={"project_name": "p", "file_path": path,
                           "content": content})


def _run_task(command):
    return Task(tool="run_command",
                arguments={"project_name": "p", "command": command})


def _finish(content="done"):
    return Decision(action=DecisionAction.FINISH, content=content)


def _task_decision(task):
    return Decision(action=DecisionAction.TASK, task=task)


def _real_runner(tools, decisions, executions, **overrides):
    fv = overrides.pop("final_verification", None) or T.FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK))
    return Runner(
        planner=T.FakePlanner(decisions),
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=overrides.pop(
            "operational_memory", OperationalMemory(tools)),
        checklist=T.FakeChecklist(),
        error_checklist=overrides.pop("error_checklist",
                                      T.FakeErrorChecklist()),
        planner_error_memory=overrides.pop("planner_error_memory", None)
        or __import__("app.agent.context.planner_error_memory",
                      fromlist=["PlannerErrorMemory"]).PlannerErrorMemory(),
        final_verification=fv,
        **overrides,
    )


def _run_events(runner, **kwargs):
    events = []
    runner.on_event = lambda e: events.append((e.type, e.data))
    result = runner.run(**kwargs)
    return result, events


def _executor_errors(events):
    return [d["error"] for kind, d in events if kind == "executor_error"]


def _repetition_blocks(events):
    return [e for e in _executor_errors(events)
            if "repeat an operation" in e]


@pytest.fixture
def real_tools(projects_root, monkeypatch):
    monkeypatch.setattr(Config, "sandbox_mode", "none")
    tools = ToolRegistry()
    tools.load_defaults()
    return tools


# 1. Repetição idêntica sem progresso → bloqueada.
def test_identical_repeat_without_progress_is_blocked(real_tools):
    task = _write_task()
    runner = _real_runner(
        real_tools,
        [_task_decision(task), _task_decision(task), _finish()],
        [ExecutionDecision(tool="write_file", arguments=task.arguments),
         ExecutionDecision(tool="write_file", arguments=task.arguments)],
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert len(_repetition_blocks(events)) == 1


# 5. Repetição real de run_command mutante também é detectada.
def test_identical_mutating_run_command_is_blocked(real_tools):
    write_file("p", "dummy.txt", "x")
    cmd = "cat > out.txt <<'EOF'\nhi\nEOF"
    task = _run_task(cmd)
    runner = _real_runner(
        real_tools,
        [_task_decision(task), _task_decision(task), _finish()],
        [ExecutionDecision(tool="run_command", arguments=task.arguments),
         ExecutionDecision(tool="run_command", arguments=task.arguments)],
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert len(_repetition_blocks(events)) == 1


# 2. read (investigação) → write → permitida.
def test_read_then_write_is_allowed(real_tools):
    write_file("p", "app.py", "x = 1\n")
    write = _write_task(content="x = 1\n")
    investigated = _write_task(content="x = 1\n")
    investigated.dependencies = [Dependency(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "app.py"})]
    runner = _real_runner(
        real_tools,
        [_task_decision(write), _task_decision(investigated), _finish()],
        [ExecutionDecision(tool="write_file", arguments=write.arguments),
         ExecutionDecision(tool="write_file",
                           arguments=investigated.arguments)],
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert _repetition_blocks(events) == []


# 3. write → teste com erro → mesmo write → permitida.
def test_write_after_failing_test_is_allowed(real_tools):
    write = _write_task()
    fail = _run_task(FAIL_PYTEST)
    runner = _real_runner(
        real_tools,
        [_task_decision(write), _task_decision(fail),
         _task_decision(write), _finish()],
        [ExecutionDecision(tool="write_file", arguments=write.arguments),
         ExecutionDecision(tool="run_command", arguments=fail.arguments),
         ExecutionDecision(tool="write_file", arguments=write.arguments)],
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert _repetition_blocks(events) == []


# 4. Fluxo observado no benchmark: write → erro → read → write → teste.
def test_observed_trace_flow_is_allowed(real_tools):
    write = _write_task()
    fail = _run_task(FAIL_PYTEST)
    read = Task(tool="read_file",
                arguments={"project_name": "p", "file_path": "app.py"},
                investigation=True)
    runner = _real_runner(
        real_tools,
        [_task_decision(write), _task_decision(fail), _task_decision(read),
         _task_decision(write), _finish()],
        [ExecutionDecision(tool="write_file", arguments=write.arguments),
         ExecutionDecision(tool="run_command", arguments=fail.arguments),
         ExecutionDecision(tool="read_file", arguments=read.arguments),
         ExecutionDecision(tool="write_file", arguments=write.arguments)],
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert _repetition_blocks(events) == []


# 6/7. Erro identificado e disponível para o Planner.
def test_failing_test_error_reaches_planner_context(real_tools):
    # Fase 5: retest sem correção é bloqueado pelo gate — o fluxo
    # correto insere um write (fix) entre a falha e o retest.
    write_file("p", "dummy.txt", "x")
    llm = StubLLM(["pytest falhou em test_x: AssertionError"])
    checklist = ErrorChecklist(llm)
    fail = _run_task(FAIL_PYTEST)
    fix = _write_task(path="app.py", content="x = 1\n")
    ok = _run_task(PASS_PYTEST)
    runner = _real_runner(
        real_tools,
        [_task_decision(fail), _finish(), _task_decision(fix),
         _task_decision(ok), _finish()],
        [ExecutionDecision(tool="run_command", arguments=fail.arguments),
         ExecutionDecision(tool="write_file", arguments=fix.arguments),
         ExecutionDecision(tool="run_command", arguments=ok.arguments)],
        error_checklist=checklist,
    )
    result, _ = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    contexts = runner.planner.received_contexts
    # Fase 6: o Planner recebe a evidência REAL (não mais o texto
    # canned do StubLLM, que era resposta fake do checklist legado).
    assert any("nonexistent_path_xyz" in ctx for ctx in contexts)
    assert checklist.pending_count == 0


# 8. Erro resolvido limpa o checklist e libera o finish.
def test_resolved_error_unblocks_finish(real_tools):
    # Fase 5: retest exige correção antes (gate); fluxo corrigido.
    write_file("p", "dummy.txt", "x")
    llm = StubLLM(["algum erro"])
    checklist = ErrorChecklist(llm)
    fail = _run_task(FAIL_PYTEST)
    fix = _write_task(path="app.py", content="x = 1\n")
    ok = _run_task(PASS_PYTEST)
    runner = _real_runner(
        real_tools,
        [_task_decision(fail), _task_decision(fix), _task_decision(ok),
         _finish()],
        [ExecutionDecision(tool="run_command", arguments=fail.arguments),
         ExecutionDecision(tool="write_file", arguments=fix.arguments),
         ExecutionDecision(tool="run_command", arguments=ok.arguments)],
        error_checklist=checklist,
    )
    result, _ = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert checklist.pending_count == 0


# 9. Correção seguida de novo teste (teste roda 2x: falha e passa).
def test_fix_is_followed_by_retest(real_tools):
    llm = StubLLM(["falha inicial"])
    checklist = ErrorChecklist(llm)
    bad = _write_task(content="x = \n")
    fail = _run_task(FAIL_PYTEST)
    good = _write_task(content="x = 1\n")
    ok = _run_task(PASS_PYTEST)
    runner = _real_runner(
        real_tools,
        [_task_decision(bad), _task_decision(fail), _task_decision(good),
         _task_decision(ok), _finish()],
        [ExecutionDecision(tool="write_file", arguments=bad.arguments),
         ExecutionDecision(tool="run_command", arguments=fail.arguments),
         ExecutionDecision(tool="write_file", arguments=good.arguments),
         ExecutionDecision(tool="run_command", arguments=ok.arguments)],
        error_checklist=checklist,
    )
    result, events = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    tool_starts = [d for kind, d in events
                   if kind == "tool_start" and d["name"] == "run_command"]
    assert len(tool_starts) == 2


# 10. Erro persistente não gera loop infinito (limite de iterações).
def test_persistent_error_is_bounded(real_tools):
    fail = _run_task(FAIL_PYTEST)
    runner = _real_runner(
        real_tools,
        [_task_decision(fail)] * 8,
        [ExecutionDecision(tool="run_command", arguments=fail.arguments)] * 8,
        max_iterations=6,
    )
    with pytest.raises(RuntimeError, match="limite de 6"):
        runner.run(objective="o", project_name="p")


# Correções idênticas repetidas disparam estagnação, não loop infinito.
def test_repeated_identical_fixes_trigger_stagnation(real_tools):
    write_file("p", "a.py", "x = 1\n")
    write = _write_task(path="a.py", content="x = 1\n")
    fail = _run_task(FAIL_PYTEST)

    decisions, executions = [_task_decision(write), _task_decision(fail)], [
        ExecutionDecision(tool="write_file", arguments=write.arguments),
        ExecutionDecision(tool="run_command", arguments=fail.arguments)]
    files = ["b.py", "c.py", "d.py", "e.py"]
    for i in range(12):
        path = files[i % len(files)]
        write_file("p", path, "y = 2\n")
        read = Task(tool="read_file",
                    arguments={"project_name": "p", "file_path": path})
        decisions += [_task_decision(read), _task_decision(write)]
        executions += [
            ExecutionDecision(tool="read_file", arguments=read.arguments),
            ExecutionDecision(tool="write_file", arguments=write.arguments)]
    decisions.append(_finish())

    runner = _real_runner(real_tools, decisions, executions)
    result, _ = _run_events(runner, objective="o", project_name="p")
    assert result == "done"
    assert any("STAGNATION" in ctx
               for ctx in runner.planner.received_contexts)


# _command_succeeded: só a primeira linha vale.
def test_command_success_uses_first_line_only(real_tools):
    runner = _real_runner(real_tools, [], [])
    ok_output = ("STATUS: success (exit code 0)\n\nSTDOUT:\nhello\n\n"
                 "STDERR:\n(empty)")
    assert runner._command_succeeded("run_command", ok_output) is True
    tricky = ("STATUS: failure (exit code 1)\n\nSTDOUT:\nSTATUS: success\n\n"
              "STDERR:\n(empty)")
    assert runner._command_succeeded("run_command", tricky) is False
    timeout = "TIMEOUT: execution exceeded 15s and was interrupted."
    assert runner._command_succeeded("run_command", timeout) is False
    assert runner._command_succeeded("write_file", "qualquer coisa") is True


# Timeout preserva itens reais e bloqueia o finish.
def test_timeout_preserves_errors_and_blocks_finish(real_tools):
    llm = StubLLM(["ERRO REAL"])
    checklist = ErrorChecklist(llm)
    runner = _real_runner(real_tools, [], [], error_checklist=checklist)
    # Fase 6: saída com falha parseável (evidência real p/ unificada).
    runner._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "STATUS: failure (exit code 1)\nFAILED t.py::test_x\nboom",
        False, iteration=1)
    assert checklist.pending_count == 1
    calls_before = len(llm.calls)

    runner._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "TIMEOUT: execution exceeded 15s and was interrupted.",
        False, iteration=2)
    # generate NÃO foi chamado de novo; itens reais preservados.
    assert len(llm.calls) == calls_before
    assert checklist.pending_count == 2
    assert any("timed out" in item.description
               for item in checklist._items)

    # Timeout com checklist vazio ainda bloqueia.
    fresh = ErrorChecklist(llm)
    runner2 = _real_runner(real_tools, [], [], error_checklist=fresh)
    runner2._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "TIMEOUT: execution exceeded 15s and was interrupted.",
        False, iteration=1)
    assert fresh.pending_count == 1

    # Sucesso limpa tudo, inclusive a nota de timeout.
    runner._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "STATUS: success (exit code 0)", True, iteration=3)
    assert checklist.pending_count == 0


# Erro de infra (tool lançou exceção) também preserva.
def test_tool_exception_preserves_errors(real_tools):
    llm = StubLLM(["ERRO REAL"])
    checklist = ErrorChecklist(llm)
    runner = _real_runner(real_tools, [], [], error_checklist=checklist)
    runner._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "STATUS: failure (exit code 1)\nFAILED t.py::test_x\nboom", False)
    runner._update_error_checklist(
        "run_command", {"command": "pytest -q"},
        "TOOL EXECUTION ERROR:\nRuntimeError: sandbox explodiu",
        False)
    assert checklist.pending_count == 2


def test_note_infra_failure_dedupes_and_clears():
    llm = StubLLM()
    checklist = ErrorChecklist(llm)
    checklist.note_infra_failure("pytest -q", "excedeu o tempo limite")
    checklist.note_infra_failure("pytest -q", "excedeu o tempo limite")
    assert checklist.pending_count == 1
    checklist.clear()
    assert checklist.pending_count == 0
