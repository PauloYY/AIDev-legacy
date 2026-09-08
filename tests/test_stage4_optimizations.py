"""Etapa 4 — redução de chamadas e contexto do LLM.

Cobre os 18 casos obrigatórios + política de skip + métricas, sem
alterar nenhum teste existente:
1-2.   Summary Updater pulado p/ leituras; mantido p/ escritas/testes.
3-5.   Contexto compacto mantém objetivo/estado/erros.
6-7.   Outputs grandes compactados; info crítica de testes preservada.
8-13.  Agrupamento seguro (leituras em lote; resto sequencial).
14-17. Short repair, full retry, finish gate e final verification intactos.
18.    Comportamento legado com flags desativadas.
Métricas: component_stats + novas linhas do perf summary.
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.context.operational_memory import OperationalMemory
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.perf import (
    AgentStats,
    format_llm_component_section,
    format_performance_summary,
)
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.planning.dependency import Dependency
from app.agent.runner import Runner
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.models import Usage
from app.llm.usage import UsageTracker
from app.tools.registry import ToolRegistry


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _task(tool, arguments, dependencies=None):
    return Task(tool=tool, arguments=arguments,
                dependencies=dependencies or [])


def _finish(content="done"):
    return Decision(action=DecisionAction.FINISH, content=content)


def _exec_for(task):
    return ExecutionDecision(tool=task.tool, arguments=task.arguments)


class _RecordingChecklist(T.FakeChecklist):
    def render(self):
        return "OBJECTIVE CHECKLIST\n[x] 1. construir o projeto"


class _PendingErrorChecklist(T.FakeErrorChecklist):
    def __init__(self):
        self.items = ["teste_x falhou: esperado 1, obteve 2"]

    def render(self):
        return "CURRENT ERROR CHECKLIST\n- 1. teste_x falhou"

    @property
    def pending_count(self):
        return 1


class _RecordingErrorChecklist(T.FakeErrorChecklist):
    def __init__(self):
        self.captured = []

    def generate(self, command, output, iteration=None):
        self.captured.append((command, output))


class _ObjectiveCatchingPlanner(T.FakePlanner):
    def __init__(self, decisions):
        super().__init__(decisions)
        self.objectives = []

    def plan(self, objective, context, iteration=None, request_type=None):
        self.objectives.append(objective)
        return super().plan(objective, context, iteration=iteration,
                            request_type=request_type)


def _runner(decisions, executions, trace, tools=None, operational_memory=None,
            updater=None, error_checklist=None, final_verification=None,
            task_context_builder=None):
    return Runner(
        planner=T.FakePlanner(decisions),
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=(
            task_context_builder or T.FakeTaskContextBuilder()),
        tools=tools or T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=(
            updater or T.FakeSummaryUpdater(T.FakeSummary("resumo"))),
        validator=T.FakeValidator(),
        operational_memory=operational_memory or T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=error_checklist or T.FakeErrorChecklist(),
        final_verification=final_verification or T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )


def _trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "trace"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


# --------------------------------------------------------------------------
# 1-2. política do Summary Updater
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tool,arguments", [
    ("read_file", {"project_name": "p", "file_path": "a.py"}),
    ("list_files", {"project_name": "p"}),
    ("find_references", {"project_name": "p", "symbol": "x"}),
    ("list_symbols", {"project_name": "p", "file_path": "a.py"}),
])
def test_updater_skipped_for_pure_reads(tmp_path, monkeypatch, tool,
                                        arguments):
    """Caso 1: leitura pura não dispara o Summary Updater."""
    monkeypatch.setattr(Config, "smart_summary", True)
    trace = _trace(tmp_path)
    updater = T.FakeSummaryUpdater(T.FakeSummary("resumo"))
    task = _task(tool, arguments)
    runner = _runner([Decision(action=DecisionAction.TASK, task=task),
                      _finish()],
                     [_exec_for(task)], trace, updater=updater)

    assert runner.run(objective="obj", project_name="p") == "done"
    assert updater.calls == 0
    assert runner._stats.summary_skipped == 1
    skipped = _events(trace, "summary_skipped")
    assert len(skipped) == 1
    assert skipped[0]["tool"] == tool
    assert skipped[0]["reason"] == "read_only_no_state_change"
    assert _events(trace, "summary_updated") == []


@pytest.mark.parametrize("tool,arguments", [
    ("write_file", {"project_name": "p", "file_path": "a.py",
                    "content": "x"}),
    ("run_command", {"project_name": "p", "command": "echo oi"}),
    ("check_project", {"project_name": "p"}),
])
def test_updater_still_called_when_needed(tmp_path, monkeypatch, tool,
                                           arguments):
    """Caso 2: escrita/teste/checagem continuam atualizando o resumo."""
    monkeypatch.setattr(Config, "smart_summary", True)
    # Comportamento da Etapa 4/5: sem os skips estendidos da Etapa 6
    # (cobertos em test_stage6_executor_summary.py).
    monkeypatch.setattr(Config, "compact_executor", False)
    trace = _trace(tmp_path)
    updater = T.FakeSummaryUpdater(T.FakeSummary("resumo"))
    task = _task(tool, arguments)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace, updater=updater,
        operational_memory=OperationalMemory(T.FakeToolRegistry()),
    )

    assert runner.run(objective="obj", project_name="p") == "done"
    assert updater.calls == 1
    assert runner._stats.summary_skipped == 0
    assert len(_events(trace, "summary_updated")) == 1
    assert _events(trace, "summary_skipped") == []


# --------------------------------------------------------------------------
# 3-5. contexto compacto preserva informação
# --------------------------------------------------------------------------

def test_compact_context_keeps_objective_and_state(
        projects_root, tmp_path, monkeypatch):
    """Casos 3-4: objetivo original e estado atual preservados."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_context", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    planner = _ObjectiveCatchingPlanner([
        Decision(action=DecisionAction.TASK,
                 task=_task("read_file", {"project_name": "p",
                                          "file_path": "a.py"})),
        _finish(),
    ])
    trace = _trace(tmp_path)
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="read_file",
                              arguments={"project_name": "p",
                                         "file_path": "a.py"})]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(tools),
        checklist=_RecordingChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )

    assert runner.run(objective="obj", project_name="p") == "done"
    # Objetivo chega intacto ao Planner (template inalterado).
    assert planner.objectives and all(o == "obj" for o in planner.objectives)
    # Estado atual: listagem integral quando a estrutura mudou.
    assert "- a.py" in planner.received_contexts[0]
    assert "CURRENT PROJECT FILES" in planner.received_contexts[0]
    assert "construir o projeto" in planner.received_contexts[0]


def test_compact_context_preserves_pending_errors(
        projects_root, tmp_path, monkeypatch):
    """Caso 5: checklist de erros pendente sobrevive à compactação."""
    monkeypatch.setattr(Config, "compact_context", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    memory = OperationalMemory(tools)
    runner = _runner([], [], _trace(tmp_path), tools=tools,
                     operational_memory=memory,
                     error_checklist=_PendingErrorChecklist())
    runner._file_list_state = None
    block = runner._build_memory_block_tracked("p", "resumo")

    assert "CURRENT ERROR CHECKLIST" in block
    assert "teste_x falhou" in block
    assert "- a.py" in block


def test_unchanged_file_list_is_compact_but_explicit(
        projects_root, tmp_path, monkeypatch):
    """Segunda verificação sem mudanças: linha compacta explícita."""
    monkeypatch.setattr(Config, "compact_context", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    memory = OperationalMemory(tools)
    text1, state1 = memory.render_files_state("p", None)
    assert "- a.py" in text1
    assert isinstance(state1, str) and state1

    text2, state2 = memory.render_files_state("p", state1)
    assert "no changes since last check" in text2
    assert "1 file(s)" in text2
    assert state2 == state1

    write_file("p", "b.py", "y = 2\n")
    text3, state3 = memory.render_files_state("p", state2)
    assert "- b.py" in text3
    assert state3 != state2


# --------------------------------------------------------------------------
# 6-7. compactação de outputs
# --------------------------------------------------------------------------

def test_big_dependency_results_are_capped_head_first():
    """Caso 6: output gigante vira cabeça + marcador (nunca silencioso)."""
    big = "STATUS: success (exit code 0)\n" + "y" * 60000
    builder = TaskContextBuilder(max_result_chars=4000)
    task = _task("write_file", {"project_name": "p"},
                 dependencies=[Dependency(
                     tool="run_command",
                     arguments={"project_name": "p",
                                "command": "pytest"})])
    out = builder.build(task, [big])

    assert out.startswith("PARENT TASK:")
    assert "STATUS: success (exit code 0)" in out
    assert "truncated result" in out
    assert "600" in out  # total indicado
    assert len(out) < 5000


def test_small_results_untouched_and_legacy_default_integral():
    """Sem limite (legado) ou resultado pequeno: byte-idêntico."""
    small = "STATUS: success (exit code 0)\nok"
    task = _task("write_file", {"project_name": "p"},
                 dependencies=[Dependency(
                     tool="run_command",
                     arguments={"project_name": "p", "command": "t"})])
    capped = TaskContextBuilder(max_result_chars=4000).build(task, [small])
    legacy = TaskContextBuilder().build(task, [small])
    assert capped == legacy
    assert small in capped

    big = "z" * 60000
    assert big in TaskContextBuilder().build(task, [big])


def test_error_checklist_still_receives_full_result(tmp_path,
                                                      monkeypatch):
    """Caso 7: extração de erros usa o resultado INTEGRAL.

    Caminho legado (Fase 6 usa análise unificada; ver teste próprio
    de resultado integral em test_fase6_unified_errors.py).
    """
    monkeypatch.setattr(Config, "error_analyzer", False)
    big = "STATUS: failure (exit code 1)\n" + "E" * 60000
    checklist = _RecordingErrorChecklist()
    runner = _runner([], [], NullTrace(),
                     operational_memory=OperationalMemory(
                         T.FakeToolRegistry()),
                     error_checklist=checklist)
    runner._update_error_checklist(
        "run_command",
        {"project_name": "p", "command": "python3 -m pytest -q"},
        big, False, 1)

    assert len(checklist.captured) == 1
    assert checklist.captured[0][1] == big
    assert "STATUS: failure (exit code 1)" in checklist.captured[0][1]


# --------------------------------------------------------------------------
# 8-13. agrupamento seguro
# --------------------------------------------------------------------------

def test_independent_reads_run_as_parallel_batch(
        projects_root, tmp_path, monkeypatch):
    """Caso 8: 3 leituras independentes viram 1 batch."""
    monkeypatch.setattr(Config, "parallel_tools", True)
    from app.tools.filesystem.write_file import write_file
    for name in ("a.py", "b.py", "c.py"):
        write_file("p", name, "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    deps = [Dependency(tool="read_file",
                       arguments={"project_name": "p",
                                  "file_path": name})
            for name in ("a.py", "b.py", "c.py")]
    task = _task("write_file", {"project_name": "p", "file_path": "d.py",
                                "content": "z = 3\n"}, dependencies=deps)
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace, tools=tools,
        operational_memory=OperationalMemory(tools))

    assert runner.run(objective="obj", project_name="p") == "done"
    batches = _events(trace, "parallel_batch")
    assert len(batches) == 1
    assert batches[0]["size"] == 3
    assert runner._stats.parallel_ops == 3


def test_dependent_ops_stay_sequential(projects_root, tmp_path, monkeypatch):
    """Casos 9+12: run_command no meio impede o batch."""
    monkeypatch.setattr(Config, "parallel_tools", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    task = _task("write_file", {"project_name": "p", "file_path": "b.py",
                                "content": "y = 2\n"},
                 dependencies=[
                     Dependency(tool="read_file",
                                arguments={"project_name": "p",
                                           "file_path": "a.py"}),
                     Dependency(tool="run_command",
                                arguments={"project_name": "p",
                                           "command": "python3 --version"}),
                 ])
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace, tools=tools,
        operational_memory=OperationalMemory(tools))

    assert runner.run(objective="obj", project_name="p") == "done"
    assert _events(trace, "parallel_batch") == []
    assert runner._stats.parallel_batches == 0
    assert runner._stats.sequential_ops == 2


def test_independent_writes_across_iterations(projects_root, tmp_path,
                                              monkeypatch):
    """Caso 10: write+write independentes funcionam (sequencial)."""
    monkeypatch.setattr(Config, "smart_summary", True)
    from app.tools.filesystem.write_file import write_file

    tools = ToolRegistry()
    tools.load_defaults()
    t1 = _task("write_file", {"project_name": "p", "file_path": "a.py",
                              "content": "x = 1\n"})
    t2 = _task("write_file", {"project_name": "p", "file_path": "b.py",
                              "content": "y = 2\n"})
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=t1),
         Decision(action=DecisionAction.TASK, task=t2),
         _finish()],
        [_exec_for(t1), _exec_for(t2)], trace, tools=tools,
        operational_memory=OperationalMemory(tools))

    assert runner.run(objective="obj", project_name="p") == "done"
    assert (tmp_path / "p" / "a.py").read_text() == "x = 1\n"
    assert (tmp_path / "p" / "b.py").read_text() == "y = 2\n"


def test_write_then_read_sees_written_content(projects_root, tmp_path,
                                              monkeypatch):
    """Caso 11: read posterior enxerga o write (ordem preservada)."""
    monkeypatch.setattr(Config, "smart_summary", True)
    tools = ToolRegistry()
    tools.load_defaults()
    planner = T.FakePlanner([
        Decision(action=DecisionAction.TASK,
                 task=_task("write_file", {"project_name": "p",
                                           "file_path": "a.py",
                                           "content": "conteudo-x-123\n"})),
        Decision(action=DecisionAction.TASK,
                 task=_task("read_file", {"project_name": "p",
                                          "file_path": "a.py"})),
        _finish(),
    ])
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="write_file",
                              arguments={"project_name": "p",
                                         "file_path": "a.py",
                                         "content": "conteudo-x-123\n"}),
            ExecutionDecision(tool="read_file",
                              arguments={"project_name": "p",
                                         "file_path": "a.py"}),
        ]),
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
        execution_trace=NullTrace(),
    )

    assert runner.run(objective="obj", project_name="p") == "done"
    # contexts[0]=iter1, [1]=iter2 (traz o RESULTADO do write),
    # [2]=iter3/finish (traz o RESULTADO do read da iter2).
    assert "File written successfully" in planner.received_contexts[1]
    assert "conteudo-x-123" in planner.received_contexts[2]


def test_parallel_batch_failure_isolated(projects_root, tmp_path,
                                        monkeypatch):
    """Caso 13: falha num item do lote não corrompe os demais."""
    monkeypatch.setattr(Config, "parallel_tools", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    task = _task("write_file", {"project_name": "p", "file_path": "c.py",
                                "content": "z = 3\n"},
                 dependencies=[
                     Dependency(tool="read_file",
                                arguments={"project_name": "p",
                                           "file_path": "a.py"}),
                     Dependency(tool="read_file",
                                arguments={"project_name": "p",
                                           "file_path": "missing.py"}),
                 ])
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace, tools=tools,
        operational_memory=OperationalMemory(tools))

    assert runner.run(objective="obj", project_name="p") == "done"
    dep_results = [e for e in _events(trace, "tool_result")
                   if e.get("dependency") is True]
    assert [e["success"] for e in dep_results] == [True, False]
    assert runner._stats.parallel_batches == 1


# --------------------------------------------------------------------------
# 14-15. retry intacto com flags ligadas (LLM roteirizada)
# --------------------------------------------------------------------------

class _ScriptedProvider:
    def __init__(self, contents):
        self._contents = list(contents)
        self.name = "scripted"

    def generate(self, messages, tools=None):
        from app.llm.models import LLMResponse, Usage
        item = self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=item, tool_calls=[],
                           usage=Usage(10, 5, 15), provider=self.name)


def _scripted_task_json(tool, arguments):
    return json.dumps({"action": "task",
                       "task": {"tool": tool, "arguments": arguments,
                                "dependencies": []}})


def _real_planner_runner(planner_provider, updater_contents, executions,
                         trace, tools):
    from app.agent.execution.validator import TaskValidator
    from app.agent.planning.decision_parser import DecisionParser
    from app.agent.planning.planner import Planner
    from app.llm.client import LLMClient
    from app.agent.context.project_summary import ProjectSummary

    planner = Planner(llm=LLMClient(planner_provider),
                      parser=DecisionParser(tools=tools), tools=tools)

    class _MemSummary:
        def __init__(self):
            self.store = {}
            self.writes = 0

        def read(self, project_name):
            return self.store.get(project_name, "")

        def write(self, project_name, value):
            self.store[project_name] = value
            self.writes += 1

    from app.agent.context.project_summary_updater import (
        ProjectSummaryUpdater)
    updater = ProjectSummaryUpdater(
        llm=LLMClient(_ScriptedProvider(updater_contents)),
        summary=_MemSummary())
    return Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=updater,
        validator=TaskValidator(tools),
        operational_memory=OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    ), updater


def test_short_repair_still_works_with_flags_on(
        projects_root, tmp_path, monkeypatch):
    """Caso 14: inválida → short repair resolve, sem full retry."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_context", True)
    tools = ToolRegistry()
    tools.load_defaults()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    trace = _trace(tmp_path)
    runner, _ = _real_planner_runner(
        _ScriptedProvider([
            "não-json {{{",
            _scripted_task_json("write_file", args),
            json.dumps({"action": "finish", "content": "done"}),
        ]),
        ["resumo novo"],
        [ExecutionDecision(tool="write_file", arguments=args)],
        trace, tools)

    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == ["short_repair"]


def test_full_retry_still_works_with_flags_on(
        projects_root, tmp_path, monkeypatch):
    """Caso 15: curto falha → fallback completo funciona."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_context", True)
    tools = ToolRegistry()
    tools.load_defaults()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    trace = _trace(tmp_path)
    runner, _ = _real_planner_runner(
        _ScriptedProvider([
            "ruim 1 {{{",
            "ruim 2 }}}",
            _scripted_task_json("write_file", args),
            json.dumps({"action": "finish", "content": "done"}),
        ]),
        ["resumo novo"],
        [ExecutionDecision(tool="write_file", arguments=args)],
        trace, tools)

    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == [
        "short_repair", "full_context"]


# --------------------------------------------------------------------------
# 16-17. finish gate e verificação intactos
# --------------------------------------------------------------------------

def test_finish_gate_still_blocks_with_flags_on(tmp_path, monkeypatch):
    """Caso 16: finish com problema bloqueia e depois aceita."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_context", True)

    class _OnceProblems(T.FakeFinalVerification):
        def __init__(self):
            super().__init__(
                result=FinalVerificationResult(
                    FinalVerificationResult.OK))
            self.n = 0

        def verify(self, objective, project_name, summary, tools_execute):
            self.n += 1
            if self.n == 1:
                return FinalVerificationResult(
                    FinalVerificationResult.PROBLEMS_FOUND, "ruim")
            return FinalVerificationResult(FinalVerificationResult.OK)

    trace = _trace(tmp_path)
    runner = _runner([_finish("a"), _finish("b")], [], trace,
                     final_verification=_OnceProblems())

    assert runner.run(objective="obj", project_name="p") == "b"
    assert runner._stats.finish_blocks == 1
    assert [e["gate"] for e in _events(trace, "finish_block")] == [
        "final_verification"]


def test_final_verification_receives_summary(tmp_path, monkeypatch):
    """Caso 17: verificação final recebe o resumo vigente."""
    monkeypatch.setattr(Config, "smart_summary", True)

    class _MemSummary:
        def __init__(self):
            self.store = {}

        def read(self, project_name):
            return self.store.get(project_name, "")

        def write(self, project_name, value):
            self.store[project_name] = value

    class _WriteThroughUpdater(T.FakeSummaryUpdater):
        def update(self, objective, project_name, task, result,
                   iteration=None):
            self.calls += 1
            self.summary.write(project_name, "novo resumo")
            return "novo resumo"

    mem = _MemSummary()
    project_context = T.FakeProjectContext()
    project_context.summary = mem
    final = T.FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK))
    task = _task("write_file", {"project_name": "p", "file_path": "a.py",
                                "content": "x"})
    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.TASK, task=task), _finish()]),
        task_decision_maker=T.FakeTaskDecisionMaker([_exec_for(task)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=project_context,
        project_summary_updater=_WriteThroughUpdater(mem),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=final,
        execution_trace=NullTrace(),
    )

    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(final.verify_calls) == 1
    assert final.verify_calls[0]["summary"] == "novo resumo"


# --------------------------------------------------------------------------
# 18. comportamento legado com flags off
# --------------------------------------------------------------------------

def test_legacy_updater_called_for_reads_when_flag_off(
        tmp_path, monkeypatch):
    """Caso 18a: SMART off → updater roda até p/ leitura."""
    monkeypatch.setattr(Config, "smart_summary", False)
    trace = _trace(tmp_path)
    updater = T.FakeSummaryUpdater(T.FakeSummary("resumo"))
    task = _task("read_file", {"project_name": "p", "file_path": "a.py"})
    runner = _runner([Decision(action=DecisionAction.TASK, task=task),
                      _finish()],
                     [_exec_for(task)], trace, updater=updater)

    assert runner.run(objective="obj", project_name="p") == "done"
    assert updater.calls == 1
    assert runner._stats.summary_skipped == 0
    assert _events(trace, "summary_skipped") == []
    assert len(_events(trace, "summary_updated")) == 1


def test_legacy_full_listing_when_compact_off(projects_root, tmp_path,
                                             monkeypatch):
    """Caso 18b: COMPACT off → listagem integral sempre."""
    monkeypatch.setattr(Config, "smart_summary", False)
    monkeypatch.setattr(Config, "compact_context", False)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = ToolRegistry()
    tools.load_defaults()
    planner = T.FakePlanner([
        Decision(action=DecisionAction.TASK,
                 task=_task("read_file", {"project_name": "p",
                                          "file_path": "a.py"})),
        Decision(action=DecisionAction.TASK,
                 task=_task("read_file", {"project_name": "p",
                                          "file_path": "a.py"})),
        _finish(),
    ])
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="read_file",
                              arguments={"project_name": "p",
                                         "file_path": "a.py"}),
            ExecutionDecision(tool="read_file",
                              arguments={"project_name": "p",
                                         "file_path": "a.py"}),
        ]),
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
        execution_trace=_trace(tmp_path),
    )

    assert runner.run(objective="obj", project_name="p") == "done"
    assert "- a.py" in planner.received_contexts[1]
    assert "no changes since last check" not in planner.received_contexts[1]


# --------------------------------------------------------------------------
# métricas: component_stats + novas linhas do resumo
# --------------------------------------------------------------------------

def test_component_stats_aggregates_per_component():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "prov", component="Planner",
                   iteration=1, prompt_chars=400, completion_chars=40,
                   duration_ms=1000.0, attempt=1)
    tracker.record(Usage(50, 5, 55), "prov", component="Planner",
                   iteration=2, prompt_chars=800, completion_chars=20,
                   duration_ms=3000.0, attempt=1, request_type="short_repair")
    tracker.record(Usage(20, 4, 24), "prov", component="TaskDecisionMaker",
                   iteration=1, prompt_chars=100, completion_chars=10,
                   duration_ms=500.0, attempt=2, success=False,
                   error="X")

    stats = tracker.component_stats()
    planner = stats["Planner"]
    assert planner["calls"] == 2
    assert planner["prompt_chars_sum"] == 1200
    assert planner["prompt_chars_avg"] == 600
    assert planner["prompt_chars_max"] == 800
    assert planner["prompt_tokens"] == 150
    assert planner["completion_tokens"] == 15
    assert planner["total_tokens"] == 165
    assert planner["latency_ms_total"] == 4000.0
    assert planner["latency_ms_avg"] == 2000.0
    assert planner["latency_ms_max"] == 3000.0
    assert planner["retries"] == 0

    executor = stats["TaskDecisionMaker"]
    assert executor["calls"] == 1
    assert executor["retries"] == 1
    assert executor["failures"] == 1


def test_perf_summary_reports_calls_latency_and_components():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "prov", component="Planner",
                   iteration=1, prompt_chars=400, completion_chars=40,
                   duration_ms=1000.0)
    tracker.record(Usage(50, 5, 55), "prov", component="Planner",
                   iteration=2, prompt_chars=200, completion_chars=20,
                   duration_ms=2000.0, request_type="short_repair")
    tracker.record(Usage(30, 6, 36), "prov",
                   component="TaskDecisionMaker", iteration=1,
                   prompt_chars=120, completion_chars=12,
                   duration_ms=700.0)
    tracker.record(Usage(40, 8, 48), "prov",
                   component="ProjectSummaryUpdater", iteration=1,
                   prompt_chars=160, completion_chars=16,
                   duration_ms=900.0)

    summary = format_performance_summary(
        "SUCCESS", tracker, {}, AgentStats(summary_skipped=2))

    assert "Executor calls: 1" in summary
    assert "Summary calls: 1" in summary
    assert "Summary skipped: 2" in summary
    assert "Planner retries: 1 (short: 1, full: 0)" in summary
    assert "Total latency:" in summary
    assert "LLM by component:" in summary
    assert "Planner" in summary.split("LLM by component:")[1]


def test_perf_summary_without_usage_keeps_working():
    summary = format_performance_summary("SUCCESS")
    assert "Wall time:" in summary
    assert "Iterations: 0" in summary
    assert "LLM by component:" not in summary


def test_component_section_tolerates_broken_usage():
    assert format_llm_component_section(None) == ""
    assert format_llm_component_section(object()) == ""
