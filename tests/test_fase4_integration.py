"""Fase 4 (integração) — as 5 evoluções: contexto, canonical
objective, PT-ASCII, plan sync e persistência.

Isolamento: persistência desligada por padrão neste arquivo (contagens
determinísticas); testes de persistência ligam explicitamente com
diretório temporário.
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.checklist import ProjectChecklist
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.agent.taskstate.canonicalizer import (
    PromptCanonicalizer,
    needs_translation,
    pt_ascii_evidence,
)
from app.agent.taskstate.persistence import (
    archive_task_state,
    load_task_state,
    save_task_state,
    state_path_for_project,
)
from app.agent.taskstate.task_state import PlanItem, TaskState
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.models import LLMResponse, Usage


@pytest.fixture(autouse=True)
def _no_disk_persistence(monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", False)


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


class _ScriptedLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        # Runner lê planner.llm.usage no finish (só métrica).
        self.usage = T.FakeLLM.usage

    def generate(self, messages, component=None, iteration=None,
                 **kwargs):
        self.calls.append({"component": component,
                           "prompt": messages[0].content})
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content, tool_calls=[],
                           usage=Usage(10, 5, 15), provider="scripted")


def _runner(decisions, executions, trace, **overrides):
    return Runner(
        planner=overrides.pop("planner", None)
        or T.FakePlanner(decisions),
        task_decision_maker=overrides.pop(
            "task_decision_maker", None)
        or T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=overrides.pop(
            "operational_memory", None) or T.FakeOperationalMemory(),
        checklist=overrides.pop("checklist", None) or T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )


def _trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "trace4i"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


# --------------------------------------------------------------------------
# Canonicalização PT-ASCII
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "crie um sistema de cadastro de clientes",
    "faca uma api para controlar tarefas",
    "adicione uma pagina de login",
    "corrija os testes",
])
def test_pt_ascii_examples_translate(prompt):
    llm = _ScriptedLLM(content="Create the requested feature now please.")
    result = PromptCanonicalizer(llm=llm).canonicalize(prompt)
    assert result.llm_calls == 1
    assert result.translation_applied is True
    assert result.language == "pt"
    assert result.original_prompt == prompt


@pytest.mark.parametrize("prompt", [
    "Fix it",
    "Do the thing",
    "obj",
    "remove the file",
    "meu objetivo real",
    "Create a notes app",
    "projeto",
    "crie",
])
def test_ambiguous_or_english_never_translates(prompt):
    llm = _ScriptedLLM(content="SHOULD NOT BE USED")
    result = PromptCanonicalizer(llm=llm).canonicalize(prompt)
    assert result.llm_calls == 0
    assert result.translation_applied is False
    assert llm.calls == []


def test_pt_ascii_flag_off_restores_accent_only(monkeypatch):
    monkeypatch.setattr(Config, "pt_ascii_translation", False)
    llm = _ScriptedLLM(content="SHOULD NOT BE USED")
    ascii_pt = "crie um sistema de cadastro de clientes"
    result = PromptCanonicalizer(llm=llm).canonicalize(ascii_pt)
    assert result.llm_calls == 0
    assert result.translation_applied is False
    assert result.language == "pt"  # detectado, não traduzido
    assert result.canonical_prompt == ascii_pt
    # Acentos continuam traduzindo com a flag desligada.
    llm2 = _ScriptedLLM(content="Create a valid system now.")
    accented = "Crie um sistema válido agora."
    result2 = PromptCanonicalizer(llm=llm2).canonicalize(accented)
    assert result2.translation_applied is True


def test_pt_ascii_evidence_unit():
    ev = pt_ascii_evidence("crie um sistema de cadastro")
    assert ev["verbs"] >= 1 and ev["funcs"] >= 1
    assert pt_ascii_evidence("obj") == {"verbs": 0, "funcs": 0,
                                        "nouns": 0}


def test_pt_ascii_failure_falls_back():
    llm = _ScriptedLLM(error=RuntimeError("down"))
    original = "corrija os testes quebrados agora"
    result = PromptCanonicalizer(llm=llm).canonicalize(original)
    assert result.translation_applied is False
    assert result.canonical_prompt == original


def test_translator_prompt_forbids_planning():
    llm = _ScriptedLLM(content="Create the client registry now.")
    PromptCanonicalizer(llm=llm).canonicalize(
        "crie um sistema de cadastro de clientes")
    sent = llm.calls[0]["prompt"]
    assert "do NOT plan" in sent
    assert "add requirements" in sent
    assert "Preserve verbatim" in sent


# --------------------------------------------------------------------------
# TaskState: render params, canonical_objective, task_id, sync_plan
# --------------------------------------------------------------------------

def test_render_compact_without_objective():
    state = TaskState(objective="Build X", requirements=[],
                      canonical_prompt="Build X")
    full = state.render_compact()
    compact = state.render_compact(include_objective=False)
    assert "TASK: Build X" in full
    assert "TASK:" not in compact
    assert "LANGUAGE:" in compact


def test_render_compact_original_ref():
    state = TaskState(objective="Build X",
                      original_prompt="Crie o X válido agora mesmo.")
    assert "ORIGINAL:" not in state.render_compact()
    out = state.render_compact(include_objective=False,
                               original_ref_chars=12)
    assert "ORIGINAL:" in out
    assert "Crie o X" in out
    assert "agora mesmo" not in out  # truncado com marcador


def test_canonical_objective_property():
    assert TaskState(objective="Build X",
                     canonical_prompt="Build X").canonical_objective == (
        "Build X")
    assert TaskState(
        canonical_prompt="Full canonical text").canonical_objective == (
        "Full canonical text")


def test_task_id_roundtrip():
    state = TaskState(task_id="abc123", objective="o")
    clone = TaskState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert clone.task_id == "abc123"
    assert clone == state


def test_sync_plan_lifecycle():
    state = TaskState()
    state.sync_plan([(1, False), (2, False), (3, False)],
                    blocked=False, started=False)
    assert [(i.index, i.status) for i in state.plan] == [
        (1, "pending"), (2, "pending"), (3, "pending")]
    assert [i.item_id for i in state.plan] == [
        "checklist-1", "checklist-2", "checklist-3"]
    # Início: primeiro pendente vira in_progress.
    state.sync_plan([(1, False), (2, False), (3, False)],
                    blocked=False, started=True)
    assert state.plan[0].status == "in_progress"
    # Conclusão via checklist.
    state.sync_plan([(1, True), (2, False), (3, False)],
                    blocked=False, started=True)
    assert state.plan[0].status == "completed"
    assert state.plan[0].done is True
    assert state.plan[1].status == "in_progress"
    # Falha: bloqueia o atual.
    state.sync_plan([(1, True), (2, False), (3, False)],
                    blocked=True, started=True)
    assert state.plan[1].status == "blocked"
    # Correção: desbloqueia sem concluir.
    state.sync_plan([(1, True), (2, False), (3, False)],
                    blocked=False, started=True)
    assert state.plan[1].status == "in_progress"
    # Idempotente: segunda sincronização não muda nada.
    before = state.to_dict()
    state.sync_plan([(1, True), (2, False), (3, False)],
                    blocked=False, started=True)
    assert state.to_dict() == before


def test_sync_plan_tolerates_garbage():
    state = TaskState()
    state.sync_plan(None)
    state.sync_plan([])
    state.sync_plan([("x", True), (None, False)])
    assert state.plan == []
    state.sync_plan([(1, False)], blocked=True, started=True)
    assert state.plan[0].status == "blocked"


def test_plan_render_shows_statuses():
    state = TaskState(objective="o")
    state.sync_plan([(1, True), (2, False)], blocked=False,
                    started=True)
    out = state.render_compact()
    assert "PLAN: 1/2 done (1 in progress)" in out


def test_plan_item_defaults():
    item = PlanItem(index=5)
    assert item.status == "pending"
    assert item.done is False
    assert item.item_id == ""


# --------------------------------------------------------------------------
# Planner: recebe TASK STATE + canonical objective
# --------------------------------------------------------------------------

def test_planner_receives_task_state():
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append((objective, context))
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        planner=_CatchingPlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             _finish()]))
    assert runner.run(objective="Create a notes app",
                      project_name="p") == "done"
    objective, context = seen[0]
    assert objective == "Create a notes app"
    assert "TASK STATE:" in context
    assert "LANGUAGE: en" in context


def test_planner_receives_canonical_objective():
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append((objective, context))
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    planner = _CatchingPlanner([_finish()])
    planner.llm = _ScriptedLLM(
        content="Create a valid notes application.")
    runner = _runner([], [], NullTrace(), planner=planner)
    assert runner.run(objective="Crie um aplicativo de notas válido",
                      project_name="p") == "done"
    objective, context = seen[0]
    assert objective == "Create a valid notes application."
    assert "TASK STATE:" in context
    # Original acessível, não descartado.
    assert "Crie um aplicativo" in context
    assert runner.task_state.original_prompt.startswith("Crie um")


def test_no_unnecessary_duplication_in_planner_context():
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append(context)
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    big_content = "CONTEUDO-INTEGRAL-" * 500
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py",
                                "content": big_content})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        planner=_CatchingPlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             _finish()]))
    runner.run(objective="Create a notes app", project_name="p")
    block = seen[0].split("TASK STATE:")[1].split("HISTÓRICO")[0]
    assert "CONTEUDO-INTEGRAL" not in block
    assert "TASK PAI:" not in block
    assert "STDOUT" not in block
    assert len(block) < 3000  # semântico + referências, não evidência


def test_context_flag_off_preserves_legacy(monkeypatch):
    monkeypatch.setattr(Config, "task_state_context", False)
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append((objective, context))
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        planner=_CatchingPlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             _finish()]))
    runner.run(objective="Create a notes app", project_name="p")
    assert "TASK STATE:" not in seen[0][1]


def test_canonical_flag_off_keeps_original(monkeypatch):
    monkeypatch.setattr(Config, "canonical_objective", False)
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append(objective)
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    planner = _CatchingPlanner([_finish()])
    planner.llm = _ScriptedLLM(
        content="Create a valid notes application.")
    runner = _runner([], [], NullTrace(), planner=planner)
    runner.run(objective="Crie um aplicativo de notas válido",
               project_name="p")
    # Estado canonicaliza, mas o Planner recebe o original.
    assert seen == ["Crie um aplicativo de notas válido"]
    assert runner.task_state.translation_applied is True


# --------------------------------------------------------------------------
# Executor: recebe TASK STATE
# --------------------------------------------------------------------------

def test_executor_receives_task_state():
    seen = []

    class _CatchingMaker(T.FakeTaskDecisionMaker):
        def decide(self, objective, task, context, iteration=None):
            seen.append(context)
            return super().decide(objective, task, context,
                                  iteration=iteration)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        task_decision_maker=_CatchingMaker([_exec_for(task)]))
    runner.run(objective="Create a notes app", project_name="p")
    assert seen and "TASK STATE:" in seen[0]


def test_executor_without_state_gets_intact_context(monkeypatch):
    monkeypatch.setattr(Config, "task_state_context", False)
    seen = []

    class _CatchingMaker(T.FakeTaskDecisionMaker):
        def decide(self, objective, task, context, iteration=None):
            seen.append(context)
            return super().decide(objective, task, context,
                                  iteration=iteration)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        task_decision_maker=_CatchingMaker([_exec_for(task)]))
    runner.run(objective="obj", project_name="p")
    assert seen == ["task context"]


# --------------------------------------------------------------------------
# Checklist ↔ plan (integração com checklist real)
# --------------------------------------------------------------------------

def test_plan_syncs_with_real_checklist(tmp_path):
    llm = _ScriptedLLM(content=json.dumps({"items": ["step one",
                                                      "step two"]}))
    checklist = ProjectChecklist(llm=llm)
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    planner_task = Decision(action=DecisionAction.TASK, task=task)
    planner_task.checklist_progress = [1]
    runner = _runner([planner_task, _finish()], [_exec_for(task)],
                     NullTrace(), checklist=checklist)
    runner.run(objective="Create a notes app", project_name="p")
    plan = [(i.index, i.status) for i in runner.task_state.plan]
    assert plan == [(1, "completed"), (2, "in_progress")]
    assert runner.task_state.plan[0].ref == "checklist #1"


def test_plan_sync_flag_off_keeps_plan_empty(monkeypatch):
    monkeypatch.setattr(Config, "task_plan_sync", False)
    llm = _ScriptedLLM(content=json.dumps({"items": ["step one"]}))
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(),
        checklist=ProjectChecklist(llm=llm))
    runner.run(objective="Create a notes app", project_name="p")
    assert runner.task_state.plan == []


# --------------------------------------------------------------------------
# Persistência
# --------------------------------------------------------------------------

def _persist_on(monkeypatch):
    monkeypatch.setattr(Config, "task_state_persist", True)


def test_save_and_load_roundtrip(projects_root, monkeypatch):
    _persist_on(monkeypatch)
    state = TaskState(original_prompt="o", canonical_prompt="o",
                      task_id="tid123", objective="o")
    state.record_problem("p1", iteration=1)
    result = save_task_state(state, "p")
    assert result["ok"] is True
    assert result["bytes"] > 0
    path = state_path_for_project("p")
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    loaded = load_task_state("p")
    assert loaded == state


def test_load_missing_or_invalid_returns_none(projects_root,
                                              monkeypatch):
    _persist_on(monkeypatch)
    assert load_task_state("nope") is None
    bad = state_path_for_project("p")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{json invalido", encoding="utf-8")
    assert load_task_state("p") is None
    bad.write_text(json.dumps({"objective": "x"}), encoding="utf-8")
    assert load_task_state("p") is None  # sem task_id


def test_different_task_archives_old_state(projects_root, monkeypatch):
    _persist_on(monkeypatch)
    old = TaskState(task_id="oldtask0001", objective="old")
    assert save_task_state(old, "p")["ok"] is True
    assert archive_task_state("p", "oldtask0001") is True
    assert not state_path_for_project("p").exists()
    backups = list(state_path_for_project("p").parent.glob(
        "task_state.prev-*.archived.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["task_id"] == "oldtask0001"


def test_persist_flag_off_writes_nothing(projects_root, tmp_path):
    trace = _trace(tmp_path)
    runner = _runner([_finish()], [], trace)
    runner.run(objective="obj", project_name="p")
    assert not (tmp_path / "p" / ".aidev" / "task_state.json").exists()
    assert _events(trace, "task_state_persist") == []


def test_filesystem_error_returns_ok_false():
    result = save_task_state(TaskState(task_id="t"), "../fora")
    assert result["ok"] is False
    assert "error" in result


def test_unchanged_state_is_not_rewritten(projects_root, monkeypatch):
    _persist_on(monkeypatch)
    state = TaskState(task_id="t", objective="o")
    assert save_task_state(state, "p")["ok"] is True
    path = state_path_for_project("p")
    first = path.stat().st_mtime_ns
    # Runner não reescreve sem mudança (assinatura igual).
    runner = _runner([_finish()], [], NullTrace())
    runner._task_state_project = "p"
    runner.task_state = state
    runner._task_state_saved_sig = runner._task_state_signature()
    runner._save_task_state("noop")
    assert path.stat().st_mtime_ns == first


def test_atomic_write_leaves_no_tmp(projects_root, monkeypatch):
    _persist_on(monkeypatch)
    save_task_state(TaskState(task_id="t"), "p")
    leftovers = list(state_path_for_project("p").parent.glob("*.tmp"))
    assert leftovers == []


# --------------------------------------------------------------------------
# Runner: init/persist/restore/trace/erros
# --------------------------------------------------------------------------

def test_state_initialized_persisted_and_traced(projects_root, tmp_path,
                                                monkeypatch):
    _persist_on(monkeypatch)
    trace = _trace(tmp_path)
    runner = _runner([_finish()], [], trace)
    runner.run(objective="Create a notes app", project_name="p")
    assert (tmp_path / "p" / ".aidev" / "task_state.json").exists()
    assert _events(trace, "task_state_init")
    assert _events(trace, "task_state_persist")
    snapshot = _events(trace, "task_state_init")[0]["state"]
    assert snapshot["original_prompt"] == "Create a notes app"
    assert snapshot["task_id"]


def test_same_task_restores_state(projects_root, tmp_path, monkeypatch):
    _persist_on(monkeypatch)
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})

    def _run_once():
        runner = _runner(
            [Decision(action=DecisionAction.TASK, task=task),
             _finish()],
            [_exec_for(task)], NullTrace())
        runner.run(objective="Create a notes app", project_name="p")
        return runner

    first = _run_once()
    assert len(first.task_state.decisions) == 2
    second = _run_once()
    # Estado restaurado acumula (2 decisões da run anterior + 2 novas).
    assert len(second.task_state.decisions) == 4


def test_restore_event_recorded(projects_root, tmp_path, monkeypatch):
    _persist_on(monkeypatch)
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    _runner([Decision(action=DecisionAction.TASK, task=task),
             _finish()],
            [_exec_for(task)], NullTrace()).run(
                objective="Create a notes app", project_name="p")
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace)
    runner.run(objective="Create a notes app", project_name="p")
    restores = _events(trace, "task_state_restore")
    assert len(restores) == 1
    assert restores[0]["decisions"] == 2


def test_task_state_update_event_shape(projects_root, tmp_path,
                                       monkeypatch):
    _persist_on(monkeypatch)
    trace = _trace(tmp_path)
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace)
    runner.run(objective="Create a notes app", project_name="p")
    updates = _events(trace, "task_state_update")
    assert updates, "esperava eventos de evolução do estado"
    first = updates[0]
    assert first["iteration"] == 1
    assert "decisions" in first and "progress" in first
    assert "open_problems" in first


def test_persist_failure_does_not_break_run(tmp_path, monkeypatch):
    _persist_on(monkeypatch)
    runner = _runner([_finish()], [], NullTrace())
    # ".." escapa do diretório de projetos → save falha sempre,
    # mas a run continua e o estado em memória existe.
    assert runner.run(objective="obj", project_name="../fora") == "done"
    assert runner.task_state is not None


def test_all_flags_off_restores_legacy(projects_root, tmp_path,
                                       monkeypatch):
    for flag in ("task_state_context", "task_state_persist",
                 "canonical_objective", "pt_ascii_translation",
                 "task_plan_sync"):
        monkeypatch.setattr(Config, flag, False)
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append((objective, context))
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    trace = _trace(tmp_path)
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], trace,
        planner=_CatchingPlanner(
            [Decision(action=DecisionAction.TASK, task=task),
             _finish()]))
    assert runner.run(objective="Crie um app válido",
                      project_name="p") == "done"
    objective, context = seen[0]
    assert objective == "Crie um app válido"  # original, sem tradução
    assert "TASK STATE:" not in context
    assert runner.task_state.plan == []
    assert not (tmp_path / "p" / ".aidev" / "task_state.json").exists()
    assert _events(trace, "task_state_persist") == []
