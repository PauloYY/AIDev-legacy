"""Fase 4 — TaskState + canonicalização + Interpreter + integração.

TaskState (criação/serialização/render), canonicalização PT→EN
(com/sem LLM, fallbacks), Interpreter determinístico (sem invenção) e
integração ao Runner (estado criado/mantido, fluxo antigo intacto).
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.config import Config
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.runner import Runner
from app.agent.taskstate.canonicalizer import (
    PromptCanonicalizer,
    detect_language,
    needs_translation,
    normalize_prompt,
)
from app.agent.taskstate.interpreter import TaskInterpreter
from app.agent.taskstate.task_state import TaskState
from app.agent.trace import ExecutionTrace, NullTrace
from app.llm.models import LLMResponse, Usage


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_disk_persistence(monkeypatch):
    """Isolamento: estes testes contam decisões/progresso exatos; um
    restore de outra run (mesmo task_id "obj") os tornaria frágeis.
    Persistência é coberta em test_fase4_integration.py com tmp_path."""
    monkeypatch.setattr(Config, "task_state_persist", False)

def _task(tool, arguments, dependencies=None):
    return Task(tool=tool, arguments=arguments,
                dependencies=dependencies or [])


def _finish(content="done"):
    return Decision(action=DecisionAction.FINISH, content=content)


def _exec_for(task):
    return ExecutionDecision(tool=task.tool, arguments=task.arguments)


class _ScriptedLLM:
    """Double de LLMClient p/ canonicalização (captura componente)."""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

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
        planner=T.FakePlanner(decisions),
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=overrides.pop(
            "operational_memory", None) or T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=overrides.pop(
            "error_checklist", None) or T.FakeErrorChecklist(),
        final_verification=overrides.pop(
            "final_verification", None) or T.FakeFinalVerification(
                result=FinalVerificationResult(
                    FinalVerificationResult.OK)),
        execution_trace=trace,
    )


# --------------------------------------------------------------------------
# TaskState: criação, defaults, campos
# --------------------------------------------------------------------------

def test_state_creation_defaults():
    state = TaskState()
    assert state.original_prompt == ""
    assert state.canonical_prompt == ""
    assert state.objective == ""
    assert state.requirements == []
    assert state.constraints == []
    assert state.ambiguities == []
    assert state.plan == []
    assert state.decisions == []
    assert state.progress == []
    assert state.problems == []
    assert state.corrections == []
    assert state.verification == []
    assert state.translation_applied is False
    assert state.open_problems == []
    assert state.progress_summary == {"total": 0, "succeeded": 0,
                                      "failed": 0, "decisions": 0}


def test_state_record_methods_never_raise():
    state = TaskState()
    state.record_decision(None, None, None)
    state.record_progress(None, None, None)
    state.record_problem("")
    state.record_problem(None)
    state.record_correction("")
    state.record_verification(None, False)
    # Entradas ruins viram registros defensivos (iteração 0), nunca exceção;
    # descrições vazias são descartadas.
    assert state.progress_summary["total"] == 1
    assert state.decisions[0].iteration == 0
    assert state.problems == []
    assert state.corrections == []


def test_state_progress_and_resolve():
    state = TaskState()
    state.record_decision(1, "task", "write_file", "a.py", "write")
    state.record_progress(1, "write_file", True, "a.py", "STATUS: ok")
    state.record_progress(2, "run_command", False, None, "STATUS: falha")
    state.record_problem("test_x failed", iteration=2)
    assert len(state.open_problems) == 1
    assert state.progress_summary == {"total": 2, "succeeded": 1,
                                      "failed": 1, "decisions": 1}
    assert state.resolve_problems() == 1
    assert state.open_problems == []
    assert state.resolve_problems() == 0


# --------------------------------------------------------------------------
# TaskState: serialização
# --------------------------------------------------------------------------

def test_state_to_dict_json_serializable():
    state = TaskState(original_prompt="  Crie X  ",
                      canonical_prompt="Create X", language="pt",
                      translation_applied=True, objective="Create X")
    state.record_problem("boom", iteration=1)
    data = state.to_dict()
    assert data["original_prompt"] == "  Crie X  "
    assert data["translation_applied"] is True
    assert json.dumps(data, sort_keys=True)  # serializável
    assert data["problems"][0]["description"] == "boom"
    assert data["problems"][0]["resolved"] is False


def test_state_from_dict_roundtrip():
    state = TaskState(original_prompt="o", canonical_prompt="o",
                      language="en", objective="o")
    state.record_decision(1, "task", "write_file", "a.py")
    state.record_verification("check_project", True, "", 2)
    clone = TaskState.from_dict(json.loads(
        json.dumps(state.to_dict(), sort_keys=True)))
    assert clone == state
    assert clone.to_dict() == state.to_dict()


def test_state_from_dict_tolerates_garbage():
    assert TaskState.from_dict(None) == TaskState()
    assert TaskState.from_dict("lixo") == TaskState()
    partial = TaskState.from_dict({"objective": "x", "nope": [1, 2],
                                   "problems": [{"description": "p"}]})
    assert partial.objective == "x"
    assert len(partial.problems) == 1
    assert TaskState.from_dict(
        {"problems": ["not-a-dict"]}).problems == []


def test_state_serialization_deterministic():
    def _build():
        state = TaskState(original_prompt="o", canonical_prompt="o",
                          objective="o")
        state.record_decision(1, "task", "write_file", "a.py")
        state.record_problem("p", iteration=1)
        return json.dumps(state.to_dict(), sort_keys=True)

    assert _build() == _build()


# --------------------------------------------------------------------------
# TaskState: render_compact
# --------------------------------------------------------------------------

def test_render_compact_sections_and_empty_markers():
    state = TaskState(objective="Create notes app", language="en")
    out = state.render_compact()
    assert "TASK: Create notes app" in out
    assert "REQUIREMENTS: none" in out
    assert "CONSTRAINTS: none" in out
    assert "PROBLEMS: none" in out
    assert "VERIFICATION: none" in out
    assert "PROGRESS: 0/0 succeeded (0 decisions)" in out


def test_render_compact_distinguishes_planned_executed_verified():
    state = TaskState(objective="obj")
    state.record_problem("test_x failed", iteration=1)
    state.record_progress(1, "run_command", False)
    state.record_correction("tests green", iteration=2)
    state.resolve_problems()
    state.record_verification("check_project", True, "", 2)
    out = state.render_compact()
    assert "PROBLEMS: 0 open / 1 total" in out
    assert "CORRECTIONS: 1" in out
    assert "VERIFICATION: check_project: PASS" in out
    assert "PROGRESS: 0/1 succeeded" in out


def test_render_compact_is_bounded_and_deterministic():
    state = TaskState(objective="o" * 5000)
    for i in range(30):
        state.record_problem(f"problem {i}", iteration=i)
    out1 = state.render_compact()
    out2 = state.render_compact()
    assert out1 == out2
    assert len(out1) < 6000
    # Janela de detalhe (Fase 6): 30 - 8 por render.
    assert "[+22 more]" in out1


# --------------------------------------------------------------------------
# Canonicalização: normalização e idioma
# --------------------------------------------------------------------------

def test_normalize_preserves_lines_and_content():
    assert normalize_prompt("  Crie X  \n\n  - item 1  \n- item 2\n\n\n") == (
        "Crie X\n\n- item 1\n- item 2")
    assert normalize_prompt("") == ""
    assert normalize_prompt(None) == ""


def test_detect_language():
    assert detect_language("Crie um sistema de notas válido") == "pt"
    assert detect_language("Create a simple notes system") == "en"
    assert detect_language("obj") == "en"
    assert detect_language("") == "en"  # vazio: default inofensivo
    assert detect_language(None) == "unknown"


def test_needs_translation_only_on_strong_signal():
    # Integração Fase 4: acentos OU PT-ASCII com forte evidência
    # (verbo + outro token). Token isolado nunca dispara.
    assert needs_translation("Crie um sistema válido") is True
    assert needs_translation("meu objetivo real") is False
    assert needs_translation("Analise o projeto e descreva",
                             allow_pt_ascii=False) is False
    assert needs_translation("Create notes app") is False
    assert needs_translation("crie um sistema de cadastro",
                             allow_pt_ascii=True) is True
    assert needs_translation("crie um sistema de cadastro",
                             allow_pt_ascii=False) is False


def test_english_prompt_needs_no_llm():
    llm = _ScriptedLLM(content="SHOULD NOT BE USED")
    result = PromptCanonicalizer(llm=llm).canonicalize(
        "Create a notes app with add/list/remove and pytest tests.")
    assert result.llm_calls == 0
    assert result.translation_applied is False
    assert result.language == "en"
    assert result.canonical_prompt.startswith("Create a notes app")
    assert result.original_prompt.startswith("Create a notes app")
    assert llm.calls == []


def test_portuguese_ascii_needs_no_llm_but_detected():
    llm = _ScriptedLLM(content="SHOULD NOT BE USED")
    result = PromptCanonicalizer(llm=llm).canonicalize(
        "meu objetivo real")
    assert result.llm_calls == 0
    assert result.translation_applied is False
    assert result.language == "pt"
    assert result.canonical_prompt == "meu objetivo real"


def test_portuguese_prompt_translated_with_llm():
    llm = _ScriptedLLM(
        content="Create a simple task management system with notes.")
    original = "Crie um sistema simples de controle de tarefas válidas."
    result = PromptCanonicalizer(llm=llm).canonicalize(original)
    assert result.llm_calls == 1
    assert result.translation_applied is True
    assert result.language == "pt"
    assert result.canonical_prompt.startswith("Create a simple")
    assert result.original_prompt == original  # original intacto
    assert llm.calls[0]["component"] == "PromptCanonicalizer"


def test_translation_preserves_requirements_and_identifiers():
    llm = _ScriptedLLM(content=(
        "Create notes.py with add_note and list_notes, then validate "
        "with python -m pytest -q before finishing."))
    original = ("Crie o notes.py com add_note e list_notes, valide com "
                "python -m pytest -q antes de finalizar a validação.")
    result = PromptCanonicalizer(llm=llm).canonicalize(original)
    assert result.translation_applied is True
    for token in ("notes.py", "add_note", "list_notes",
                  "python -m pytest -q"):
        assert token in result.canonical_prompt


def test_translation_does_not_invent_or_plan():
    llm = _ScriptedLLM(content="Create the valid app.")
    result = PromptCanonicalizer(llm=llm).canonicalize("Crie o app válido.")
    # Tradução é linguística: sem arquitetura/decisões no prompt enviado.
    sent = llm.calls[0]["prompt"]
    assert "do NOT plan" in sent
    assert "add requirements" in sent


def test_empty_prompt_behavior():
    result = PromptCanonicalizer(llm=None).canonicalize("   \n  ")
    assert result.canonical_prompt == ""
    assert result.error == "empty_prompt"
    assert result.llm_calls == 0


def test_translation_fallback_on_error():
    llm = _ScriptedLLM(error=RuntimeError("LLM down"))
    original = "Crie um sistema válido e simples."
    result = PromptCanonicalizer(llm=llm).canonicalize(original)
    assert result.translation_applied is False
    assert result.canonical_prompt == normalize_prompt(original)
    assert "translation_error" in (result.error or "")
    assert result.llm_calls == 0


def test_translation_fallback_without_llm():
    result = PromptCanonicalizer(llm=None).canonicalize(
        "Crie um sistema válido.")
    assert result.translation_applied is False
    assert result.canonical_prompt == "Crie um sistema válido."
    assert result.language == "pt"


def test_degenerate_translation_rejected():
    for bad in ("", "   ", "ok", "Sure!"):
        llm = _ScriptedLLM(content=bad)
        result = PromptCanonicalizer(llm=llm).canonicalize(
            "Crie um sistema simples de controle de tarefas válidas.")
        assert result.translation_applied is False
        assert result.error == "translation_rejected_degenerate"


def test_original_prompt_never_altered():
    original = "  Crie   X válido.\n\n- item   1\n"
    result = PromptCanonicalizer(llm=None).canonicalize(original)
    assert result.original_prompt == original
    assert result.canonical_prompt == "Crie X válido.\n\n- item 1"


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------

def test_interpreter_objective():
    interp = TaskInterpreter().interpret(
        "Create a notes app. It must support add and remove.")
    assert interp.objective == "Create a notes app."


def test_interpreter_requirements_from_bullets():
    interp = TaskInterpreter().interpret(
        "Create a notes app:\n- add_note function\n- list_notes function\n"
        "* remove_note function")
    assert [r.text for r in interp.requirements] == [
        "add_note function", "list_notes function",
        "remove_note function"]
    assert [r.req_id for r in interp.requirements] == ["R1", "R2", "R3"]
    assert all(r.origin == "interpreted" for r in interp.requirements)


def test_interpreter_constraints_en_and_pt():
    interp = TaskInterpreter().interpret(
        "Build the API. Use only stdlib. Validate with pytest before "
        "finishing.\nCrie sem interface gráfica.")
    texts = [c.text for c in interp.constraints]
    assert any("only stdlib" in t for t in texts)
    assert any("before finishing" in t for t in texts)
    assert any("sem interface" in t for t in texts)


def test_interpreter_bullet_constraint_is_not_requirement():
    interp = TaskInterpreter().interpret(
        "Tasks:\n- implement Cart\n- never use a database")
    assert [r.text for r in interp.requirements] == ["implement Cart"]
    assert [c.text for c in interp.constraints] == [
        "never use a database"]


def test_interpreter_ambiguities():
    interp = TaskInterpreter().interpret(
        "Create something appropriate, etc. Should it support undo?")
    reasons = [a.reason for a in interp.ambiguities]
    assert any("etc" in r for r in reasons)
    assert "open_question" in reasons


def test_interpreter_empty_and_garbage():
    assert TaskInterpreter().interpret("") == TaskInterpreter(
    ).interpret("")
    empty = TaskInterpreter().interpret("   ")
    assert empty.objective == "" and empty.requirements == []
    assert TaskInterpreter().interpret(None).objective == ""


def test_interpreter_does_not_invent():
    interp = TaskInterpreter().interpret(
        "Create a simple notes manager with in-memory storage.")
    assert interp.requirements == []  # sem lista → sem requisitos
    assert interp.constraints == []
    assert interp.objective.startswith("Create a simple notes manager")


def test_interpreter_complex_prompt():
    prompt = ("Create a task manager with a JSON backend.\n"
              "- add_task(title) adds a task\n"
              "- list_tasks() lists all tasks\n"
              "- remove_task(id) removes one\n"
              "Validate with python -m pytest -q before finishing.\n"
              "Never use external services, etc.")
    interp = TaskInterpreter().interpret(prompt)
    assert len(interp.requirements) == 3
    assert len(interp.constraints) == 2
    assert len(interp.ambiguities) == 1
    assert interp.objective == "Create a task manager with a JSON backend."


# --------------------------------------------------------------------------
# Integração: estado criado, mantido, fluxo intacto
# --------------------------------------------------------------------------

def test_state_created_at_run_start(tmp_path):
    trace = ExecutionTrace(trace_dir=str(tmp_path / "t"))
    runner = _runner([_finish()], [], trace)
    assert runner.task_state is None
    assert runner.run(objective="Create a notes app",
                      project_name="p") == "done"
    state = runner.task_state
    assert isinstance(state, TaskState)
    assert state.original_prompt == "Create a notes app"
    assert state.canonical_prompt == "Create a notes app"
    assert state.objective == "Create a notes app"
    assert state.translation_applied is False


def test_state_survives_iterations(tmp_path):
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace())
    runner.run(objective="obj", project_name="p")
    state = runner.task_state
    # Toda decisão validada (task + finish) é registrada.
    assert len(state.decisions) == 2
    assert state.decisions[0].tool == "write_file"
    assert state.decisions[0].file_path == "a.py"
    assert state.decisions[1].action == "finish"
    assert len(state.progress) == 1
    assert state.progress[0].success is True
    assert state.progress_summary["decisions"] == 2


def test_interpret_error_does_not_destroy_run(tmp_path, monkeypatch):
    class _BoomLLM(T.FakeLLM):
        def generate(self, *args, **kwargs):
            raise RuntimeError("LLM down")

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    planner = T.FakePlanner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()])
    planner.llm = _BoomLLM()
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker([_exec_for(task)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    # Prompt acentuado + LLM quebrada → fallback, run continua.
    assert runner.run(objective="Crie um app válido",
                      project_name="p") == "done"
    assert runner.task_state.original_prompt == "Crie um app válido"
    assert runner.task_state.translation_applied is False


def test_planner_executor_unaffected(tmp_path):
    prompts = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            prompts.append(objective)
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    # PT com acento: fluxo atual continua recebendo o ORIGINAL (Fase 4
    # não injeta o canônico nos prompts — migração futura).
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace())
    runner.planner = _CatchingPlanner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()])
    assert runner.run(objective="Crie um app válido",
                      project_name="p") == "done"
    assert prompts and all(p == "Crie um app válido" for p in prompts)


def test_finish_gate_and_verification_recorded(tmp_path):
    runner = _runner([_finish("a"), _finish("b")], [],
                     NullTrace(),
                     **{"final_verification": _OnceProblems()})
    assert runner.run(objective="obj", project_name="p") == "b"
    gates = {v.gate: v.passed for v in runner.task_state.verification}
    assert gates.get("final_verification") is True
    assert any(v.gate == "final_verification" and not v.passed
               for v in runner.task_state.verification)


def test_problems_and_corrections_from_test_cycle(tmp_path):
    from app.agent.context.error_checklist import ErrorChecklist
    from app.agent.context.operational_memory import OperationalMemory
    from app.llm.models import LLMResponse, Usage

    class _StubLLM:
        def generate(self, messages, tools=None, component=None,
                     iteration=None, attempt=1):
            import json as _json
            if component == "ErrorChecklist":
                content = _json.dumps({"items": ["test_x failed"]})
            else:
                content = "resumo"
            return LLMResponse(content=content, tool_calls=[],
                               usage=Usage(1, 1, 2), provider="stub")

    class _FlakyTools(T.FakeToolRegistry):
        def __init__(self):
            self.n = 0

        def execute(self, tool, arguments):
            # Só run_command conta (list_files da memória passa aqui).
            if tool != "run_command":
                return super().execute(tool, arguments)
            self.n += 1
            if self.n == 1:
                return ("STATUS: failure (exit code 1)\n"
                        "FAILED test_x")
            return "STATUS: success (exit code 0)\n3 passed"

    checklist = ErrorChecklist(_StubLLM())
    fail = _task("run_command", {"project_name": "p",
                                 "command": "python -m pytest -q"})
    fix = _task("write_file", {"project_name": "p",
                               "file_path": "fix.py", "content": "x\n"})
    ok_task = _task("run_command", {"project_name": "p",
                                    "command": "python -m pytest -q"})
    tools = _FlakyTools()
    runner = Runner(
        planner=T.FakePlanner(
            [Decision(action=DecisionAction.TASK, task=fail),
             Decision(action=DecisionAction.TASK, task=fix),
             Decision(action=DecisionAction.TASK, task=ok_task),
             _finish()]),
        task_decision_maker=T.FakeTaskDecisionMaker(
            [_exec_for(fail), _exec_for(fix), _exec_for(ok_task)]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=checklist,
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    assert runner.run(objective="obj", project_name="p") == "done"
    state = runner.task_state
    # Fase 5: problema vem do Analyzer (fallback determinístico), não
    # mais do espelho do checklist; correção do fix + nota do verde.
    assert len(state.problems) == 1
    assert "test_x" in state.problems[0].description
    assert state.problems[0].status == "resolved"
    assert state.open_problems == []  # resolvido pelo teste verde
    assert len(state.corrections) == 2
    assert state.corrections[0].problem_id == (
        state.problems[0].problem_id)
    assert "pytest" in state.corrections[1].description


def test_trace_init_event_is_valid_json(tmp_path):
    trace = ExecutionTrace(trace_dir=str(tmp_path / "t"))
    runner = _runner([_finish()], [], trace)
    runner.run(objective="obj", project_name="p")
    by_type = {}
    for event in trace.read_events():
        by_type.setdefault(event["event"], []).append(event)
    assert "task_state_init" in by_type
    snapshot = by_type["task_state_init"][0]["state"]
    assert snapshot["original_prompt"] == "obj"
    clone = TaskState.from_dict(json.loads(json.dumps(snapshot)))
    assert clone.objective == "obj"


def test_no_extra_llm_calls_for_english_run(tmp_path):
    from app.agent.planning.decision_parser import DecisionParser
    from app.agent.planning.planner import Planner
    from app.llm.client import LLMClient
    from app.tools.registry import ToolRegistry

    tools = ToolRegistry()
    tools.load_defaults()

    class _CountingProvider:
        def __init__(self):
            self.calls = 0
            self.name = "counting"

        def generate(self, messages, tools=None):
            import json as _json
            self.calls += 1
            return LLMResponse(
                content=_json.dumps(
                    {"action": "finish", "content": "done"}),
                tool_calls=[], usage=Usage(1, 1, 2), provider="counting")

    provider = _CountingProvider()
    planner = Planner(llm=LLMClient(provider),
                      parser=DecisionParser(tools=tools), tools=tools)
    runner = Runner(
        planner=planner,
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
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    # EN: só o finish (1 call, FakeChecklist não chama LLM);
    # canonicalização = 0 calls (qualquer chamada extra apareceria aqui).
    assert runner.run(objective="Create a notes app",
                      project_name="p") == "done"
    assert provider.calls == 1
    assert runner.task_state.translation_applied is False


class _OnceProblems(T.FakeFinalVerification):
    def __init__(self):
        super().__init__(result=FinalVerificationResult(
            FinalVerificationResult.OK))
        self.n = 0

    def verify(self, objective, project_name, summary, tools_execute):
        self.n += 1
        if self.n == 1:
            return FinalVerificationResult(
                FinalVerificationResult.PROBLEMS_FOUND, "ruim")
        return FinalVerificationResult(FinalVerificationResult.OK)
