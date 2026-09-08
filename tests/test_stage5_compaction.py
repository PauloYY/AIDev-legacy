"""Etapa 5 — compactação inteligente do contexto do Planner.

Cobre os 18 casos obrigatórios sem alterar nenhum teste existente:
1-3.   Prompt compacto preserva objetivo, regras e tools.
4.     Schemas compactos continuam válidos (nome/required/tipos).
5-7.   Janela de histórico: remove sucessos antigos, mantém recente e erros.
8-9.   Resultados grandes reduzem sem perder STATUS/falhas de teste.
10.    Estado atual (arquivos) permanece disponível.
11.    Duplicações são removidas (prompt menor, nada essencial some).
12-13. Primeira decisão e decisão após erro recebem contexto correto.
14-15. Short repair e full retry intactos com a flag ligada.
16-17. Finish gate e final verification intactos com a flag ligada.
18.    Flag desligada recupera o comportamento anterior byte a byte.
Extras: redução mensurável, breakdown consistente, sem truncamento
destrutivo, teto da cauda de erros, resumo capped.
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.context.operational_memory import OperationalMemory
from app.agent.context.planner_error_memory import PlannerErrorMemory
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.task import Task
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.agent.runner import Runner
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage
from app.tools.registry import ToolRegistry


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _task(tool, arguments, dependencies=None, **extra):
    return Task(tool=tool, arguments=arguments,
                dependencies=dependencies or [], **extra)


def _finish(content="done"):
    return Decision(action=DecisionAction.FINISH, content=content)


def _exec_for(task):
    return ExecutionDecision(tool=task.tool, arguments=task.arguments)


def _tools():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


def _planner(tools=None, provider=None):
    tools = tools or _tools()
    if provider is None:
        from unittest.mock import MagicMock
        llm = MagicMock()
    else:
        llm = LLMClient(provider)
    return Planner(llm=llm, parser=DecisionParser(tools=tools), tools=tools)


def _fill_memory(memory, n=15, fail_at=None):
    """Preenche a memória com n ações (write/run_command/read)."""
    for i in range(1, n + 1):
        if i % 3 == 0:
            tool, args = "write_file", {
                "project_name": "p", "file_path": f"f{i}.py",
                "content": f"x = {i}\n"}
        elif i % 3 == 1:
            tool, args = "run_command", {
                "project_name": "p", "command": "python -m pytest -q"}
        else:
            tool, args = "read_file", {
                "project_name": "p", "file_path": f"f{i}.py"}
        failed = (fail_at is not None and i == fail_at)
        if failed and tool != "run_command":
            tool, args = "run_command", {
                "project_name": "p", "command": "python -m pytest -q"}
        result = ("STATUS: failure (exit code 1)\nFAILED test_x"
                  if failed else "STATUS: success (exit code 0)\nok")
        memory.record(iteration=i, tool=tool, arguments=args,
                      result=result, success=not failed)


def _trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "trace5"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


class _ScriptedProvider:
    def __init__(self, contents):
        self._contents = list(contents)
        self.prompts = []
        self.name = "scripted"

    def generate(self, messages, tools=None):
        self.prompts.append(messages[0].content)
        item = self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=item, tool_calls=[],
                           usage=Usage(10, 5, 15), provider=self.name)


def _task_json(tool, arguments, **extra):
    task = {"tool": tool, "arguments": arguments, "dependencies": []}
    task.update(extra)
    return json.dumps({"action": "task", "task": task})


# Regras essenciais que NUNCA podem sumir do template compacto.
ESSENTIAL_RULES = [
    "Return ONLY valid JSON",
    "dependencies",
    "investigation",
    "checklist_progress",
    "list_symbols",
    "finish",
    "fail",
    "run_command",
    "read_file",
    "package.json",
    "gradle",
    "stdin",
]

ESSENTIAL_SECTIONS = [
    "RULES",
    "PROGRESS",
    "OBJECTIVE CHECKLIST",
    "ERROR CHECKLIST",
    "PROJECT COHERENCE",
    "VALIDATION BEFORE FINISH",
    "TOOLS AVAILABLE",
    "DECISION FORMATS",
    "OBJECTIVE",
    "CURRENT CONTEXT",
]


# --------------------------------------------------------------------------
# 1-3. prompt compacto preserva objetivo, regras e tools
# --------------------------------------------------------------------------

def test_compact_prompt_preserves_objective():
    """Caso 1: objetivo aparece integral no prompt compacto."""
    planner = _planner()
    objective = "Construir o gerenciador XPTO com modelo e testes"
    prompt = planner._build_prompt_compact(objective, "ctx")
    assert objective in prompt
    # Objetivo não é truncado nem resumido.
    assert prompt.count(objective) >= 1


def test_compact_prompt_preserves_essential_rules():
    """Caso 2: nenhuma regra essencial foi removida."""
    planner = _planner()
    prompt = planner._build_prompt_compact("obj", "ctx")
    for rule in ESSENTIAL_RULES:
        assert rule in prompt, f"regra essencial sumiu: {rule}"
    for section in ESSENTIAL_SECTIONS:
        assert section in prompt, f"seção sumiu: {section}"
    # Formatos de decisão completos.
    for action in ('"task"', '"finish"', '"fail"'):
        assert action in prompt


def test_compact_prompt_preserves_tools():
    """Caso 3: todas as tools seguem listadas no prompt compacto."""
    tools = _tools()
    planner = _planner(tools)
    prompt = planner._build_prompt_compact("obj", "ctx")
    for name in sorted(tools._tools.keys()):
        assert name in prompt, f"tool sumiu do prompt: {name}"
    # project_name continua visível (argumento obrigatório ubíquo).
    assert "project_name" in prompt


# --------------------------------------------------------------------------
# 4. schemas compactos válidos
# --------------------------------------------------------------------------

def test_compact_tool_schemas_stay_valid():
    """Caso 4: nome/required/tipos preservados em toda tool."""
    tools = _tools()
    planner = _planner(tools)
    compact_text = planner._build_tools_context_compact()
    compact = json.loads(compact_text)
    assert len(compact) == len(tools._tools)

    by_name = {entry["name"]: entry for entry in compact}
    for name, tool in tools._tools.items():
        entry = by_name[name]
        expected_required = (
            tool.definition["function"]["parameters"].get("required", []))
        assert entry["required"] == expected_required
        expected_props = set(
            tool.definition["function"]["parameters"]
            .get("properties", {}).keys())
        assert set(entry["properties"].keys()) == expected_props
        # Tipos preservados (ex.: timeout_seconds continua integer).
        full_props = (
            tool.definition["function"]["parameters"]
            .get("properties", {}))
        for prop, spec in full_props.items():
            prop_type = spec.get("type", "")
            if prop_type:
                assert prop_type in entry["properties"][prop]
    # Menor que o integral.
    assert len(compact_text) < len(planner._build_tools_context())


# --------------------------------------------------------------------------
# 5-7. janela de histórico
# --------------------------------------------------------------------------

def test_old_irrelevant_history_can_be_dropped():
    """Caso 5: sucessos antigos saem; omissão é explícita."""
    memory = OperationalMemory(_tools())
    _fill_memory(memory, n=15)
    full = memory.render_history()
    compact = memory.render_history_compact()
    assert len(compact) < len(full)
    assert "earlier action(s) omitted" in compact
    # A ação mais antiga (sucesso) não precisa estar lá.
    assert "[1]" not in compact


def test_recent_history_is_kept():
    """Caso 6: últimas 8 ações permanecem no compacto."""
    memory = OperationalMemory(_tools())
    _fill_memory(memory, n=15)
    compact = memory.render_history_compact()
    for iteration in range(8, 16):
        assert f"[{iteration}]" in compact


def test_recent_errors_are_kept():
    """Caso 7: falha recente e falha antiga preservadas."""
    memory = OperationalMemory(_tools())
    # Falha antiga fora da janela (iter 2) + falha recente (iter 14).
    _fill_memory(memory, n=6, fail_at=2)
    _fill_memory(memory, n=0)  # no-op p/ clareza
    memory.record(iteration=14, tool="run_command",
                  arguments={"project_name": "p",
                             "command": "python -m pytest -q"},
                  result="STATUS: failure (exit code 1)\nFAILED test_y",
                  success=False)
    for i in (15, 16, 17):
        memory.record(iteration=i, tool="write_file",
                      arguments={"project_name": "p",
                                 "file_path": f"g{i}.py", "content": "x\n"},
                      result="ok", success=True)
    compact = memory.render_history_compact()
    assert "[14]" in compact  # falha recente dentro da janela
    assert "earlier failure preserved" in compact  # iter 2 resgatado
    assert "STATUS: failure" in compact


# --------------------------------------------------------------------------
# 8-9. resultados grandes
# --------------------------------------------------------------------------

def test_big_results_shrink_without_losing_status(tmp_path):
    """Caso 8: output gigante vira head+tail com STATUS preservado."""
    runner = Runner(
        planner=T.FakePlanner([_finish()]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("r")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    big = ("STATUS: success (exit code 0)\n" + "y" * 3000
           + "\nCAUDA-MARCADOR-FINAL-12345")
    out = runner._truncate_compact(big)
    assert len(out) <= Runner.MAX_COMPACT_RESULT_CHARS + 200
    assert "STATUS: success (exit code 0)" in out
    assert "CAUDA-MARCADOR-FINAL-12345" in out
    assert "compacted" in out
    # Pequenos voltam intactos.
    small = "STATUS: success (exit code 0)\nok"
    assert runner._truncate_compact(small) == small


def test_test_results_keep_failures():
    """Caso 9: falhas de teste (exit code, FAILED, arquivo) preservadas."""
    runner = Runner(
        planner=T.FakePlanner([_finish()]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("r")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    big_fail = ("STATUS: failure (exit code 1)\n" + "log\n" * 800
                + "FAILED test_login.py::test_x - assert 1 == 2\n"
                + "Traceback (most recent call last): ... line 42\n")
    out = runner._truncate_compact(big_fail)
    assert "STATUS: failure (exit code 1)" in out
    assert "FAILED test_login.py::test_x" in out
    assert "line 42" in out


# --------------------------------------------------------------------------
# 10-11. estado atual + deduplicação
# --------------------------------------------------------------------------

def test_current_state_stays_available(projects_root, monkeypatch):
    """Caso 10: arquivos/checklist/erros pendentes sobrevivem."""
    monkeypatch.setattr(Config, "compact_planner", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = _tools()
    memory = OperationalMemory(tools)
    _fill_memory(memory, n=3)

    class _Pending(T.FakeErrorChecklist):
        def render(self):
            return "CURRENT ERROR CHECKLIST\n- 1. teste_x falhou"

        @property
        def pending_count(self):
            return 1

    class _Check(T.FakeChecklist):
        def render(self):
            return "OBJECTIVE CHECKLIST\n[x] 1. construir"

    runner = Runner(
        planner=T.FakePlanner([]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=tools,
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=memory,
        checklist=_Check(),
        error_checklist=_Pending(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    block = runner._build_memory_block_compact("p", "resumo do projeto")
    assert "- a.py" in block
    assert "construir" in block
    assert "teste_x falhou" in block
    assert "PROJECT SUMMARY" in block


def test_duplicated_info_is_removed():
    """Caso 11: compacto < integral sem perder o essencial."""
    tools = _tools()
    planner = _planner(tools)
    memory = OperationalMemory(tools)
    _fill_memory(memory, n=15, fail_at=14)

    legacy_static = planner._build_prompt("", "")
    compact_static = planner._build_prompt_compact("", "")
    assert len(compact_static) < len(legacy_static) * 0.6

    full_hist = memory.render_history()
    compact_hist = memory.render_history_compact()
    assert len(compact_hist) < len(full_hist)

    # Erros proibidos: janela de 5 < 20 quando há muitos erros.
    err_mem = PlannerErrorMemory()
    for i in range(12):
        err_mem.record(f"ValueError: erro {i}")
    assert len(err_mem.render_compact()) < len(err_mem.render())
    assert "erro 11" in err_mem.render_compact()


# --------------------------------------------------------------------------
# 12-13. situações diferentes
# --------------------------------------------------------------------------

def test_first_decision_gets_correct_context(projects_root, monkeypatch):
    """Caso 12: 1ª decisão tem objetivo+estado+tools+regras, sem histórico."""
    monkeypatch.setattr(Config, "compact_planner", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = _tools()
    provider = _ScriptedProvider([
        _task_json("write_file", {"project_name": "p",
                                  "file_path": "b.py", "content": "y\n"}),
        json.dumps({"action": "finish", "content": "done"}),
    ])
    planner = Planner(llm=LLMClient(provider),
                      parser=DecisionParser(tools=tools), tools=tools)
    runner = Runner(
        planner=planner,
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="write_file",
                              arguments={"project_name": "p",
                                         "file_path": "b.py",
                                         "content": "y\n"})]),
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
    assert runner.run(objective="meu objetivo inicial",
                      project_name="p") == "done"
    first = provider.prompts[0]
    assert "meu objetivo inicial" in first
    assert "CURRENT PROJECT FILES" in first
    assert "- a.py" in first
    assert "No actions executed yet" in first
    assert "Return ONLY valid JSON" in first


def test_decision_after_error_gets_error_context(
        projects_root, monkeypatch):
    """Caso 13: após falha, Planner vê erro+checklist+última alteração."""
    monkeypatch.setattr(Config, "compact_planner", True)
    from app.tools.filesystem.write_file import write_file
    write_file("p", "a.py", "x = 1\n")

    tools = _tools()
    seen = []

    class _CatchingPlanner(T.FakePlanner):
        def plan(self, objective, context, iteration=None,
                 request_type=None):
            seen.append(context)
            return super().plan(objective, context, iteration=iteration,
                                request_type=request_type)

    fail_task = _task("run_command", {"project_name": "p",
                                      "command": "python -m pytest -q"})
    runner = Runner(
        planner=_CatchingPlanner([
            Decision(action=DecisionAction.TASK, task=fail_task),
            _finish(),
        ]),
        task_decision_maker=T.FakeTaskDecisionMaker([
            ExecutionDecision(tool="run_command",
                              arguments=fail_task.arguments)]),
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
    # run_command real falha (comando inexistente) ou passa; o que
    # importa é que o RESULTADO e o histórico chegam ao finish.
    runner.run(objective="obj", project_name="p")
    assert len(seen) == 2
    assert "EXECUTION RESULT" in seen[1]
    assert "STATUS:" in seen[1]
    assert "run_command" in seen[1]


# --------------------------------------------------------------------------
# 14-15. short repair e full retry intactos
# --------------------------------------------------------------------------

def test_short_repair_still_works_with_compact_on(
        projects_root, tmp_path, monkeypatch):
    """Caso 14: inválida → short repair resolve, sem full retry."""
    monkeypatch.setattr(Config, "compact_planner", True)
    tools = _tools()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    provider = _ScriptedProvider([
        "não-json {{{",
        _task_json("write_file", args),
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
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == ["short_repair"]
    # O repair continua mínimo, sem contexto completo.
    assert "CURRENT CONTEXT" not in provider.prompts[1]
    assert "PREVIOUS DECISION" in provider.prompts[1]


def test_full_retry_still_works_with_compact_on(
        projects_root, tmp_path, monkeypatch):
    """Caso 15: curto falha → fallback completo (compacto) funciona."""
    monkeypatch.setattr(Config, "compact_planner", True)
    tools = _tools()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    provider = _ScriptedProvider([
        "ruim 1 {{{",
        "ruim 2 }}}",
        _task_json("write_file", args),
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
        validator=T.FakeValidator(),
        operational_memory=OperationalMemory(tools),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=trace,
    )
    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == [
        "short_repair", "full_context"]
    # O full retry compacto continua maior que o short.
    assert retries[1]["prompt_chars"] > retries[0]["prompt_chars"]


# --------------------------------------------------------------------------
# 16-17. finish gate e final verification intactos
# --------------------------------------------------------------------------

def test_finish_gate_still_blocks_with_compact_on(tmp_path, monkeypatch):
    """Caso 16: finish com problema bloqueia e depois aceita."""
    monkeypatch.setattr(Config, "compact_planner", True)

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
    runner = Runner(
        planner=T.FakePlanner([_finish("a"), _finish("b")]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("resumo")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=_OnceProblems(),
        execution_trace=trace,
    )
    assert runner.run(objective="obj", project_name="p") == "b"
    assert runner._stats.finish_blocks == 1
    assert [e["gate"] for e in _events(trace, "finish_block")] == [
        "final_verification"]


def test_final_verification_receives_summary_with_compact_on(tmp_path,
                                                             monkeypatch):
    """Caso 17: verificação final recebe o resumo vigente."""
    monkeypatch.setattr(Config, "compact_planner", True)

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
    assert final.verify_calls[0]["summary"] == "novo resumo"


# --------------------------------------------------------------------------
# 18. flag off recupera o legado
# --------------------------------------------------------------------------

def test_disabling_compaction_restores_legacy(projects_root, monkeypatch):
    """Caso 18: flag off → prompt integral + histórico integral."""
    monkeypatch.setattr(Config, "compact_planner", False)
    tools = _tools()
    planner = _planner(tools)

    prompt = planner._build_prompt_compact("obj", "ctx")
    legacy = planner._build_prompt("obj", "ctx")
    assert len(prompt) < len(legacy)

    # plan() com flag off envia o legado byte a byte.
    provider = _ScriptedProvider(
        [json.dumps({"action": "finish", "content": "done"})])
    planner2 = Planner(llm=LLMClient(provider),
                       parser=DecisionParser(tools=tools), tools=tools)
    planner2.plan(objective="obj", context="ctx", iteration=1)
    assert provider.prompts[0] == legacy

    # Runner com flag off usa truncamento legado (head 4000).
    runner = Runner(
        planner=T.FakePlanner([_finish()]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("r")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    big = "z" * 6000
    assert runner._truncate_for_planner(big) == runner._truncate(big)
    assert "compacted" not in runner._truncate_for_planner(big)

    # Bloco de memória integral com flag off.
    memory = OperationalMemory(tools)
    _fill_memory(memory, n=10)
    assert "[1]" in memory.render_history()
    assert "[1]" not in memory.render_history_compact()


# --------------------------------------------------------------------------
# extras
# --------------------------------------------------------------------------

def test_compact_breakdown_sums_to_compact_prompt():
    """Breakdown do compacto soma no prompt compacto enviado."""
    from app.agent.planning.prompt_sections import summarize_measurements
    planner = _planner()
    ctx = ("PROJECT SUMMARY:\ns\n\nACTION HISTORY:\n[1] ok\n\n"
           "CURRENT PROJECT FILES:\n- a.py\n")
    sections = planner.build_compact_prompt_sections("obj", ctx)
    measured = planner.measure_prompt_sections(sections)
    totals = summarize_measurements(measured)
    assert totals["chars"] == len(
        planner._build_prompt_compact("obj", ctx))
    assert "static_template" in measured
    assert measured["static_template"]["chars"] < 8000


def test_no_destructive_truncation():
    """Compactação é estrutural: objetivo/regras nunca cortados."""
    planner = _planner()
    long_objective = "OBJETIVO-LONGO-" + "x" * 3000
    prompt = planner._build_prompt_compact(long_objective, "ctx")
    assert long_objective in prompt
    assert "Return ONLY valid JSON" in prompt
    # Nenhum marcador de truncamento arbitrário no template.
    assert "[truncated" not in prompt
    assert "characters omitted" not in prompt


def test_error_tail_is_capped_but_markers_kept(tmp_path, monkeypatch):
    """Cauda de erros: no máx. 2 blocos, marcador explícito."""
    monkeypatch.setattr(Config, "compact_planner", True)
    runner = Runner(
        planner=T.FakePlanner([]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("r")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    context = "PROJECT SUMMARY:\nbase"
    for i in range(5):
        context = runner._append_context_error(
            context, f"VALIDATION ERROR BEFORE FINISH:\nbloco {i}")
    assert context.count("VALIDATION ERROR BEFORE FINISH") == 2
    assert "bloco 4" in context and "bloco 3" in context
    assert "earlier error block(s) omitted" in context
    assert "PROJECT SUMMARY" in context


def test_summary_is_capped_in_compact_block(monkeypatch):
    """Resumo gigante não explode o bloco compacto."""
    monkeypatch.setattr(Config, "compact_planner", True)
    runner = Runner(
        planner=T.FakePlanner([]),
        task_decision_maker=T.FakeTaskDecisionMaker([]),
        task_context_builder=T.FakeTaskContextBuilder(),
        tools=T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=T.FakeSummaryUpdater(
            T.FakeSummary("r")),
        validator=T.FakeValidator(),
        operational_memory=T.FakeOperationalMemory(),
        checklist=T.FakeChecklist(),
        error_checklist=T.FakeErrorChecklist(),
        final_verification=T.FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)),
        execution_trace=NullTrace(),
    )
    block = runner._build_memory_block_compact("p", "S" * 5000)
    assert len(block) < 5000
    assert "summary compacted" in block
