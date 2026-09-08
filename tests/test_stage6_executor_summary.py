"""Etapa 6 — eficiência do Executor + SummaryUpdater.

Cobre os 18 casos obrigatórios sem alterar nenhum teste existente
(um teste da Etapa 4 pinou compact_executor=False para preservar as
asserções originais; aqui a política nova é verificada):
1-4.   Executor compacto preserva decisão/args/dependências/restrições.
5.     Sem schemas no Executor; resultados de dependency com cap seguro.
6-8.   Updater recebe contexto suficiente; skips permitidos; estado
       preservado de forma determinística (pular = manter, nunca fabrica).
9-11.  Erros, resultados de teste e exit codes continuam disponíveis.
12-13. Duplicação reduzida; outputs grandes compactados com marcador.
14-15. Short repair e full retry intactos com a flag ligada.
16-17. Finish gate e final verification intactos com a flag ligada.
18.    Flag desligada recupera o comportamento da Etapa 5.
"""

import json

import pytest

import tests.test_runner as T
from app.agent.context.final_verification import FinalVerificationResult
from app.agent.context.operational_memory import OperationalMemory
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.execution_decision_parser import (
    ExecutionDecisionParser,
)
from app.agent.execution.task import Task
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.planning.dependency import Dependency
from app.agent.runner import Runner
from app.agent.trace import ExecutionTrace, NullTrace
from app.config import Config
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, Usage
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


def _tools():
    registry = ToolRegistry()
    registry.load_defaults()
    return registry


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


def _exec_provider(decisions):
    return _ScriptedProvider(decisions)


def _write_task_json(tool, arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def _trace(tmp_path):
    return ExecutionTrace(trace_dir=str(tmp_path / "trace6"))


def _events(trace, name):
    return [e for e in trace.read_events() if e["event"] == name]


def _runner(decisions, executions, trace, **overrides):
    return Runner(
        planner=T.FakePlanner(decisions),
        task_decision_maker=T.FakeTaskDecisionMaker(executions),
        task_context_builder=overrides.pop(
            "task_context_builder", None) or T.FakeTaskContextBuilder(),
        tools=overrides.pop("tools", None) or T.FakeToolRegistry(),
        project_context=T.FakeProjectContext(),
        project_summary_updater=overrides.pop(
            "updater", None) or T.FakeSummaryUpdater(
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
        **overrides,
    )


# Restrições normativas do Executor que nunca podem sumir.
EXECUTOR_CONSTRAINTS = [
    "ONLY valid JSON",
    "Do NOT execute tools",
    "tool calls",
    "exclusively the indicated tool",
    "do not create tasks",
    "dependencies",
    "purpose",
    "full file content",
]


# --------------------------------------------------------------------------
# 1-4. executor compacto preserva decisão
# --------------------------------------------------------------------------

def test_compact_executor_preserves_decision():
    """Caso 1: tool + objetivo vão integrais no prompt compacto."""
    task = _task("write_file", {"project_name": "p",
                                "file_path": "svc.py",
                                "content": "implementar Client"})
    prompt = TaskDecisionMaker._build_prompt_compact(
        "construir o gerenciador XPTO", task, "DEPENDÊNCIAS:\nNenhuma.")
    assert "write_file" in prompt
    assert "construir o gerenciador XPTO" in prompt
    assert "svc.py" in prompt


def test_compact_executor_preserves_arguments():
    """Caso 2: argumentos completos (incl. conteúdo) preservados."""
    content = "brief description with details id, nome, status"
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py",
                                "content": content})
    prompt = TaskDecisionMaker._build_prompt_compact("obj", task, "ctx")
    assert content in prompt
    assert "project_name" in prompt


def test_compact_executor_preserves_dependencies():
    """Caso 3: contexto das dependencies vai integral ao Executor."""
    dep_result = "conteudo real do arquivo lido linha1\nlinha2"
    context = ("TASK PAI:\nTool: write_file\n\nDEPENDÊNCIAS:\n"
               f"DEPENDÊNCIA 1:\nRESULTADO:\n{dep_result}")
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    prompt = TaskDecisionMaker._build_prompt_compact("obj", task, context)
    assert dep_result in prompt
    assert "DEPENDENCY CONTEXT" in prompt


def test_compact_executor_preserves_constraints():
    """Caso 4: todas as restrições normativas mantidas."""
    task = _task("read_file", {"project_name": "p",
                               "file_path": "a.py"})
    prompt = TaskDecisionMaker._build_prompt_compact("obj", task, "ctx")
    for constraint in EXECUTOR_CONSTRAINTS:
        assert constraint in prompt, f"restrição sumiu: {constraint}"
    assert '"tool": "read_file"' in prompt


# --------------------------------------------------------------------------
# 5. sem schemas desnecessários; cap seguro
# --------------------------------------------------------------------------

def test_executor_needs_no_tool_schemas():
    """Caso 5: Executor não recebe schemas (só a tool da task)."""
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    for builder in (TaskDecisionMaker._build_prompt,
                    TaskDecisionMaker._build_prompt_compact):
        prompt = builder("obj", task, "ctx")
        assert "TOOLS AVAILABLE" not in prompt
        assert '"type": "function"' not in prompt
        assert '"required"' not in prompt


def test_dependency_results_capped_safely(monkeypatch):
    """Caso 5b: cap 2000 com flag on; 4000/None preservados sem ela."""
    dep = Dependency(tool="read_file",
                     arguments={"project_name": "p",
                                "file_path": "a.py"})
    task = _task("write_file", {"project_name": "p"},
                 dependencies=[dep])
    big = "STATUS: success (exit code 0)\n" + "y" * 6000

    monkeypatch.setattr(Config, "compact_executor", True)
    out = TaskContextBuilder(max_result_chars=4000).build(task, [big])
    assert "STATUS: success (exit code 0)" in out
    assert "truncated result" in out
    assert len(out) < 3000

    monkeypatch.setattr(Config, "compact_executor", False)
    out_legacy = TaskContextBuilder(
        max_result_chars=4000).build(task, [big])
    assert len(out_legacy) > len(out)
    assert len(out_legacy) < 5000

    # None = integral explícito, mesmo com a flag ligada.
    monkeypatch.setattr(Config, "compact_executor", True)
    assert big in TaskContextBuilder().build(task, [big])


# --------------------------------------------------------------------------
# 6-8. summary updater: contexto, skips, estado determinístico
# --------------------------------------------------------------------------

def test_updater_receives_enough_context():
    """Caso 6: objetivo+resumo+task+resultado chegam ao updater."""
    from app.agent.context.project_summary import ProjectSummary
    from app.agent.context.project_summary_updater import (
        ProjectSummaryUpdater)
    from unittest.mock import MagicMock
    updater = ProjectSummaryUpdater(llm=MagicMock(),
                                    summary=MagicMock())
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x" * 5000})
    prompt = updater._build_prompt(
        objective="meu objetivo",
        current_summary="resumo vigente aqui",
        task=task,
        result="STATUS: success (exit code 0)\nfeito",
    )
    assert "meu objetivo" in prompt
    assert "resumo vigente aqui" in prompt
    assert "a.py" in prompt
    assert "STATUS: success (exit code 0)" in prompt
    # Conteúdo gigante não vai integral (política da Etapa 2D).
    assert "x" * 5000 not in prompt


@pytest.mark.parametrize("tool,arguments,reason", [
    ("read_file", {"project_name": "p", "file_path": "a.py"},
     "read_only_no_state_change"),
    ("list_files", {"project_name": "p"},
     "read_only_no_state_change"),
    ("check_project", {"project_name": "p"},
     "analysis_no_state_change"),
    ("run_command", {"project_name": "p", "command": "python3 --version"},
     "inspect_no_state_change"),
    ("run_command", {"project_name": "p", "command": "ls"},
     "inspect_no_state_change"),
])
def test_updater_skipped_when_allowed(tmp_path, monkeypatch, tool,
                                      arguments, reason):
    """Caso 7: skips sem mudança de estado (Etapas 4+6)."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_executor", True)
    trace = _trace(tmp_path)
    updater = T.FakeSummaryUpdater(T.FakeSummary("resumo"))
    task = _task(tool, arguments)
    runner = _runner([Decision(action=DecisionAction.TASK, task=task),
                      _finish()],
                     [_exec_for(task)], trace, updater=updater,
                     operational_memory=OperationalMemory(
                         T.FakeToolRegistry()))
    assert runner.run(objective="obj", project_name="p") == "done"
    assert updater.calls == 0
    skipped = _events(trace, "summary_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == reason
    assert _events(trace, "summary_updated") == []


@pytest.mark.parametrize("tool,arguments", [
    ("write_file", {"project_name": "p", "file_path": "a.py",
                    "content": "x"}),
    ("run_command", {"project_name": "p",
                    "command": "cat > out.txt <<'EOF'\nhi\nEOF"}),
    ("run_command", {"project_name": "p",
                    "command": "python -m pytest -q"}),
    ("run_command", {"project_name": "p",
                    "command": "npm test"}),
])
def test_updater_still_called_for_state_changes(
        tmp_path, monkeypatch, tool, arguments):
    """Caso 8: mutação ou teste/build sempre atualizam (nada fabricado)."""
    monkeypatch.setattr(Config, "smart_summary", True)
    monkeypatch.setattr(Config, "compact_executor", True)
    trace = _trace(tmp_path)
    updater = T.FakeSummaryUpdater(T.FakeSummary("resumo"))
    task = _task(tool, arguments)
    runner = _runner([Decision(action=DecisionAction.TASK, task=task),
                      _finish()],
                     [_exec_for(task)], trace, updater=updater,
                     operational_memory=OperationalMemory(
                         T.FakeToolRegistry()))
    assert runner.run(objective="obj", project_name="p") == "done"
    assert updater.calls == 1
    assert len(_events(trace, "summary_updated")) == 1
    assert _events(trace, "summary_skipped") == []


# --------------------------------------------------------------------------
# 9-11. erros, testes e exit codes disponíveis
# --------------------------------------------------------------------------

def test_errors_tests_exit_codes_available(monkeypatch):
    """Casos 9-11: veredito integral no contexto do Executor."""
    monkeypatch.setattr(Config, "compact_executor", True)
    result = ("STATUS: failure (exit code 1)\n"
              "FAILED test_cart.py::test_total - assert 10 == 12\n"
              "test_cart.py:42 AssertionError")
    dep = Dependency(tool="run_command",
                     arguments={"project_name": "p",
                                "command": "python -m pytest -q"})
    task = _task("write_file", {"project_name": "p",
                                "file_path": "cart.py", "content": "fix"},
                 dependencies=[dep])
    out = TaskContextBuilder(max_result_chars=4000).build(task, [result])
    assert "STATUS: failure (exit code 1)" in out
    assert "FAILED test_cart.py::test_total" in out
    assert "test_cart.py:42" in out

    prompt = TaskDecisionMaker._build_prompt_compact("obj", task, out)
    assert "exit code 1" in prompt
    assert "FAILED test_cart.py::test_total" in prompt


# --------------------------------------------------------------------------
# 12-13. deduplicação + outputs grandes
# --------------------------------------------------------------------------

def test_duplicated_info_is_reduced():
    """Caso 12: compacto < integral sem perder decisão nem contexto."""
    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"},
                 dependencies=[Dependency(
                     tool="read_file",
                     arguments={"project_name": "p",
                                "file_path": "a.py"})])
    context = "TASK PAI:\nTool: write_file\nRESULTADO:\nconteudo"
    legacy = TaskDecisionMaker._build_prompt("obj", task, context)
    compact = TaskDecisionMaker._build_prompt_compact("obj", task, context)
    assert len(compact) < len(legacy)
    for keeper in ("write_file", "a.py", "conteudo", "obj",
                   "ONLY valid JSON"):
        assert keeper in compact


def test_big_outputs_compacted_with_marker(monkeypatch):
    """Caso 13: outputs grandes com marcador; pequenos intactos."""
    monkeypatch.setattr(Config, "compact_executor", True)
    builder = TaskContextBuilder(max_result_chars=4000)
    small = "STATUS: success (exit code 0)\nok"
    task = _task("write_file", {"project_name": "p"})
    assert builder._format_result(small) == small

    big = "STATUS: success (exit code 0)\n" + "z" * 10000
    out = builder._format_result(big)
    assert out.startswith(
        f"[truncated result: {len(big)} chars in total]")
    assert "STATUS: success (exit code 0)" in out
    assert "chars omitted" in out
    assert len(out) < 2500


# --------------------------------------------------------------------------
# 14-15. short repair e full retry com a flag ligada
# --------------------------------------------------------------------------

def _real_executor_runner(exec_provider_contents, updater_contents,
                          executions, trace, tools, monkeypatch=None):
    from app.agent.execution.validator import TaskValidator
    from app.agent.planning.decision_parser import DecisionParser
    from app.agent.planning.planner import Planner
    from app.agent.context.project_summary import ProjectSummary
    from app.agent.context.project_summary_updater import (
        ProjectSummaryUpdater)

    planner_provider, executor_provider = exec_provider_contents
    planner = Planner(llm=LLMClient(_ScriptedProvider(planner_provider)),
                      parser=DecisionParser(tools=tools), tools=tools)
    decision_maker = TaskDecisionMaker(
        llm=LLMClient(_ScriptedProvider(executor_provider)),
        parser=ExecutionDecisionParser())

    class _MemSummary:
        def __init__(self):
            self.store = {}

        def read(self, project_name):
            return self.store.get(project_name, "")

        def write(self, project_name, value):
            self.store[project_name] = value

    updater = ProjectSummaryUpdater(
        llm=LLMClient(_ScriptedProvider(updater_contents)),
        summary=_MemSummary())
    return Runner(
        planner=planner,
        task_decision_maker=decision_maker,
        task_context_builder=TaskContextBuilder(max_result_chars=4000),
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
    )


def _write_json(tool, arguments):
    return json.dumps({"action": "task",
                       "task": {"tool": tool, "arguments": arguments,
                                "dependencies": []}})


def test_short_repair_still_works_with_compact_executor(
        projects_root, tmp_path, monkeypatch):
    """Caso 14: short repair do Planner intacto com a flag ligada."""
    monkeypatch.setattr(Config, "compact_executor", True)
    tools = _tools()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    trace = _trace(tmp_path)
    runner = _real_executor_runner(
        ([  # planner: inválida -> short repair resolve
            "não-json {{{",
            _write_json("write_file", args),
            json.dumps({"action": "finish", "content": "done"}),
        ], [  # executor: decisão direta
            _write_task_json("write_file", args),
        ]),
        ["resumo novo"],
        [ExecutionDecision(tool="write_file", arguments=args)],
        trace, tools)

    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == ["short_repair"]


def test_full_retry_still_works_with_compact_executor(
        projects_root, tmp_path, monkeypatch):
    """Caso 15: fallback completo intacto com a flag ligada."""
    monkeypatch.setattr(Config, "compact_executor", True)
    tools = _tools()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    trace = _trace(tmp_path)
    runner = _real_executor_runner(
        ([  # planner: 2 inválidas -> short + full
            "ruim 1 {{{",
            "ruim 2 }}}",
            _write_json("write_file", args),
            json.dumps({"action": "finish", "content": "done"}),
        ], [
            _write_task_json("write_file", args),
        ]),
        ["resumo novo"],
        [ExecutionDecision(tool="write_file", arguments=args)],
        trace, tools)

    assert runner.run(objective="obj", project_name="p") == "done"
    retries = _events(trace, "planner_retry")
    assert [e["retry_type"] for e in retries] == [
        "short_repair", "full_context"]


def test_executor_retry_still_works_with_compact_prompt(
        projects_root, tmp_path, monkeypatch):
    """Executor com erro de decisão tenta de novo (prompt compacto)."""
    monkeypatch.setattr(Config, "compact_executor", True)
    tools = _tools()
    args = {"project_name": "p", "file_path": "a.py", "content": "x = 1\n"}
    trace = _trace(tmp_path)
    runner = _real_executor_runner(
        ([  # planner
            _write_json("write_file", args),
            json.dumps({"action": "finish", "content": "done"}),
        ], [  # executor: 1ª inválida (tool trocada), 2ª ok
            _write_task_json("read_file", {"project_name": "p",
                                           "file_path": "a.py"}),
            _write_task_json("write_file", args),
        ]),
        ["resumo novo"],
        [ExecutionDecision(tool="write_file", arguments=args)],
        trace, tools)

    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(_events(trace, "executor_error")) == 1


# --------------------------------------------------------------------------
# 16-17. finish gate e final verification com a flag ligada
# --------------------------------------------------------------------------

def test_finish_gate_still_blocks_with_compact_executor(
        tmp_path, monkeypatch):
    """Caso 16: finish com problema bloqueia e depois aceita."""
    monkeypatch.setattr(Config, "compact_executor", True)

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


def test_final_verification_receives_summary_with_compact_executor(
        tmp_path, monkeypatch):
    """Caso 17: verificação final recebe o resumo vigente."""
    monkeypatch.setattr(Config, "compact_executor", True)
    final = T.FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK))
    task = _task("write_file", {"project_name": "p", "file_path": "a.py",
                                "content": "x"})
    runner = _runner(
        [Decision(action=DecisionAction.TASK, task=task), _finish()],
        [_exec_for(task)], NullTrace(), final_verification=final)
    assert runner.run(objective="obj", project_name="p") == "done"
    assert len(final.verify_calls) == 1


# --------------------------------------------------------------------------
# 18. flag off recupera o comportamento da Etapa 5
# --------------------------------------------------------------------------

def test_disabling_compact_executor_restores_legacy(monkeypatch):
    """Caso 18: flag off = prompt integral + cap 4000 + updater p/ tudo."""
    monkeypatch.setattr(Config, "compact_executor", False)
    monkeypatch.setattr(Config, "smart_summary", True)

    task = _task("write_file", {"project_name": "p",
                                "file_path": "a.py", "content": "x"})
    # Prompt do Executor volta ao integral.
    assert (TaskDecisionMaker._build_prompt("obj", task, "ctx")
            != TaskDecisionMaker._build_prompt_compact("obj", task, "ctx"))

    # Builder volta ao cap 4000 da Etapa 4.
    big = "y" * 6000
    dep = Dependency(tool="read_file",
                     arguments={"project_name": "p",
                                "file_path": "a.py"})
    out = TaskContextBuilder(max_result_chars=4000).build(
        _task("write_file", {"project_name": "p"},
              dependencies=[dep]), [big])
    assert len(out) > 3000  # cap 4000, não 2000

    # Runner: check_project e inspect voltam a atualizar o resumo.
    runner = _runner([], [], NullTrace(),
                     operational_memory=OperationalMemory(
                         T.FakeToolRegistry()))
    assert runner._summary_skip_reason(
        "check_project", {"project_name": "p"}) is None
    assert runner._summary_skip_reason(
        "run_command",
        {"project_name": "p", "command": "ls"}) is None
    # Reads continuam pulando (Etapa 4 congelada).
    assert runner._summary_skip_reason(
        "read_file",
        {"project_name": "p", "file_path": "a.py"}) == (
            "read_only_no_state_change")


def test_legacy_executor_prompt_is_byte_identical():
    """Garantia: _build_prompt reproduz o template em inglês.

    Template congelado inline (a versão via `git show HEAD` quebrou
    quando o refactor foi commitado — o intent é o template, não o
    VCS). Atualizado na padronização linguística: o protocolo do
    Executor agora é inglês, e o teste continua garantindo que o
    template é determinístico byte a byte.
    """
    task = _task("write_file", {"a": 1})
    objective, context = "OBJ", "CTX"
    expected = """
You are the executor of a software development task.

Your role is to decide how to execute the given task,
using the information available in the context.

OBJECTIVE:
OBJ

TASK:
Tool: write_file

ARGUMENTS:
{'a': 1}

DEPENDENCY CONTEXT:
CTX

RULES:
- Return ONLY valid JSON.
- Do NOT execute tools.
- Do NOT produce tool calls.
- Use exclusively the tool indicated in the task.
- Do not create new tasks.
- Do not create dependencies.
- Do not change the purpose of the task.
- Arguments must be valid for the tool.
- For write_file, provide the full file content.
- For read_file, keep the arguments needed for reading.

FORMAT:
{
    "tool": "write_file",
    "arguments": {
        ...
    }
}

Return ONLY the JSON.
"""
    assert (TaskDecisionMaker._build_prompt(objective, task, context)
            == expected)
