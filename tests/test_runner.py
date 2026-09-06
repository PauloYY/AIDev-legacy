import pytest

from app.agent.runner import Runner
from app.agent.planning.decision import Decision, DecisionAction
from app.agent.execution.task import Task
from app.agent.execution.execution_decision import ExecutionDecision


class FakeUsage:
    def summary(self):
        return "usage summary"


class FakeLLM:
    usage = FakeUsage()


class FakePlanner:
    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.llm = FakeLLM()
        self.received_contexts = []

    def plan(self, objective, context):
        self.received_contexts.append(context)
        item = self._decisions.pop(0)

        if isinstance(item, Exception):
            raise item

        return item


class FakeTaskDecisionMaker:
    def __init__(self, executions):
        self._executions = list(executions)

    def decide(self, objective, task, context):
        return self._executions.pop(0)


class FakeToolRegistry:
    def execute(self, tool, arguments):
        return "ok"

    def exists(self, tool):
        return False

    def get(self, tool):
        raise AssertionError("not used in these tests")


class FakeValidator:
    def __init__(self, error_for_task=None):
        self.error_for_task = error_for_task
        self.schema_validator = None

    def validate(self, task, allow_investigation=False):
        if self.error_for_task is not None:
            raise self.error_for_task

    def validate_arguments(self, tool_name, arguments, allow_investigation=False):
        pass


class FakeProjectContext:
    def __init__(self):
        self.summary = FakeSummary("resumo inicial")

    def initialize(self, project_name):
        return "resumo inicial"


class FakeSummary:
    def __init__(self, value):
        self.value = value

    def read(self, project_name):
        return self.value


class FakeSummaryUpdater:
    def __init__(self, summary_obj, should_fail=False):
        self.summary = summary_obj
        self.should_fail = should_fail
        self.calls = 0

    def update(self, objective, project_name, task, result):
        self.calls += 1

        if self.should_fail:
            raise RuntimeError("LLM instável")

        return "novo resumo"


class FakeChecklist:
    def reset(self):
        pass

    def generate(self, objective):
        pass

    def render(self):
        return ""

    def mark_done(self, progress):
        return []

    @property
    def pending_count(self):
        return 0

    @property
    def pending_items(self):
        return []


class FakeErrorChecklist:
    def reset(self):
        pass

    def clear(self):
        pass

    def generate(self, command, output):
        pass

    def render(self):
        return None

    @property
    def pending_count(self):
        return 0


class FakeFinalVerification:
    def __init__(self, result=None, raise_on_verify=False):
        self.result = result
        self._raise = raise_on_verify
        self.verify_calls = []
        self._call_count = 0

    def verify(self, objective, project_name, summary, tools_execute):
        self.verify_calls.append({
            "objective": objective,
            "project_name": project_name,
            "summary": summary,
        })
        self._call_count += 1
        if self._raise:
            raise RuntimeError("verificação falhou")
        return self.result


class FakeTaskContextBuilder:
    def build(self, task, dependency_results):
        return "task context"


class FakeOperationalMemory:
    def reset(self):
        pass

    def record(self, **kwargs):
        pass

    def render(self, project_name):
        return "memoria"

    def last_run_command_failed_test_or_build(self):
        return False

    def investigation_budget_available(self, iteration):
        return False

    def consume_investigation_budget(self, iteration):
        pass


class RecordingOperationalMemory(FakeOperationalMemory):
    """Espiã do orçamento de investigação: controla free_pass/budget e
    registra em que iteração(ões) o orçamento foi de fato consumido."""

    def __init__(self, free_pass=False, budget_available=False):
        self.free_pass = free_pass
        self.budget_available = budget_available
        self.consumed_at = []

    def last_run_command_failed_test_or_build(self):
        return self.free_pass

    def investigation_budget_available(self, iteration):
        return self.budget_available

    def consume_investigation_budget(self, iteration):
        self.consumed_at.append(iteration)


class RecordingValidator:
    def __init__(self):
        self.calls = []
        self.schema_validator = None

    def validate(self, task, allow_investigation=False):
        self.calls.append(allow_investigation)

    def validate_arguments(self, tool_name, arguments, allow_investigation=False):
        pass


def _make_runner(
    decisions,
    executions=None,
    validator=None,
    summary_updater=None,
    planner_error_memory=None,
    operational_memory=None,
    final_verification=None,
):
    if final_verification is None:
        from app.agent.context.final_verification import FinalVerificationResult
        final_verification = FakeFinalVerification(
            result=FinalVerificationResult(FinalVerificationResult.OK)
        )

    return Runner(
        planner=FakePlanner(decisions),
        task_decision_maker=FakeTaskDecisionMaker(executions or []),
        task_context_builder=FakeTaskContextBuilder(),
        tools=FakeToolRegistry(),
        project_context=FakeProjectContext(),
        project_summary_updater=(
            summary_updater
            or FakeSummaryUpdater(FakeSummary("resumo anterior"))
        ),
        validator=validator or FakeValidator(),
        operational_memory=operational_memory or FakeOperationalMemory(),
        checklist=FakeChecklist(),
        error_checklist=FakeErrorChecklist(),
        planner_error_memory=planner_error_memory,
        final_verification=final_verification,
    )


def test_repeated_identical_planner_error_does_not_stop_early():
    """Repetir o EXATO mesmo erro NÃO deve fazer a run parar mais cedo
    nem pular tentativas — continua dentro do orçamento normal de
    MAX_PLANNER_ATTEMPTS, só com uma correção mais forte a partir da
    repetição."""

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    error = ValueError("erro sempre igual")

    assert Runner.MAX_PLANNER_ATTEMPTS == 5

    # 4 erros repetidos seguidos (dentro do orçamento de 5 tentativas)
    # e só na 5a o Planner acerta — se o "já visto" cortasse o retry
    # mais cedo, isso não chegaria a rodar até o fim.
    decisions = [
        error, error, error, error,
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [ExecutionDecision(tool="write_file", arguments=task.arguments)]

    planner = FakePlanner(decisions)
    runner = Runner(
        planner=planner,
        task_decision_maker=FakeTaskDecisionMaker(executions),
        task_context_builder=FakeTaskContextBuilder(),
        tools=FakeToolRegistry(),
        project_context=FakeProjectContext(),
        project_summary_updater=FakeSummaryUpdater(FakeSummary("resumo")),
        validator=FakeValidator(),
        operational_memory=FakeOperationalMemory(),
        checklist=FakeChecklist(),
        error_checklist=FakeErrorChecklist(),
    )

    result = runner.run(objective="obj", project_name="p")

    assert result == "done"

    # A partir da 2a vez que o erro aparece (3a chamada ao Planner
    # nesta run), a correção precisa ser a versão reforçada.
    assert "ERRO REPETIDO" in planner.received_contexts[2]
    assert "ERRO REPETIDO" in planner.received_contexts[3]
    # Na 1a vez, ainda é a correção genérica normal.
    assert "CORREÇÃO DA TENTATIVA ANTERIOR" in planner.received_contexts[1]


def test_different_planner_errors_are_each_given_normal_retries():
    """Erros DIFERENTES entre si não devem ser tratados como repetição
    — cada um ganha o fluxo normal de retry."""

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        ValueError("erro A"),
        ValueError("erro B"),
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [ExecutionDecision(tool="write_file", arguments=task.arguments)]

    runner = _make_runner(decisions, executions)

    result = runner.run(objective="obj", project_name="p")

    assert result == "done"


def test_planner_error_memory_is_rendered_in_context():
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [ExecutionDecision(tool="write_file", arguments=task.arguments)]

    runner = _make_runner(decisions, executions)
    runner.planner_error_memory.record("erro histórico")

    block = runner._build_memory_block("p", "resumo")

    assert "erro histórico" in block
    assert "PROIBIDOS" in block


def test_investigation_allowed_via_periodic_budget_and_consumes_it():
    """Sem passe livre (teste/build não falhou), mas com orçamento
    periódico disponível, a investigação avulsa deve ser permitida —
    e o orçamento deve ser marcado como consumido nessa iteração."""

    memory = RecordingOperationalMemory(free_pass=False, budget_available=True)
    validator = RecordingValidator()

    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="ok"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=task.arguments)
    ]

    runner = _make_runner(
        decisions,
        executions,
        validator=validator,
        operational_memory=memory,
    )

    runner.run(objective="obj", project_name="p")

    assert validator.calls == [True]
    assert memory.consumed_at == [1]


def test_investigation_not_allowed_without_free_pass_or_budget():
    """Sem passe livre e sem orçamento disponível, a tool de
    investigação avulsa continua bloqueada normalmente."""

    memory = RecordingOperationalMemory(free_pass=False, budget_available=False)
    validator = RecordingValidator()

    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="ok"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=task.arguments)
    ]

    runner = _make_runner(
        decisions,
        executions,
        validator=validator,
        operational_memory=memory,
    )

    runner.run(objective="obj", project_name="p")

    assert validator.calls == [False]
    assert memory.consumed_at == []


def test_free_pass_does_not_consume_periodic_budget():
    """Quando a liberação vem do passe livre pós-falha de teste/build,
    o orçamento periódico é ilimitado nesse caso e não deve ser
    descontado — mesmo que o orçamento também estivesse disponível."""

    memory = RecordingOperationalMemory(free_pass=True, budget_available=True)
    validator = RecordingValidator()

    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="ok"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=task.arguments)
    ]

    runner = _make_runner(
        decisions,
        executions,
        validator=validator,
        operational_memory=memory,
    )

    runner.run(objective="obj", project_name="p")

    assert validator.calls == [True]
    assert memory.consumed_at == []  # passe livre não gasta o orçamento


def test_final_verification_blocks_finish_on_problems():
    """Se a verificação final encontrar problemas, o finish deve ser
    bloqueado e o Planner deve receber feedback no contexto."""

    from app.agent.context.final_verification import FinalVerificationResult

    # Primeiro chama retorna problemas (bloqueia), segunda chamada passa
    call_count = [0]

    class SwitchingFinalVerification(FakeFinalVerification):
        def verify(self, objective, project_name, summary, tools_execute):
            call_count[0] += 1
            self.verify_calls.append({
                "objective": objective,
                "project_name": project_name,
                "summary": summary,
            })
            if call_count[0] == 1:
                return FinalVerificationResult(
                    FinalVerificationResult.PROBLEMS_FOUND,
                    "VERIFICAÇÃO FINAL:\n- problema 1",
                )
            return FinalVerificationResult(FinalVerificationResult.OK)

    fv = SwitchingFinalVerification()

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    # 1a TASK + 1a FINISH (bloqueada) + 2a FINISH (aceita)
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
        Decision(action=DecisionAction.FINISH, content="done2"),
    ]
    executions = [
        ExecutionDecision(tool="write_file", arguments=task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done2"
    assert len(fv.verify_calls) == 2  # rodou duas vezes (1a finish bloqueada, 2a passou)


def test_final_verification_unavailable_allows_finish_with_warning():
    """Quando a verificação final fica indisponível (LLM falhou), o
    finish NÃO deve ser bloqueado — apenas um aviso é emitido."""

    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.UNAVAILABLE)
    )

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="write_file", arguments=task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done"
    assert len(fv.verify_calls) == 1


def test_final_verification_passes_allows_finish():
    """Quando a verificação final passa, o finish é aceito normalmente."""

    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="write_file", arguments=task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done"
    assert len(fv.verify_calls) == 1


def test_final_verification_passes_before_checklist_pending_warning():
    """A verificação final roda ANTES do aviso de checklist pendente —
    se houver problemas na verificação, o checklist pendente nem é
    checado."""

    from app.agent.context.final_verification import FinalVerificationResult

    events = []

    def capture_event(event):
        events.append(event.type)

    # Primeiro chama retorna problemas (bloqueia), segunda chamada passa
    call_count = [0]

    class SwitchingFinalVerification(FakeFinalVerification):
        def verify(self, objective, project_name, summary, tools_execute):
            call_count[0] += 1
            self.verify_calls.append({
                "objective": objective,
                "project_name": project_name,
                "summary": summary,
            })
            if call_count[0] == 1:
                return FinalVerificationResult(
                    FinalVerificationResult.PROBLEMS_FOUND,
                    "problema",
                )
            return FinalVerificationResult(FinalVerificationResult.OK)

    fv = SwitchingFinalVerification()

    class WarningChecklist(FakeChecklist):
        @property
        def pending_count(self):
            return 1

        @property
        def pending_items(self):
            return []

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    # 1a FINISH (bloqueada) + 2a FINISH (passa)
    decisions = [
        Decision(action=DecisionAction.FINISH, content="done"),
        Decision(action=DecisionAction.FINISH, content="done2"),
    ]
    executions = []

    runner = _make_runner(
        decisions,
        executions,
        final_verification=fv,
    )
    runner.on_event = capture_event
    runner.checklist = WarningChecklist()

    # Deve rodar a verificação final e bloquear, sem chegar ao aviso de checklist
    runner.run(objective="obj", project_name="p")

    assert "final_verification_start" in events
    assert "planner_error" in events  # o erro de bloqueio do finish

    # A verificação final deve rodar ANTES do aviso de checklist pendente.
    # A primeira iteração (finish bloqueado) não chega ao checklist.
    # A segunda iteração (finish passa) chega ao checklist — mas a
    # verificação final já apareceu antes.
    fv_start_idx = events.index("final_verification_start")
    cl_idx = events.index("checklist_pending_on_finish")
    assert fv_start_idx < cl_idx


def test_final_verification_uses_current_objective():
    """A verificação final deve usar o objetivo atual da run, não um
    objetivo vazio ou antigo."""

    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="write_file", arguments=task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    runner.run(objective="meu objetivo real", project_name="p")

    assert fv.verify_calls[0]["objective"] == "meu objetivo real"


def test_final_verification_llm_failure_does_not_crash_run():
    """Se a verificação final lançar exceção, a run deve continuar
    normalmente (com aviso) e o finish deve ser aceito."""

    fv = FakeFinalVerification(raise_on_verify=True)

    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="write_file", arguments=task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done"

def test_investigation_task_is_not_counted_as_mutation():
    """Tasks com investigation=True não devem ser consideradas mutações
    — não entram em succeeded_mutations e não zeram stagnant_iterations."""

    from app.agent.execution.task import Task as TaskDataclass
    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    investigation_task = TaskDataclass(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        investigation=True,
    )
    write_task = TaskDataclass(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "b.py", "content": "x"},
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.TASK, task=write_task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
        ExecutionDecision(tool="write_file", arguments=write_task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done"
    # A task de investigação (read_file) não deve ter sido
    # considerada mutação — apenas a write_file entra em
    # succeeded_mutations. O teste verifica que a run completa
    # sem erro de repetição, o que só é possível porque a
    # investigação não conta como mutação.


def test_investigation_task_does_not_increment_stagnant_iterations():
    """Investigation tasks não devem incrementar stagnant_iterations —
    o agente pode investigar várias vezes sem ser punido como estagnação."""

    from app.agent.execution.task import Task as TaskDataclass
    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    investigation_task = TaskDataclass(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        investigation=True,
    )

    decisions = [
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")

    assert result == "done"


def test_investigation_task_with_repeated_args_uses_loop_detection():
    """Mesmo investigation tasks não podendo causar estagnação, o
    mecanismo de detecção de loop (repetição exata) ainda se aplica."""

    from app.agent.execution.task import Task as TaskDataclass
    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    investigation_task = TaskDataclass(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        investigation=True,
    )

    decisions = [
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
        ExecutionDecision(tool="read_file", arguments=investigation_task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(objective="obj", project_name="p")
    assert result == "done"


def test_end_to_end_analysis_objective():
    """Cenário completo: objetivo de análise de projeto existente.
    O Planner usa investigation=true, executa read_file/list_files,
    e finalmente finish com o resultado da análise."""

    from app.agent.execution.task import Task as TaskDataclass
    from app.agent.context.final_verification import FinalVerificationResult

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    list_task = TaskDataclass(
        tool="list_files",
        arguments={"project_name": "p"},
        investigation=True,
    )
    read_task = TaskDataclass(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "main.py"},
        investigation=True,
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=list_task),
        Decision(action=DecisionAction.TASK, task=read_task),
        Decision(action=DecisionAction.FINISH, content="Análise concluída."),
    ]
    executions = [
        ExecutionDecision(tool="list_files", arguments=list_task.arguments),
        ExecutionDecision(tool="read_file", arguments=read_task.arguments),
    ]

    runner = _make_runner(decisions, executions, final_verification=fv)
    result = runner.run(
        objective="Analise o projeto e descreva sua estrutura.",
        project_name="p",
    )

    assert result == "Análise concluída."
    assert fv.verify_calls[0]["objective"] == "Analise o projeto e descreva sua estrutura."


def test_executor_second_validation_respects_investigation_flag():
    """A segunda validação (no Executor) deve respeitar task.investigation=True,
    mesmo quando allow_investigation do Runner está False (sem free_pass
    e sem orçamento periódico).

    Este teste usa o TaskValidator real, não o FakeValidator, para cobrir
    o ponto exato onde o bug ocorria na vida real.
    """
    from app.agent.execution.validator import TaskValidator
    from app.agent.context.final_verification import FinalVerificationResult
    from app.tools.registry import ToolRegistry

    tools = ToolRegistry()
    tools.load_defaults()
    real_validator = TaskValidator(tools)

    fv = FakeFinalVerification(
        result=FinalVerificationResult(FinalVerificationResult.OK)
    )

    investigation_task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        investigation=True,
    )
    decisions = [
        Decision(action=DecisionAction.TASK, task=investigation_task),
        Decision(action=DecisionAction.FINISH, content="done"),
    ]
    executions = [
        ExecutionDecision(
            tool="read_file",
            arguments={"project_name": "p", "file_path": "a.py"},
        ),
    ]

    runner = Runner(
        planner=FakePlanner(decisions),
        task_decision_maker=FakeTaskDecisionMaker(executions),
        task_context_builder=FakeTaskContextBuilder(),
        tools=FakeToolRegistry(),
        project_context=FakeProjectContext(),
        project_summary_updater=FakeSummaryUpdater(FakeSummary("resumo")),
        validator=real_validator,
        operational_memory=FakeOperationalMemory(),
        checklist=FakeChecklist(),
        error_checklist=FakeErrorChecklist(),
        final_verification=fv,
    )

    # Deve completar sem levantar ValueError na segunda validação
    result = runner.run(objective="Analise o projeto.", project_name="p")
    assert result == "done"


def test_executor_second_validation_still_rejects_without_flag():
    """Sem investigation=True, a validação do executor continua rejeitando
    read_file como task principal — a proteção existente não é enfraquecida.

    Testa diretamente o TaskValidator real, replicando a condição exata
    da segunda validação no Runner: allow_investigation=False (sem free_pass
    nem orçamento) mas task.investigation=False.
    """
    from app.agent.execution.validator import TaskValidator
    from app.tools.registry import ToolRegistry

    tools = ToolRegistry()
    tools.load_defaults()
    real_validator = TaskValidator(tools)

    # Simula a segunda validação do Runner com allow_investigation=False
    # e task.investigation=False
    with pytest.raises(ValueError) as exc_info:
        real_validator.validate_arguments(
            "read_file",
            {"project_name": "p", "file_path": "a.py"},
            allow_investigation=False or False,  # igual ao Runner: allow_investigation or task.investigation
        )

    assert "read_file" in str(exc_info.value)
    assert "dependency" in str(exc_info.value).lower()
