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
):
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