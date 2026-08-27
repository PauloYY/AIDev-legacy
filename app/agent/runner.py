from app.agent.events import AgentEvent
from app.agent.planning.decision import DecisionAction
from app.agent.planning.planner import Planner
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.context.project_context import ProjectContext
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.exceptions import LLMInvalidResponseError
from app.tools.registry import ToolRegistry


class Runner:

    MAX_PLANNER_ATTEMPTS = 3
    MAX_EXECUTOR_ATTEMPTS = 3
    MAX_REPEATED_TASKS = 3

    def __init__(
        self,
        planner: Planner,
        task_decision_maker: TaskDecisionMaker,
        task_context_builder: TaskContextBuilder,
        tools: ToolRegistry,
        project_context: ProjectContext,
        project_summary_updater: ProjectSummaryUpdater,
        on_event=None,
    ):
        self.planner = planner
        self.task_decision_maker = task_decision_maker
        self.task_context_builder = task_context_builder
        self.tools = tools
        self.project_context = project_context
        self.project_summary_updater = project_summary_updater
        self.on_event = on_event

    def _emit(self, event_type: str, **data):
        if self.on_event:
            self.on_event(
                AgentEvent(
                    type=event_type,
                    data=data,
                )
            )

    def _task_signature(self, task):
        return (
            task.tool,
            repr(task.arguments),
        )

    def run(
        self,
        objective: str,
        project_name: str,
        context: str = "",
    ):
        self._emit("agent_start")

        summary = self.project_context.initialize(
            project_name
        )

        context = (
            f"RESUMO DO PROJETO:\n"
            f"{summary}\n\n"
            f"{context}"
        )

        last_task_signature = None
        repeated_task_count = 0
        planner_retry_context = ""

        while True:

            planner_attempts = 0

            while True:
                planner_attempts += 1

                self._emit("planner_start")

                try:
                    decision = self.planner.plan(
                        objective=objective,
                        context=(
                            f"{context}\n\n"
                            f"{planner_retry_context}"
                            if planner_retry_context
                            else context
                        ),
                    )

                except (ValueError, LLMInvalidResponseError) as error:
                    self._emit(
                        "planner_error",
                        error=str(error),
                    )

                    if planner_attempts >= self.MAX_PLANNER_ATTEMPTS:
                        raise RuntimeError(
                            "O Planner excedeu o limite de tentativas."
                        ) from error

                    planner_retry_context = (
                        "CORREÇÃO DA TENTATIVA ANTERIOR:\n"
                        f"{type(error).__name__}: {error}\n\n"
                        "A resposta anterior foi inválida.\n"
                        "Não execute ferramentas.\n"
                        "Retorne somente o JSON de decisão esperado pelo Planner."
                    )

                    continue

                self._emit(
                    "planner_end",
                    action=decision.action.value,
                )

                break

            if decision.action == DecisionAction.FINISH:
                self._emit("agent_done")
                return decision.content

            if decision.action == DecisionAction.FAIL:
                self._emit(
                    "agent_error",
                    error=decision.reason,
                )
                raise RuntimeError(decision.reason)

            if decision.action != DecisionAction.TASK:
                raise ValueError(
                    f"Ação desconhecida: {decision.action}"
                )

            task = decision.task

            # Detecta somente repetição consecutiva.
            task_signature = self._task_signature(task)

            if task_signature == last_task_signature:
                repeated_task_count += 1
            else:
                repeated_task_count = 1
                last_task_signature = task_signature

            if repeated_task_count > self.MAX_REPEATED_TASKS:
                error = (
                    "O Planner está repetindo a mesma task "
                    "sem produzir progresso."
                )

                self._emit(
                    "planner_error",
                    error=error,
                )

                context = (
                    f"{context}\n\n"
                    f"ERRO DE REPETIÇÃO:\n"
                    f"{error}\n\n"
                    f"Tool repetida: {task.tool}\n"
                    f"Argumentos: {task.arguments}\n\n"
                    "Não repita essa mesma ação novamente. "
                    "Analise o estado atual do projeto e escolha "
                    "uma ação diferente, obtenha novas informações "
                    "através de uma dependency ou use finish caso "
                    "o objetivo já tenha sido concluído."
                )

                repeated_task_count = 0
                last_task_signature = None

                continue

            dependency_results = []

            for dependency in task.dependencies:

                self._emit(
                    "tool_start",
                    name=dependency.tool,
                    arguments=dependency.arguments,
                    dependency=True,
                )

                try:
                    result = self.tools.execute(
                        dependency.tool,
                        dependency.arguments,
                    )

                except Exception as error:
                    result = (
                        f"ERRO NA DEPENDENCY:\n"
                        f"{type(error).__name__}: {error}"
                    )

                    self._emit(
                        "tool_error",
                        name=dependency.tool,
                        error=str(error),
                    )

                else:
                    self._emit(
                        "tool_end",
                        name=dependency.tool,
                        dependency=True,
                    )

                dependency_results.append(result)

            task_context = self.task_context_builder.build(
                task,
                dependency_results,
            )

            executor_attempts = 0

            while True:
                executor_attempts += 1

                self._emit(
                    "executor_start",
                    tool=task.tool,
                )

                try:
                    execution = self.task_decision_maker.decide(
                        objective=objective,
                        task=task,
                        context=task_context,
                    )

                except (ValueError, LLMInvalidResponseError) as error:
                    self._emit(
                        "executor_error",
                        error=str(error),
                    )

                    if executor_attempts >= self.MAX_EXECUTOR_ATTEMPTS:
                        raise RuntimeError(
                            "O Executor excedeu o limite de tentativas."
                        ) from error

                    task_context = (
                        f"{task_context}\n\n"
                        f"ERRO NA DECISÃO DE EXECUÇÃO:\n"
                        f"{type(error).__name__}: {error}\n\n"
                        "A decisão de execução anterior era inválida. "
                        "Analise o erro e tente novamente seguindo "
                        "rigorosamente as regras da task."
                    )

                    continue

                if execution.tool != task.tool:
                    error = (
                        "A LLM tentou executar uma tool diferente "
                        "da tool definida na task."
                    )

                    self._emit(
                        "executor_error",
                        error=error,
                    )

                    if executor_attempts >= self.MAX_EXECUTOR_ATTEMPTS:
                        raise RuntimeError(
                            "O Executor excedeu o limite de tentativas."
                        )

                    task_context = (
                        f"{task_context}\n\n"
                        f"ERRO NA DECISÃO DE EXECUÇÃO:\n"
                        f"{error}\n\n"
                        f"Tool esperada: {task.tool}\n"
                        f"Tool recebida: {execution.tool}\n\n"
                        "Corrija a decisão e tente novamente."
                    )

                    continue

                self._emit(
                    "executor_end",
                    tool=execution.tool,
                )

                break

            self._emit(
                "tool_start",
                name=execution.tool,
                arguments=execution.arguments,
            )

            try:
                result = self.tools.execute(
                    execution.tool,
                    execution.arguments,
                )

            except Exception as error:
                result = (
                    f"ERRO NA EXECUÇÃO DA TOOL:\n"
                    f"{type(error).__name__}: {error}"
                )

                self._emit(
                    "tool_error",
                    name=execution.tool,
                    error=str(error),
                )

            else:
                self._emit(
                    "tool_end",
                    name=execution.tool,
                )

            summary = self.project_summary_updater.update(
                objective=objective,
                project_name=project_name,
                task=task,
                result=result,
            )

            context = (
                f"RESUMO DO PROJETO:\n"
                f"{summary}\n\n"
                f"{task_context}\n\n"
                f"RESULTADO DA EXECUÇÃO:\n"
                f"{result}"
            )