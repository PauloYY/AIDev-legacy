import logging
from collections import deque

from app.agent.events import AgentEvent
from app.agent.execution.validator import TaskValidator
from app.agent.planning.decision import DecisionAction
from app.agent.planning.planner import Planner
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.context.project_context import ProjectContext
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.agent.context.operational_memory import OperationalMemory
from app.exceptions import LLMInvalidResponseError
from app.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


class Runner:

    MAX_PLANNER_ATTEMPTS = 3
    MAX_EXECUTOR_ATTEMPTS = 3
    MAX_ITERATIONS = 50
    MAX_TASK_HISTORY = 8
    MAX_DISTINCT_IN_HISTORY = 2
    MAX_STAGNANT_ITERATIONS = 10
    MAX_CONTEXT_CHARS = 4000

    # Padrões que indicam que um run_command está escrevendo/alterando
    # arquivos no disco (heredocs, redirecionamentos, mkdir, etc.).
    # Usado para tratar shell commands como progresso real, e não só
    # a tool write_file.
    MUTATION_PATTERNS = (
        "cat >", "cat >>", " > ", " >> ", "tee ", "touch ",
        "mkdir ", "sed -i", "cp ", "mv ", "rm ", "npm init",
        "go mod init",
    )

    FINISH_CHECK_TOOL = "check_project"

    def __init__(
        self,
        planner: Planner,
        task_decision_maker: TaskDecisionMaker,
        task_context_builder: TaskContextBuilder,
        tools: ToolRegistry,
        project_context: ProjectContext,
        project_summary_updater: ProjectSummaryUpdater,
        validator: TaskValidator | None = None,
        operational_memory: OperationalMemory | None = None,
        on_event=None,
        max_iterations: int | None = None,
    ):
        self.planner = planner
        self.task_decision_maker = task_decision_maker
        self.task_context_builder = task_context_builder
        self.tools = tools
        self.project_context = project_context
        self.project_summary_updater = project_summary_updater
        self.validator = validator or TaskValidator(tools)
        self.operational_memory = (
            operational_memory or OperationalMemory(tools)
        )
        self.on_event = on_event
        self.max_iterations = max_iterations or self.MAX_ITERATIONS

    def _emit(self, event_type: str, **data):
        if self.on_event:
            self.on_event(
                AgentEvent(
                    type=event_type,
                    data=data,
                )
            )

    def _detect_loop(self, history):
        if len(history) < history.maxlen:
            return None

        distinct = set(history)

        if len(distinct) <= self.MAX_DISTINCT_IN_HISTORY:
            return distinct

        return None

    def _task_signature(self, task):
        return (
            task.tool,
            repr(task.arguments),
        )

    def _command_mutates_files(self, command: str) -> bool:
        lowered = command.lower()
        return any(
            pattern in lowered for pattern in self.MUTATION_PATTERNS
        )

    def _is_mutating(self, tool: str, arguments: dict) -> bool:
        if tool == "write_file":
            return True

        if tool == "run_command":
            return self._command_mutates_files(
                str(arguments.get("command", ""))
            )

        return False

    def _truncate(self, text) -> str:
        text = str(text)

        if len(text) <= self.MAX_CONTEXT_CHARS:
            return text

        omitted = len(text) - self.MAX_CONTEXT_CHARS

        return (
            f"{text[:self.MAX_CONTEXT_CHARS]}\n"
            f"...[truncado, {omitted} caracteres omitidos]"
        )

    def _build_memory_block(self, project_name: str, summary: str) -> str:
        """Monta o bloco de contexto sempre injetado a cada iteração.

        Combina o RESUMO DO PROJETO (gerado por LLM, pode ficar impreciso)
        com a memória operacional determinística (histórico de ações +
        lista real de arquivos, gerados em Python puro por
        `OperationalMemory`). O segundo bloco serve como fonte de verdade
        caso o resumo tenha esquecido ou distorcido algo.
        """

        return (
            f"RESUMO DO PROJETO:\n"
            f"{summary}\n\n"
            f"{self.operational_memory.render(project_name)}"
        )

    def _run_finish_check(self, project_name: str) -> str | None:
        """Roda check_project antes de aceitar um finish.

        Retorna o relatório de erro (string) se algo falhou, ou None
        se está tudo certo, a tool não está registrada, ou a própria
        checagem não conseguiu rodar (nesses últimos dois casos não
        bloqueamos o finish por causa da ferramenta de checagem em si).
        """

        if not self.tools.exists(self.FINISH_CHECK_TOOL):
            return None

        self._emit(
            "tool_start",
            name=self.FINISH_CHECK_TOOL,
            arguments={"project_name": project_name},
        )

        try:
            result = self.tools.execute(
                self.FINISH_CHECK_TOOL,
                {"project_name": project_name},
            )

        except Exception as error:
            self._emit(
                "tool_error",
                name=self.FINISH_CHECK_TOOL,
                error=str(error),
            )
            logger.warning(
                "check_project falhou ao rodar antes do finish: %s",
                error,
            )
            return None

        self._emit(
            "tool_end",
            name=self.FINISH_CHECK_TOOL,
        )

        if "FALHOU" in result:
            return result

        return None

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

        self.operational_memory.reset()

        context = (
            f"{self._build_memory_block(project_name, summary)}\n\n"
            f"{context}"
        )

        task_history = deque(maxlen=self.MAX_TASK_HISTORY)
        succeeded_mutations = set()
        stagnant_iterations = 0
        planner_retry_context = ""
        iteration = 0

        while True:
            iteration += 1

            if iteration > self.max_iterations:
                error = (
                    f"O agente excedeu o limite de {self.max_iterations} "
                    "iterações sem concluir o objetivo."
                )

                logger.error(error)

                self._emit(
                    "agent_error",
                    error=error,
                )

                raise RuntimeError(error)

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

                    if decision.action == DecisionAction.TASK:
                        self.validator.validate(decision.task)

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
                check_error = self._run_finish_check(project_name)

                if check_error:
                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas o "
                            "check_project encontrou problemas."
                        ),
                    )

                    context = (
                        f"{context}\n\n"
                        f"ERRO DE VALIDAÇÃO ANTES DO FINISH:\n"
                        "Você tentou finalizar, mas a checagem "
                        "check_project encontrou problemas que "
                        "precisam ser corrigidos antes:\n\n"
                        f"{self._truncate(check_error)}\n\n"
                        "Corrija os problemas acima antes de tentar "
                        "finalizar novamente."
                    )

                    task_history.clear()
                    stagnant_iterations = 0

                    continue

                self._emit(
                    "agent_done",
                    usage=self.planner.llm.usage.summary(),
                )
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
            stagnant_iterations += 1

            # Detecta repetição consecutiva/cíclica E estagnação
            # (muitas iterações seguidas sem nenhuma escrita real).
            task_signature = self._task_signature(task)
            task_history.append(task_signature)

            loop_signatures = self._detect_loop(task_history)
            is_stagnant = stagnant_iterations > self.MAX_STAGNANT_ITERATIONS

            if loop_signatures is not None or is_stagnant:
                if is_stagnant and loop_signatures is None:
                    error = (
                        f"O Planner executou {stagnant_iterations} tasks "
                        "seguidas sem produzir nenhuma alteração real."
                    )
                    detail = (
                        "O agente está apenas lendo arquivos e rodando "
                        "comandos de verificação repetidamente, sem "
                        "escrever nenhuma mudança nova. Pare de investigar "
                        "e faça a próxima alteração de código necessária, "
                        "ou use finish/fail se não for possível avançar."
                    )
                else:
                    error = (
                        "O Planner está preso em um padrão de tasks "
                        "repetidas sem produzir progresso."
                    )
                    tasks_desc = "\n".join(
                        f"- tool={tool}, args={self._truncate(args)}"
                        for tool, args in (loop_signatures or [])
                    )
                    detail = (
                        f"Tasks envolvidas no loop:\n{tasks_desc}\n\n"
                        "Não repita nenhuma dessas ações. Analise o "
                        "estado atual do projeto e escolha uma ação "
                        "diferente, ou use finish caso o objetivo já "
                        "tenha sido concluído."
                    )

                self._emit("planner_error", error=error)

                context = (
                    f"{context}\n\n"
                    f"ERRO DE REPETIÇÃO/ESTAGNAÇÃO:\n"
                    f"{error}\n\n"
                    f"{detail}"
                )

                task_history.clear()
                stagnant_iterations = 0

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
                    dependency_succeeded = False

                    self._emit(
                        "tool_error",
                        name=dependency.tool,
                        error=str(error),
                    )

                else:
                    dependency_succeeded = True

                    self._emit(
                        "tool_end",
                        name=dependency.tool,
                        dependency=True,
                    )

                self.operational_memory.record(
                    iteration=iteration,
                    tool=dependency.tool,
                    arguments=dependency.arguments,
                    result=result,
                    success=dependency_succeeded,
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

                try:
                    self.validator.validate_arguments(
                        execution.tool,
                        execution.arguments,
                    )

                except ValueError as error:
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
                        f"ERRO DE VALIDAÇÃO DE ARGUMENTOS:\n"
                        f"{error}\n\n"
                        "Corrija os argumentos usando exatamente os "
                        "nomes de parâmetro esperados pela tool e "
                        "tente novamente."
                    )

                    continue

                self._emit(
                    "executor_end",
                    tool=execution.tool,
                )

                break

            # Bloqueia repetição exata de uma operação mutante
            # (write_file ou run_command que escreve arquivos) que já
            # foi executada com sucesso antes — mesmo que os detectores
            # de janela/estagnação acima não tenham pego, por estar
            # muito distante no histórico.
            execution_signature = (
                execution.tool,
                repr(execution.arguments),
            )

            if (
                self._is_mutating(execution.tool, execution.arguments)
                and execution_signature in succeeded_mutations
            ):
                error = (
                    "O Executor tentou repetir uma operação que já foi "
                    "executada com sucesso anteriormente."
                )

                self._emit("executor_error", error=error)

                context = (
                    f"{context}\n\n"
                    f"ERRO DE REPETIÇÃO:\n"
                    f"{error}\n\n"
                    f"Tool: {execution.tool}\n"
                    f"Argumentos: {self._truncate(execution.arguments)}\n\n"
                    "Essa exata operação já foi executada com sucesso "
                    "antes; refazê-la é inútil, as mudanças já existem. "
                    "Verifique o estado atual (list_files/read_file) e "
                    "escolha a próxima ação realmente necessária, ou use "
                    "finish caso o objetivo já tenha sido concluído."
                )

                task_history.clear()
                stagnant_iterations = 0

                continue

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
                execution_succeeded = False

                self._emit(
                    "tool_error",
                    name=execution.tool,
                    error=str(error),
                )

            else:
                execution_succeeded = True

                self._emit(
                    "tool_end",
                    name=execution.tool,
                )

                if self._is_mutating(execution.tool, execution.arguments):
                    stagnant_iterations = 0
                    succeeded_mutations.add(execution_signature)

            self.operational_memory.record(
                iteration=iteration,
                tool=execution.tool,
                arguments=execution.arguments,
                result=result,
                success=execution_succeeded,
                dependency=False,
            )

            summary = self.project_summary_updater.update(
                objective=objective,
                project_name=project_name,
                task=task,
                result=result,
            )

            context = (
                f"{self._build_memory_block(project_name, summary)}\n\n"
                f"{self._truncate(task_context)}\n\n"
                f"RESULTADO DA EXECUÇÃO:\n"
                f"{self._truncate(result)}"
            )