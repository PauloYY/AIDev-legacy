import logging
import time
from collections import deque

from app.agent.events import AgentEvent
from app.agent.execution.validator import TaskValidator
from app.agent.perf import AgentStats, format_performance_summary
from app.agent.planning.decision import DecisionAction
from app.agent.planning.planner import Planner
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.context.project_context import ProjectContext
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.agent.context.operational_memory import OperationalMemory
from app.agent.context.checklist import ProjectChecklist
from app.agent.context.error_checklist import ErrorChecklist
from app.agent.context.final_verification import FinalVerification
from app.agent.context.planner_error_memory import PlannerErrorMemory
from app.exceptions import LLMInvalidResponseError
from app.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


class Runner:

    MAX_PLANNER_ATTEMPTS = 5
    MAX_EXECUTOR_ATTEMPTS = 5
    MAX_ITERATIONS = 50
    MAX_TASK_HISTORY = 8
    MAX_DISTINCT_IN_HISTORY = 2
    MAX_STAGNANT_ITERATIONS = 10
    MAX_CONTEXT_CHARS = 4000
    MAX_LAST_RESORT_RECOVERIES = 3

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

    # Fase 3 (OPT-1): tools puramente observadoras — não alteram o disco
    # de forma relevante para o veredito do check_project, logo não
    # invalidam o cache de validação do finish.
    READ_ONLY_TOOLS = frozenset({
        "read_file",
        "list_files",
        "find_references",
        "list_symbols",
    })

    _FINISH_CHECK_MISS = object()

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
        checklist: ProjectChecklist | None = None,
        error_checklist: ErrorChecklist | None = None,
        planner_error_memory: PlannerErrorMemory | None = None,
        final_verification: FinalVerification | None = None,
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
        self.checklist = checklist or ProjectChecklist(planner.llm)
        self.error_checklist = (
            error_checklist or ErrorChecklist(planner.llm)
        )
        self.planner_error_memory = (
            planner_error_memory or PlannerErrorMemory()
        )
        self.final_verification = (
            final_verification or FinalVerification(planner.llm)
        )
        self.on_event = on_event
        self.max_iterations = max_iterations or self.MAX_ITERATIONS

    def _emit(self, event_type: str, **data):
        # Fase 3: contadores de performance (não alteram eventos).
        stats = getattr(self, "_stats", None)
        if stats is not None:
            if event_type == "planner_error":
                stats.corrections += 1
            elif event_type == "executor_error":
                stats.executor_errors += 1

        if self.on_event:
            self.on_event(
                AgentEvent(
                    type=event_type,
                    data=data,
                )
            )

    def _note_test_run(self, tool: str, arguments: dict, succeeded: bool) -> None:
        """Conta execuções de teste/build (Fase 3, só métrica)."""

        stats = getattr(self, "_stats", None)
        if stats is None:
            return

        if tool != "run_command":
            return

        command = ""
        if isinstance(arguments, dict):
            command = str(arguments.get("command", ""))

        if not self.operational_memory.is_test_or_build_command(command):
            return

        stats.test_runs += 1
        if succeeded:
            stats.test_passed += 1
        else:
            stats.test_failed += 1

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

    def _last_resort_recover_investigation(
        self,
        error: Exception,
        decision,
        iteration: int,
        context: str,
    ) -> str | None:
        """Último recurso antes de crashar a run.

        A regra de dependency-only (read_file/list_files/find_references
        não podem ser task principal, salvo logo após um teste/build
        falhar) continua valendo normalmente — isso aqui NÃO desativa
        a regra nem libera de forma geral. Ela só entra em ação quando
        o Planner já esgotou MAX_PLANNER_ATTEMPTS insistindo nesse
        exato erro: em vez de derrubar a run inteira por uma LLM presa
        num hábito de formatação recuperável, executamos a
        investigação (é uma tool de análise, segura, só leitura) e
        devolvemos o resultado real no contexto, dando mais uma
        chance à run continuar em vez de crashar.

        Retorna o novo `context` já com o resultado embutido, ou None
        se isso não se aplica a este erro (a run crasha normalmente).
        """

        if not isinstance(error, ValueError):
            return None

        if decision is None or decision.action != DecisionAction.TASK:
            return None

        task = decision.task
        tool_name = task.tool

        if tool_name not in TaskValidator.DEPENDENCY_ONLY_TOOLS:
            return None

        try:
            tool = self.tools.get(tool_name)
            self.validator.schema_validator.validate(tool, task.arguments)
        except Exception:
            # Os argumentos em si também estão errados — não dá pra
            # executar com segurança; deixa a run crashar.
            return None

        self._emit(
            "tool_start",
            name=tool_name,
            arguments=task.arguments,
            dependency=True,
        )

        try:
            result = self.tools.execute(tool_name, task.arguments)
        except Exception as tool_error:
            result = f"ERRO NA INVESTIGAÇÃO: {tool_error}"

        self._note_tool_execution(tool_name)

        investigation_succeeded = self._command_succeeded(tool_name, result)

        self._emit(
            "tool_end",
            name=tool_name,
            dependency=True,
            success=investigation_succeeded,
        )

        self.operational_memory.record(
            iteration=iteration,
            tool=tool_name,
            arguments=task.arguments,
            result=result,
            success=investigation_succeeded,
            dependency=True,
        )
        self._note_test_run(
            tool_name, task.arguments, investigation_succeeded
        )

        return (
            f"{context}\n\n"
            "INVESTIGAÇÃO REALIZADA COMO ÚLTIMO RECURSO (você insistiu "
            f"em usar '{tool_name}' como task principal mesmo após "
            "vários avisos — em vez de travar a run, a investigação foi "
            "executada por você desta vez):\n\n"
            f"{tool_name}({task.arguments}) ->\n"
            f"{self._truncate(result)}\n\n"
            "Isso NÃO é permissão geral: continue anexando "
            "read_file/list_files/find_references como dependency da "
            "ação real, exceto logo após um teste/build falhar. Agora "
            "escolha a próxima AÇÃO REAL usando essa informação."
        )

    def _command_succeeded(self, tool: str, result) -> bool:
        """Para run_command, sucesso da TOOL (não lançou exceção) não é
        o mesmo que sucesso do COMANDO (exit code 0) — run_command
        nunca levanta exceção por causa do exit code do shell, só por
        erro de infraestrutura (sandbox, argumentos inválidos etc.).
        Sem essa distinção, um comando que falhou (ex.: testes
        quebrando) ficaria marcado como "OK" no histórico e podia até
        ser lembrado como comando de teste/build que funciona.
        """

        if tool == "run_command":
            # O formato é controlado por run_command._format_result, cuja
            # primeira linha é sempre o veredito real ("STATUS: sucesso
            # (exit code 0)" ou "STATUS: falha ..."). Buscar no texto
            # inteiro gerava falso-positivo quando a saída do comando
            # continha o literal "STATUS: sucesso" apesar de falhar.
            first_line = str(result).lstrip().split("\n", 1)[0]
            return first_line.startswith("STATUS: sucesso")

        return True

    def _update_error_checklist(
        self,
        tool: str,
        arguments: dict,
        result,
        succeeded: bool,
        iteration: int | None = None,
    ) -> None:
        """Mantém o CHECKLIST DE ERROS sincronizado com o resultado
        real do último run_command de teste/build.

        Sucesso -> limpa (os testes voltaram a passar). Falha ->
        regera do zero a partir da saída atual (evidência real, não
        autoavaliação da LLM). Qualquer outra tool, ou um run_command
        que não pareça teste/build, não mexe no checklist de erros.

        Falha de infra (timeout do sandbox, tool que lançou exceção)
        NÃO regenera nem limpa: preserva os itens reais anteriores e
        garante bloqueio até um run com sucesso (timeout não é passe).
        """

        if tool != "run_command":
            return

        command = str(arguments.get("command", ""))

        if not self.operational_memory.is_test_or_build_command(command):
            return

        if succeeded:
            self.error_checklist.clear()
            return

        text = str(result).lstrip()

        if text.startswith("TIMEOUT:") or text.startswith(
            "ERRO NA EXECUÇÃO DA TOOL:"
        ):
            # Infraestrutura, não evidência de teste: não descarta os
            # itens reais anteriores nem finge que está tudo certo.
            reason = (
                "excedeu o tempo limite"
                if text.startswith("TIMEOUT:")
                else "falhou por erro de infraestrutura"
            )
            self.error_checklist.note_infra_failure(command, reason)
            return

        try:
            self.error_checklist.generate(command, result, iteration)
        except Exception as error:
            # O checklist de erros é um auxílio, não um requisito — se
            # a extração falhar, o Planner ainda tem o texto bruto do
            # resultado no contexto normal.
            logger.warning(
                "Falha ao gerar checklist de erros: %s", error
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

        Combina o RESUMO DO PROJETO (gerado por LLM, pode ficar impreciso),
        o CHECKLIST DO OBJETIVO (gerado uma vez, texto fixo, só o estado
        concluído/pendente muda), o CHECKLIST DE ERROS (efêmero, extraído
        por LLM da última falha de teste/build e limpo automaticamente
        quando os testes voltam a passar) e a memória operacional
        determinística (histórico de ações + lista real de arquivos,
        gerados em Python puro por `OperationalMemory`). Os últimos três
        servem como fonte de verdade caso o resumo tenha esquecido ou
        distorcido algo.
        """

        error_block = self.error_checklist.render()

        return (
            f"RESUMO DO PROJETO:\n"
            f"{summary}\n\n"
            f"{self.checklist.render()}\n\n"
            + (f"{error_block}\n\n" if error_block else "")
            + f"{self.planner_error_memory.render()}\n\n"
            + f"{self.operational_memory.render(project_name)}"
        )

    def _note_tool_execution(self, tool_name: str) -> None:
        """Invalida o cache do finish-check após escrita/efeito (Fase 3).

        Leituras puras preservam o cache: o veredito do check_project é
        função determinística dos fontes no disco. O próprio check_project
        também não invalida (sintaxe/compilação só-leitura; artefatos vão
        para diretórios ignorados como __pycache__/target).
        """

        if tool_name not in self.READ_ONLY_TOOLS:
            self._finish_check_cache = self._FINISH_CHECK_MISS

    def _cached_finish_check(self, project_name: str) -> str | None:
        """check_project com cache entre finishs consecutivos (Fase 3).

        Um finish bloqueado dá `continue` sem executar nada; o próximo
        finish revalidaria arquivos idênticos (N subprocess/containers).
        A verificação LLM (FinalVerification, estocástica) continua
        sempre fresca — só o veredito determinístico é reutilizado.
        """

        cached = getattr(
            self, "_finish_check_cache", self._FINISH_CHECK_MISS
        )
        if cached is not self._FINISH_CHECK_MISS:
            logger.debug("Reutilizando check_project do finish anterior.")
            return cached

        report = self._run_finish_check(project_name)
        self._finish_check_cache = report
        return report

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
            success="FALHOU" not in result,
        )

        if "FALHOU" in result:
            return result

        return None

    def _run_final_verification(self, project_name: str, summary: str) -> str | None:
        """Roda a verificação final (análise semântica/integração) antes
        de aceitar um finish.

        Esta camada checa se o objetivo foi REALMENTE atendido — não
        só sintaxe ou testes passando, mas se o projeto como um todo
        corresponde ao que foi pedido, sem inconsistências de integração
        entre arquivos.

        Retorna o relatório de problemas (string) se algo falhou, ou
        None se a verificação passou, não foi configurada, ou não pôde
        rodar (nesse último caso, NÃO bloqueia — apenas loga o aviso).
        """

        self._emit("final_verification_start")

        try:
            result = self.final_verification.verify(
                objective=getattr(self, "_current_objective", ""),
                project_name=project_name,
                summary=summary,
                tools_execute=self.tools.execute,
            )
        except Exception as error:
            self._emit(
                "final_verification_error",
                error=str(error),
            )
            logger.warning(
                "Verificação final falhou ao rodar: %s", error,
            )
            return None

        self._emit(
            "final_verification_end",
            status=result.status,
        )

        if result.is_unavailable:
            logger.warning(
                "Verificação final indisponível (LLM falhou). "
                "Não bloqueando o finish, mas registrando o estado degradado."
            )
            self._emit(
                "final_verification_warning",
                message="Verificação final indisponível — LLM falhou ou retornou JSON inválido.",
            )
            return None

        if result.is_problems:
            return result.report

        return None

    def run(
        self,
        objective: str,
        project_name: str,
        context: str = "",
    ):
        self._emit("agent_start")

        self._current_objective = objective
        self._stats = AgentStats()
        self._finish_check_cache = self._FINISH_CHECK_MISS
        run_start = time.monotonic()

        summary = self.project_context.initialize(
            project_name
        )

        self.operational_memory.reset()
        self.checklist.reset()
        self.error_checklist.reset()
        self.planner_error_memory.reset()

        try:
            self.checklist.generate(objective)
        except Exception as error:
            # O checklist é um auxílio, não um requisito — se a
            # geração falhar (erro de LLM, JSON malformado etc.), o
            # agente segue sem ele em vez de travar a run inteira.
            logger.warning(
                "Falha ao gerar checklist do objetivo: %s", error
            )

        context = (
            f"{self._build_memory_block(project_name, summary)}\n\n"
            f"{context}"
        )

        task_history = deque(maxlen=self.MAX_TASK_HISTORY)
        # Assinatura (tool, args) -> época da última execução bem-sucedida.
        # Uma re-execução idêntica só é bloqueada quando NENHUMA outra
        # execução ocorreu desde aquele sucesso (sem evidência nova).
        # Qualquer execução posterior que não seja uma mutação
        # bem-sucedida — falha de teste/tool OU nova investigação —
        # avança a época e libera nova tentativa, que pode ser uma
        # correção informada pela nova evidência. Laços reais seguem
        # contidos pelo detector de janela, estagnação e max_iterations.
        succeeded_mutations: dict = {}
        progress_epoch = 0
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
            allow_investigation = False
            used_investigation_budget = False

            while True:
                planner_attempts += 1
                decision = None

                self._emit("planner_start")
                self._stats.planner_calls += 1

                try:
                    decision = self.planner.plan(
                        objective=objective,
                        context=(
                            f"{context}\n\n"
                            f"{planner_retry_context}"
                            if planner_retry_context
                            else context
                        ),
                        iteration=iteration,
                    )

                    if decision.action == DecisionAction.TASK:
                        is_investigation_task = (
                            decision.task.tool
                            in TaskValidator.DEPENDENCY_ONLY_TOOLS
                        )
                        free_pass = (
                            self.operational_memory
                            .last_run_command_failed_test_or_build()
                        )
                        budget_available = (
                            self.operational_memory
                            .investigation_budget_available(iteration)
                        )

                        allow_investigation = is_investigation_task and (
                            free_pass or budget_available
                        )
                        # Só consome o orçamento periódico quando ele foi
                        # de fato o motivo da liberação — o passe livre
                        # pós-falha de teste/build é ilimitado e não deve
                        # gastar essa janela.
                        used_investigation_budget = (
                            is_investigation_task
                            and not free_pass
                            and budget_available
                        )

                        self.validator.validate(
                            decision.task,
                            allow_investigation=allow_investigation,
                        )

                except (ValueError, LLMInvalidResponseError) as error:
                    self._emit(
                        "planner_error",
                        error=str(error),
                    )

                    error_signature = str(error)
                    already_forbidden = self.planner_error_memory.seen(
                        error_signature
                    )
                    self.planner_error_memory.record(error_signature)

                    if planner_attempts >= self.MAX_PLANNER_ATTEMPTS:
                        recovered_context = (
                            self._last_resort_recover_investigation(
                                error, decision, iteration, context,
                            )
                        )

                        if recovered_context is not None:
                            context = recovered_context
                            planner_retry_context = ""
                            planner_attempts = 0
                            continue

                        raise RuntimeError(
                            "O Planner excedeu o limite de tentativas."
                        ) from error

                    # Verificação extra: se este exato erro já tinha
                    # sido cometido antes (nesta mesma iteração ou em
                    # alguma anterior), a correção genérica já não
                    # funcionou da última vez — em vez de repetir a
                    # mesma mensagem fraca, recalcula com uma correção
                    # bem mais explícita e direta, pra aumentar a
                    # chance de acertar já na próxima tentativa, dentro
                    # do mesmo orçamento normal de MAX_PLANNER_ATTEMPTS
                    # (não pula tentativa nenhuma, só melhora o pedido).
                    if already_forbidden:
                        planner_retry_context = (
                            "ERRO REPETIDO — LEIA COM ATENÇÃO:\n"
                            f"{type(error).__name__}: {error}\n\n"
                            "Você JÁ cometeu esse EXATO erro antes "
                            "nesta run e está proibido de repeti-lo. "
                            "A tentativa anterior de corrigir não "
                            "funcionou — não repita a mesma decisão "
                            "de novo. Mude a tool, os argumentos, ou a "
                            "abordagem de forma diferente da tentativa "
                            "anterior.\n"
                            "Não execute ferramentas.\n"
                            "Retorne somente o JSON de decisão "
                            "esperado pelo Planner."
                        )
                    else:
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

            marked = self.checklist.mark_done(decision.checklist_progress)

            if marked:
                self._emit("checklist_updated", marked=marked)

            if decision.action == DecisionAction.FINISH:
                check_error = self._cached_finish_check(project_name)

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
                    self._stats.finish_blocks += 1

                    continue

                if self.error_checklist.pending_count:
                    error_block = self.error_checklist.render()

                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas há "
                            "falhas de teste/build ainda não "
                            "corrigidas."
                        ),
                    )

                    context = (
                        f"{context}\n\n"
                        f"ERRO DE VALIDAÇÃO ANTES DO FINISH:\n"
                        "Você tentou finalizar, mas o último "
                        "teste/build rodado ainda está falhando:\n\n"
                        f"{self._truncate(error_block)}\n\n"
                        "Corrija essas falhas e rode o teste/build de "
                        "novo, confirmando que ele passa, antes de "
                        "tentar finalizar de novo."
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    continue

                current_summary = self.project_context.summary.read(
                    project_name
                )
                final_check_error = self._run_final_verification(
                    project_name, current_summary,
                )

                if final_check_error:
                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas a "
                            "verificação final encontrou problemas."
                        ),
                    )

                    context = (
                        f"{context}\n\n"
                        f"ERRO DE VALIDAÇÃO ANTES DO FINISH:\n"
                        "Você tentou finalizar, mas a verificação "
                        "final do projeto encontrou problemas que "
                        "precisam ser corrigidos antes:\n\n"
                        f"{self._truncate(final_check_error)}\n\n"
                        "Corrija os problemas acima antes de tentar "
                        "finalizar novamente."
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    continue

                if self.checklist.pending_count:
                    self._emit(
                        "checklist_pending_on_finish",
                        pending=[
                            item.description
                            for item in self.checklist.pending_items
                        ],
                    )
                    logger.warning(
                        "Finish aceito com %d item(ns) de checklist "
                        "ainda pendente(s).",
                        self.checklist.pending_count,
                    )

                updater_summary_fn = getattr(
                    self.planner.llm.usage,
                    "project_summary_section",
                    None,
                )
                self._stats.iterations = iteration
                self._stats.wall_ms = (time.monotonic() - run_start) * 1000
                tool_stats_fn = getattr(
                    self.tools, "tool_stats", None
                )
                try:
                    perf_summary = format_performance_summary(
                        "SUCCESS",
                        self.planner.llm.usage,
                        tool_stats_fn() if tool_stats_fn else {},
                        self._stats,
                    )
                except Exception:
                    perf_summary = None
                self._emit(
                    "agent_done",
                    usage=self.planner.llm.usage.summary(),
                    usage_breakdown=self.planner.llm.usage.breakdown(),
                    updater_summary=updater_summary_fn()
                    if updater_summary_fn
                    else None,
                    perf_summary=perf_summary,
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
            if not task.investigation:
                stagnant_iterations += 1

            if used_investigation_budget:
                self.operational_memory.consume_investigation_budget(
                    iteration
                )

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
                        "e faça a próxima ação de fato necessária, "
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
                self._stats.loop_hits += 1

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
                    dependency_succeeded = self._command_succeeded(
                        dependency.tool, result
                    )

                    self._emit(
                        "tool_end",
                        name=dependency.tool,
                        dependency=True,
                        success=dependency_succeeded,
                    )

                logger.debug(
                    "Resultado de %s (dependency): %s",
                    dependency.tool,
                    self._truncate(result),
                )

                self.operational_memory.record(
                    iteration=iteration,
                    tool=dependency.tool,
                    arguments=dependency.arguments,
                    result=result,
                    success=dependency_succeeded,
                    dependency=True,
                )
                self._note_tool_execution(dependency.tool)
                self._note_test_run(
                    dependency.tool,
                    dependency.arguments,
                    dependency_succeeded,
                )

                if not (
                    dependency_succeeded
                    and self._is_mutating(
                        dependency.tool, dependency.arguments
                    )
                ):
                    progress_epoch += 1

                self._update_error_checklist(
                    dependency.tool,
                    dependency.arguments,
                    result,
                    dependency_succeeded,
                    iteration,
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
                        iteration=iteration,
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
                        allow_investigation=allow_investigation or task.investigation,
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
            # foi executada com sucesso antes SEM que nenhuma outra
            # execução tenha ocorrido desde então — nesse caso refazê-la
            # é inútil, as mudanças já existem. Se houve falha posterior
            # (teste quebrou, tool errou) ou nova investigação, a
            # repetição pode ser uma tentativa legítima de correção
            # informada pela nova evidência, e é permitida (outros
            # detectores — janela de tasks, estagnação, max_iterations
            # — seguem valendo).
            execution_signature = (
                execution.tool,
                repr(execution.arguments),
            )

            if (
                self._is_mutating(execution.tool, execution.arguments)
                and succeeded_mutations.get(execution_signature)
                == progress_epoch
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
                self._stats.loop_hits += 1

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
                execution_succeeded = self._command_succeeded(
                    execution.tool, result
                )

                self._emit(
                    "tool_end",
                    name=execution.tool,
                    success=execution_succeeded,
                )

                if execution_succeeded and self._is_mutating(
                    execution.tool, execution.arguments
                ):
                    # Só zera a estagnação para operações inéditas: um
                    # sucesso com conteúdo idêntico a sucesso anterior
                    # não é evidência de progresso (a correção repetida
                    # precisa se provar no teste seguinte).
                    if execution_signature not in succeeded_mutations:
                        stagnant_iterations = 0
                    succeeded_mutations[execution_signature] = progress_epoch

            logger.debug(
                "Resultado de %s: %s",
                execution.tool,
                self._truncate(result),
            )

            self.operational_memory.record(
                iteration=iteration,
                tool=execution.tool,
                arguments=execution.arguments,
                result=result,
                success=execution_succeeded,
                dependency=False,
            )
            self._note_tool_execution(execution.tool)
            self._note_test_run(
                execution.tool, execution.arguments, execution_succeeded
            )

            if not (
                execution_succeeded
                and self._is_mutating(execution.tool, execution.arguments)
            ):
                progress_epoch += 1

            self._update_error_checklist(
                execution.tool,
                execution.arguments,
                result,
                execution_succeeded,
                iteration,
            )

            summary = self.project_summary_updater.update(
                objective=objective,
                project_name=project_name,
                task=task,
                result=result,
                iteration=iteration,
            )

            context = (
                f"{self._build_memory_block(project_name, summary)}\n\n"
                f"{self._truncate(task_context)}\n\n"
                f"RESULTADO DA EXECUÇÃO:\n"
                f"{self._truncate(result)}"
            )