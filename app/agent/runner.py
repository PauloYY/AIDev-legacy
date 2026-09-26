import logging
import time
from collections import deque

from app.agent.events import AgentEvent
from app.agent.execution.validator import TaskValidator
from app.agent.parallel import (
    all_pure_read,
    estimated_saved_ms,
    run_concurrent,
)
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
from app.agent.trace import (
    ExecutionTrace,
    NullTrace,
    error_type_and_signature,
    extract_file_path,
    is_timeout_result,
    parse_exit_code,
    sanitize_arguments,
    sanitize_result,
    truncate_text,
)
from app.exceptions import LLMInvalidResponseError
from app.config import Config
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

    MUTATION_PATTERNS = (
        "cat >", "cat >>", " > ", " >> ", "tee ", "touch ",
        "mkdir ", "sed -i", "cp ", "mv ", "rm ", "npm init",
        "go mod init",
    )

    FINISH_CHECK_TOOL = "check_project"

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
        execution_trace: ExecutionTrace | NullTrace | None = None,
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
        self.execution_trace = execution_trace
        self.task_state = None
        self._task_state_project = ""
        self._task_state_saved_sig = None

    def _emit(self, event_type: str, **data):
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


    def _init_task_state(self, objective: str, project_name: str = ""):
        """Pipeline prompt → canonical → Interpreter → TaskState.

        Nunca levanta: qualquer erro (LLM, parsing) resulta num estado
        mínimo com o prompt original — a run continua no fluxo antigo.
        Com persistência ligada, tenta recuperar o estado da MESMA
        tarefa (task_id); tarefa diferente → arquiva o antigo e começa
        novo (nunca apaga, nunca mistura).
        """
        from hashlib import sha1

        from app.agent.taskstate.canonicalizer import PromptCanonicalizer
        from app.agent.taskstate.interpreter import TaskInterpreter
        from app.agent.taskstate.task_state import TaskState

        try:
            text = objective if isinstance(objective, str) else str(objective)
        except Exception:
            text = ""

        def _fresh():
            try:
                planner_llm = getattr(getattr(self, "planner", None),
                                      "llm", None)
                canon = PromptCanonicalizer(
                    llm=planner_llm).canonicalize(text)
                interpretation = TaskInterpreter().interpret(
                    canon.canonical_prompt, canon.language)
                try:
                    task_id = sha1(canon.canonical_prompt.encode(
                        "utf-8")).hexdigest()[:12]
                except Exception:
                    task_id = ""
                return TaskState(
                    original_prompt=canon.original_prompt,
                    canonical_prompt=canon.canonical_prompt,
                    language=canon.language,
                    translation_applied=canon.translation_applied,
                    task_id=task_id,
                    objective=(interpretation.objective
                               or canon.canonical_prompt[:300]),
                    requirements=interpretation.requirements,
                    constraints=interpretation.constraints,
                    ambiguities=interpretation.ambiguities,
                )
            except Exception as error:
                logger.warning("TaskState inicial mínimo (falha: %s)",
                               error)
                try:
                    return TaskState(original_prompt=text,
                                     canonical_prompt=text,
                                     objective=text[:300])
                except Exception:
                    return None

        fresh = _fresh()
        if fresh is None or not bool(
                getattr(Config, "task_state_persist", False)):
            return fresh
        try:
            from app.agent.taskstate.persistence import (
                archive_task_state,
                load_task_state,
            )

            if not project_name or not getattr(fresh, "task_id", ""):
                return fresh
            restored = load_task_state(project_name)
            if restored is not None and (
                    restored.task_id == fresh.task_id):
                try:
                    self._trace_or_null().record(
                        "task_state_restore",
                        task_id=restored.task_id,
                        decisions=len(restored.decisions),
                        progress=len(restored.progress),
                    )
                except Exception:
                    pass
                return restored
            if restored is not None:
                archive_task_state(project_name, restored.task_id)
        except Exception as error:
            logger.warning("TaskState restore falhou: %s", error)
        return fresh

    def _note_task_decision(self, iteration, decision) -> None:
        """Planner → decisions do TaskState (compacto, sem conteúdos)."""
        try:
            state = getattr(self, "task_state", None)
            if state is None or decision is None:
                return
            action = getattr(decision.action, "value", str(
                decision.action))
            tool = file_path = summary = None
            task = getattr(decision, "task", None)
            if task is not None:
                tool = getattr(task, "tool", None)
                try:
                    file_path = extract_file_path(
                        getattr(task, "arguments", None))
                except Exception:
                    file_path = None
                try:
                    n_deps = len(getattr(task, "dependencies", None) or [])
                except Exception:
                    n_deps = 0
                summary = f"tool={tool} deps={n_deps}"
            else:
                text = (getattr(decision, "content", None)
                        or getattr(decision, "reason", None) or "")
                summary = str(text)[:200]
            state.record_decision(iteration, action, tool, file_path,
                                  summary)
            self._save_task_state("decision")
            self._emit_task_state_update(iteration)
        except Exception as error:
            logger.warning("TaskState decision falhou: %s", error)

    def _note_task_progress(self, iteration, tool, arguments, result,
                            success) -> None:
        """Executor → progress do TaskState (1 linha por execução)."""
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return
            try:
                file_path = extract_file_path(arguments)
            except Exception:
                file_path = None
            try:
                note = str(result).lstrip().split("\n", 1)[0][:200]
            except Exception:
                note = ""
            state.record_progress(iteration, tool, success, file_path,
                                  note)
            self._save_task_state("progress")
            self._emit_task_state_update(iteration)
        except Exception as error:
            logger.warning("TaskState progress falhou: %s", error)

    def _note_state_verification(self, gate, passed, detail,
                                  iteration) -> None:
        """Finish gates → verification do TaskState."""
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return
            state.record_verification(gate, passed, detail, iteration)
            self._save_task_state(f"verification:{gate}")
        except Exception as error:
            logger.warning("TaskState verification falhou: %s", error)


    def _downstream_objective(self, objective: str) -> str:
        """Objective que Planner/Executor/checklist/summary recebem.

        Com AIDEV_CANONICAL_OBJECTIVE=1: prompt canônico integral (a
        fonte de verdade passa a ser o TaskState; o original segue
        preservado no estado + trace + bloco TASK STATE). Com 0 (ou
        estado ausente): original, como antes.
        """
        try:
            if not bool(getattr(Config, "canonical_objective", False)):
                return objective
            state = getattr(self, "task_state", None)
            canonical = getattr(state, "canonical_prompt", "") or ""
            if canonical.strip():
                return canonical
        except Exception as error:
            logger.warning("Objective canônico falhou: %s", error)
        return objective

    def _with_task_state(self, context,
                           problem_detail: bool = True) -> str:
        """Injeta o bloco TASK STATE no início do contexto.

        Com AIDEV_TASK_STATE_CONTEXT=0 ou sem estado: devolve o
        contexto intacto. Com 1: `TASK STATE:\\n<render_compact>\\n\\n`
        + contexto - estado semântico primeiro, evidência operacional
        depois, sem conteúdos integrais nem duplicação do OBJECTIVE
        (a seção OBJECTIVE do prompt já o carrega).
        `problem_detail=False` (Executor): só contagens de problemas,
        pois o detalhe vai via KNOWN PROBLEMS (sem duplicar).
        """
        try:
            base = context if isinstance(context, str) else str(
                context or "")
        except Exception:
            return context
        try:
            if not bool(getattr(Config, "task_state_context", False)):
                return base
            state = getattr(self, "task_state", None)
            if state is None:
                return base
            try:
                block = state.render_compact(
                    include_objective=False, original_ref_chars=200,
                    problem_detail=problem_detail)
            except TypeError:
                block = state.render_compact(
                    include_objective=False, original_ref_chars=200)
            if not block.strip():
                return base
            if not base.strip():
                return f"TASK STATE:\n{block}"
            return f"TASK STATE:\n{block}\n\n{base}"
        except Exception as error:
            logger.warning("Injeção do TaskState falhou: %s", error)
            return base

    def _sync_task_plan(self) -> None:
        """Espelha o checklist em TaskState.plan (idempotente)."""
        try:
            if not bool(getattr(Config, "task_plan_sync", False)):
                return
            state = getattr(self, "task_state", None)
            if state is None:
                return
            statuses = getattr(self.checklist, "statuses", None)
            if statuses is None:
                return
            try:
                blocked = (
                    self.error_checklist.pending_count > 0)
            except Exception:
                blocked = False
            started = bool(getattr(state, "decisions", None)
                           or getattr(state, "progress", None))
            state.sync_plan(list(statuses), blocked, started)
        except Exception as error:
            logger.warning("Sync do plan falhou: %s", error)

    def _task_state_signature(self):
        """Assinatura do estado p/ só persistir quando algo mudou."""
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return None
            return (
                len(state.decisions), len(state.progress),
                len(state.problems),
                sum(1 for p in state.problems if p.resolved),
                len(state.corrections), len(state.verification),
                tuple((item.index, item.status) for item in state.plan),
                state.task_id,
            )
        except Exception:
            return None

    def _save_task_state(self, reason: str, force: bool = False) -> None:
        """Persiste.aidev/task_state.json (atômico, best-effort).

        Com AIDEV_TASK_STATE_PERSIST=0: nada faz. Sem mudança desde o
        último save (e sem force): nada faz. Falha → evento
        `task_state_persist_error`, nunca exceção.
        """
        try:
            if not bool(getattr(Config, "task_state_persist", False)):
                return
            state = getattr(self, "task_state", None)
            project = getattr(self, "_task_state_project", "")
            if state is None or not project:
                return
            sig = self._task_state_signature()
            if not force and sig is not None and sig == getattr(
                    self, "_task_state_saved_sig", None):
                return
            from app.agent.taskstate.persistence import save_task_state

            result = save_task_state(state, project)
            trace = self._trace_or_null()
            if result.get("ok"):
                self._task_state_saved_sig = sig
                trace.record("task_state_persist", reason=reason,
                             path=result.get("path"),
                             bytes=result.get("bytes", 0))
            else:
                trace.record("task_state_persist_error", reason=reason,
                             error=str(result.get("error", ""))[:200])
        except Exception as error:
            logger.warning("Save do TaskState falhou: %s", error)


    @staticmethod
    def _gate_command(task) -> str:
        """Comando/resumo p/ eventos do gate (nunca levanta)."""
        try:
            args = getattr(task, "arguments", None) or {}
            if isinstance(args, dict) and args.get("command"):
                return str(args.get("command", ""))[:200]
            return str(getattr(task, "tool", ""))[:60]
        except Exception:
            return ""

    def _is_gated_test_task(self, task) -> str | None:
        """Task é retest bloqueável? Retorna rótulo ou None.

        Bloqueáveis: run_command classificado teste/build (reusa
        is_test_or_build_command, sem heurística nova) e check_project
        como task principal (validação). Investigação (ls, cat,
        --version, read_file solo etc.) nunca é bloqueada.
        """
        try:
            if task is None:
                return None
            tool = getattr(task, "tool", None)
            if tool == "check_project":
                return "check_project"
            if tool == "run_command":
                args = getattr(task, "arguments", None)
                command = ""
                if isinstance(args, dict):
                    command = str(args.get("command", ""))
                if self.operational_memory.is_test_or_build_command(
                        command):
                    return command[:200]
        except Exception as error:
            logger.warning("Gate: classificação falhou: %s", error)
        return None

    def _blocking_ids(self) -> list[str]:
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return []
            return state.blocking_problem_ids()
        except Exception:
            return []

    def _check_test_gate(self, task, iteration) -> str | None:
        """Mensagem de bloqueio ou None.

        Com AIDEV_ERROR_TEST_GATE=0: nunca bloqueia (problemas seguem
        registrados). Caso contrário bloqueia retest com problemas
        pending/in_progress/blocked. `test_allowed` só é emitido
        quando há histórico de problemas (evita ruído).
        """
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return None
            gated = self._is_gated_test_task(task)
            if gated is None:
                return None
            if not bool(getattr(Config, "error_test_gate", False)):
                return None
            if not state.has_blocking_problems():
                if state.problems or state.corrections:
                    self._trace_or_null().record(
                        "test_allowed", iteration=iteration,
                        command=gated[:200],
                        past_problems=len(state.problems),
                        corrections=len(state.corrections),
                    )
                return None
            ids = state.blocking_problem_ids()
            listed = "\n".join(f"- {pid}" for pid in ids[:20])
            extra = (f"\n... [+{len(ids) - 20} more]"
                     if len(ids) > 20 else "")
            return (
                "TEST_BLOCKED_BY_PENDING_PROBLEMS\n"
                "A retest/validation was requested while known problems "
                "are still pending. Do NOT rerun tests now — fix the "
                "known problems first (use read_file to verify each "
                "diagnosis before editing).\n\n"
                f"Pending problems:\n{listed}{extra}\n\n"
                "Resolve known problems before rerunning tests."
            )
        except Exception as error:
            logger.warning("Gate de teste falhou (permitindo): %s",
                           error)
            return None

    def _known_problems_block(self) -> str:
        """Bloco KNOWN PROBLEMS p/ o Executor (capped)."""
        try:
            if not bool(getattr(Config, "error_analyzer", False)):
                return ""
            state = getattr(self, "task_state", None)
            if state is None:
                return ""
            return state.render_known_problems()
        except Exception as error:
            logger.warning("Bloco KNOWN PROBLEMS falhou: %s", error)
            return ""

    def _attempt_files(self, tool: str, arguments: dict) -> list[str]:
        """Arquivos-alvo conhecidos da execução (p/ vínculo)."""
        try:
            if tool in ("write_file", "edit_file", "delete_file") and isinstance(arguments, dict):
                path = arguments.get("file_path")
                if isinstance(path, str) and path:
                    return [path]
        except Exception:
            pass
        return []

    def _mark_correction_attempt(self, iteration, tool: str,
                                 arguments: dict) -> None:
        """Execução mutante começou → candidatos a in_progress."""
        try:
            if not self._is_mutating(tool, arguments):
                return
            state = getattr(self, "task_state", None)
            if state is None:
                return
            files = self._attempt_files(tool, arguments)
            marked = state.mark_attempt_started(files, iteration)
            if marked:
                self._trace_or_null().record(
                    "correction_started", iteration=iteration,
                    tool=tool, files=files[:5], problems=marked[:20],
                )
        except Exception as error:
            logger.warning("Attempt de correção falhou: %s", error)

    def _note_correction_applied(self, iteration, tool: str,
                                 arguments: dict) -> None:
        """Execução mutante OK → Correction + pending_verification."""
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return
            if not self._is_mutating(tool, arguments):
                return
            files = self._attempt_files(tool, arguments)
            target = files[0] if files else (
                str(arguments.get("command", ""))[:120]
                if isinstance(arguments, dict) else tool)
            correction = state.apply_correction(
                f"fix attempt: {tool} {target}", files, iteration)
            if correction is not None:
                self._trace_or_null().record(
                    "correction_applied", iteration=iteration,
                    correction_id=correction.correction_id,
                    problems=list(correction.problem_ids)[:20],
                    files=files[:5],
                )
                self._save_task_state("correction")
        except Exception as error:
            logger.warning("Correction falhou: %s", error)

    def _mark_attempt_failed(self, tool: str, arguments: dict) -> None:
        """Execução mutante com erro de tool → in_progress vira blocked."""
        try:
            if not self._is_mutating(tool, arguments):
                return
            state = getattr(self, "task_state", None)
            if state is None:
                return
            marked = state.mark_attempt_failed(
                self._attempt_files(tool, arguments))
            if marked:
                self._save_task_state("attempt_failed")
        except Exception as error:
            logger.warning("Blocked de correção falhou: %s", error)

    def _analyze_test_failure_unified(
        self, command: str, result, iteration,
    ) -> list[str]:
        """Uma análise → TaskState + Checklist (1 LLM call).

        Substitui checklist-generate(LLM) + analyzer(LLM) por UMA
        chamada que alimenta os dois. Fallback 100% determinístico
        (sem segunda análise LLM). Retorna ids de problemas criados.
        Nunca levanta, nunca inventa problemas.
        """
        created: list[str] = []
        try:
            trace = self._trace_or_null()
            trace.record("error_analysis_started", iteration=iteration,
                         command=str(command or "")[:200])
            from app.agent.errors.error_analyzer import (
                UnifiedErrorAnalyzer,
            )

            planner = getattr(self, "planner", None)
            unified = UnifiedErrorAnalyzer(
                llm=getattr(planner, "llm", None)).analyze(
                    command, result, iteration)
            state = getattr(self, "task_state", None)
            invalidated: list[str] = []
            if state is not None and unified.problems:
                created, invalidated = state.add_analyzed_problems(
                    unified.problems, command, iteration)
            apply_fn = getattr(self.error_checklist, "apply_unified",
                               None)
            if apply_fn is None:
                try:
                    items = getattr(self.error_checklist, "_items",
                                    None) or []
                    for item in items:
                        if state is not None:
                            state.record_problem(
                                getattr(item, "description", ""),
                                kind="test_failure",
                                iteration=iteration)
                except Exception as error:
                    logger.warning("TaskState problem falhou: %s",
                                   error)
                projected = 0
            else:
                try:
                    projected = apply_fn(unified)
                except Exception as error:
                    logger.warning(
                        "Projeção do checklist falhou: %s", error)
                    projected = 0
            trace.record(
                "error_analysis_completed", iteration=iteration,
                problems=len(unified.problems),
                failures=len(unified.failures),
                checklist_items=len(unified.checklist_items),
                exit_code=unified.exit_code,
                created=list(created)[:20],
                invalidated=list(invalidated)[:20],
                fallback=unified.fallback_used,
                llm_calls=unified.llm_calls,
                error=(unified.error or "")[:120],
            )
            for pid in list(created)[:20]:
                trace.record("problem_created", iteration=iteration,
                             problem_id=pid)
            for pid in list(invalidated)[:20]:
                trace.record("problem_invalidated", iteration=iteration,
                             problem_id=pid)
            if created or invalidated or projected:
                self._save_task_state("analysis")
        except Exception as error:
            logger.warning("Análise unificada falhou: %s", error)
        return created

    def _emit_task_state_update(self, iteration) -> None:
        """Evento compacto de evolução do estado (sem snapshot gigante)."""
        try:
            state = getattr(self, "task_state", None)
            if state is None:
                return
            summary = state.progress_summary
            self._trace_or_null().record(
                "task_state_update", iteration=iteration,
                decisions=summary.get("decisions", 0),
                progress=f"{summary.get('succeeded', 0)}/"
                         f"{summary.get('total', 0)}",
                open_problems=len(state.open_problems),
            )
        except Exception as error:
            logger.warning("Trace update do TaskState falhou: %s", error)

    def _note_test_run(self, tool: str, arguments: dict, succeeded: bool) -> None:
        """Conta execuções de teste/build (só métrica)."""

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

    def _trace_or_null(self):
        """Trace da run atual (nunca None, nunca levanta)."""
        trace = getattr(self, "_trace", None)
        if trace is None:
            return NullTrace()
        return trace

    def _trace_tool_result(
        self,
        iteration,
        tool: str,
        arguments: dict,
        result,
        success: bool,
        dependency: bool = False,
    ) -> None:
        """Registra tool_result (+ test_result p/ teste/build) no trace.

        Observabilidade pura: só metadados e resumos truncados, nunca
        conteúdo integral (ver app.agent.trace). Não altera o fluxo.
        """

        trace = self._trace_or_null()
        try:
            is_test_build = (
                tool == "run_command"
                and self.operational_memory.is_test_or_build_command(
                    str((arguments or {}).get("command", ""))
                )
            )
        except Exception:
            is_test_build = False

        command = None
        if tool == "run_command" and isinstance(arguments, dict):
            command = truncate_text(
                str(arguments.get("command", "")), 500)

        trace.record(
            "tool_result",
            iteration=iteration,
            tool=tool,
            dependency=dependency,
            success=success,
            file_path=extract_file_path(arguments),
            arguments=sanitize_arguments(tool, arguments),
            command=command,
            exit_code=parse_exit_code(result),
            timeout=is_timeout_result(result),
            result_summary=sanitize_result(result),
        )

        if is_test_build:
            trace.record(
                "test_result",
                iteration=iteration,
                dependency=dependency,
                command=command,
                exit_code=parse_exit_code(result),
                timeout=is_timeout_result(result),
                success=success,
                result_summary=sanitize_result(result),
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
            result = f"INVESTIGATION ERROR: {tool_error}"

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
        self._trace_tool_result(
            iteration=iteration,
            tool=tool_name,
            arguments=task.arguments,
            result=result,
            success=investigation_succeeded,
            dependency=True,
        )

        return self._append_context_error(
            context,
            "INVESTIGATION PERFORMED AS A LAST RESORT (you insisted "
            f"on using '{tool_name}' as the main task even after "
            "several warnings — instead of crashing the run, the "
            "investigation was executed for you this time):\n\n"
            f"{tool_name}({task.arguments}) ->\n"
            f"{self._truncate_for_planner(result)}\n\n"
            "This is NOT a general permission: keep attaching "
            "read_file/list_files/find_references as a dependency of "
            "the real action, except right after a failed test/build. "
            "Now choose the next REAL ACTION using this information.",
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
            first_line = str(result).lstrip().split("\n", 1)[0]
            return first_line.startswith("STATUS: success")

        return True

    def _execute_dependency(self, dependency):
        """Executa UMA dependency, capturando erro.

        Retorna (result, succeeded, error_or_None) - exatamente a
        mesma semântica do loop sequencial legado: exceção da tool
        vira string "ERRO NA DEPENDENCY" + succeeded=False; comando
        com exit code != 0 é succeeded=False SEM exceção. Usado tanto
        pelo caminho sequencial quanto pelo batch paralelo.
        """

        try:
            result = self.tools.execute(
                dependency.tool,
                dependency.arguments,
            )
        except Exception as error:
            return (
                f"DEPENDENCY ERROR:\n"
                f"{type(error).__name__}: {error}",
                False,
                error,
            )

        return (
            result,
            self._command_succeeded(dependency.tool, result),
            None,
        )

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
            try:
                state = getattr(self, "task_state", None)
                if state is not None and state.open_problems:
                    state.resolve_problems(iteration)
                    state.record_correction(
                        f"tests green: {command[:120]}", iteration)
            except Exception as error:
                logger.warning("TaskState correction falhou: %s", error)
            self.error_checklist.clear()
            self._sync_task_plan()
            self._save_task_state("tests_green")
            return

        text = str(result).lstrip()

        if text.startswith("TIMEOUT:") or text.startswith(
            "TOOL EXECUTION ERROR:"
        ):
            reason = (
                "timed out"
                if text.startswith("TIMEOUT:")
                else "failed with an infrastructure error"
            )
            self.error_checklist.note_infra_failure(command, reason)
            try:
                state = getattr(self, "task_state", None)
                if state is not None:
                    state.record_problem(
                        f"infra failure: {command[:120]} ({reason})",
                        kind="infra", iteration=iteration)
            except Exception as error:
                logger.warning("TaskState problem falhou: %s", error)
            self._sync_task_plan()
            self._save_task_state("infra_failure")
            return

        if bool(getattr(Config, "error_analyzer", False)):
            self._analyze_test_failure_unified(
                command, result, iteration)
        else:
            try:
                self.error_checklist.generate(command, result, iteration)
            except Exception as error:
                logger.warning(
                    "Falha ao gerar checklist de erros: %s", error
                )
            try:
                state = getattr(self, "task_state", None)
                if state is not None:
                    items = getattr(
                        self.error_checklist, "_items", None) or []
                    for item in items:
                        state.record_problem(
                            getattr(item, "description", ""),
                            kind="test_failure", iteration=iteration)
            except Exception as error:
                logger.warning("TaskState problem falhou: %s", error)
        self._sync_task_plan()
        self._save_task_state("test_failure")

    def _is_mutating(self, tool: str, arguments: dict) -> bool:
        if tool in ("write_file", "edit_file", "delete_file"):
            return True

        if tool == "run_command":
            return self._command_mutates_files(
                str(arguments.get("command", ""))
            )

        return False

    def _summary_skip_reason(
        self, tool: str, arguments: dict
    ) -> str | None:
        """Motivo do skip do SummaryUpdater, ou None se deve atualizar.

        (congelada, só AIDEV_SMART_SUMMARY): leitura pura.
        (só AIDEV_COMPACT_EXECUTOR): operações comprovadamente
        sem mudança de estado - check_project (análise estática, o
        veredito já vai ao Planner via resultado + finish gate) e
        run_command que nem muta (MUTATION_PATTERNS) nem é teste/build
        (ex.: --version, ls; sem sinal de teste para o resumo). Tudo o
        que muda estado (write_file, edit_file, delete_file,
        run_command mutante, teste/build)
        atualiza normalmente. Nunca fabrica resumo: pular = manter o
        vigente.
        """
        if Config.smart_summary and (tool in self.READ_ONLY_TOOLS):
            return "read_only_no_state_change"

        if not bool(getattr(Config, "compact_executor", False)):
            return None

        if tool == "check_project":
            return "analysis_no_state_change"

        if tool == "run_command" and isinstance(arguments, dict):
            command = str(arguments.get("command", ""))
            if self._is_mutating(tool, arguments):
                return None
            try:
                is_test_build = (
                    self.operational_memory.is_test_or_build_command(
                        command)
                )
            except Exception:
                is_test_build = False
            if not is_test_build:
                return "inspect_no_state_change"

        return None

    def _truncate(self, text) -> str:
        text = str(text)

        if len(text) <= self.MAX_CONTEXT_CHARS:
            return text

        omitted = len(text) - self.MAX_CONTEXT_CHARS

        return (
            f"{text[:self.MAX_CONTEXT_CHARS]}\n"
            f"...[truncated, {omitted} characters omitted]"
        )


    MAX_COMPACT_RESULT_CHARS = 2000
    MAX_COMPACT_SUMMARY_CHARS = 2000
    MAX_ERROR_TAIL_BLOCKS = 2

    def _truncate_compact(self, text) -> str:
        """Head+tail com veredito preservado.

        Mantém o início (STATUS na 1ª linha do run_command, header
        PARENT TASK no task_context) e o fim (resumo do pytest, traceback
        final, última falha) - o meio omitido é log verboso. Marca a
        omissão com o total, nunca silenciosa. Textos <= 2000 voltam
        intactos (byte-idênticos ao legado).
        """
        text = str(text)
        limit = self.MAX_COMPACT_RESULT_CHARS
        if len(text) <= limit:
            return text
        head = 800
        tail = 1000
        omitted = len(text) - head - tail
        return (
            f"{text[:head]}\n"
            f"...[compacted, {omitted} characters omitted]\n"
            f"{text[-tail:]}"
        )

    def _truncate_for_planner(self, text) -> str:
        """Truncamento conforme a flag (nunca silencioso).

        Flag off: _truncate legado (head 4000). Flag on: _truncate_compact
        (head 800 + tail 1000, veredito preservado). Textos pequenos são
        byte-idênticos nos dois modos.
        """
        if bool(getattr(Config, "compact_planner", False)):
            try:
                return self._truncate_compact(text)
            except Exception as error:
                logger.warning(
                    "Truncamento compacto falhou (usando legado): %s",
                    error,
                )
        return self._truncate(text)

    def _truncate_summary_compact(self, summary) -> str:
        """Resumo narrativo capped: head + marcador."""
        text = str(summary or "")
        limit = self.MAX_COMPACT_SUMMARY_CHARS
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return (
            f"{text[:limit]}\n"
            f"...[summary compacted, {omitted} characters omitted]"
        )

    def _append_context_error(self, context: str, block: str) -> str:
        """Anexa bloco de erro ao contexto com teto de cauda.

        Legado (flag off): concatenação pura, como antes.
        Compacto: mantém no máximo MAX_ERROR_TAIL_BLOCKS blocos de
        erro na cauda - o mais antigo além do teto é descartado, com
        marcador explícito. Blocos considerados: VALIDATION ERROR,
        REPETITION*, PREVIOUS ATTEMPT CORRECTION, REPEATED ERROR,
        INVESTIGATION PERFORMED.
        """
        context = str(context or "")
        block = str(block or "")
        if not bool(getattr(Config, "compact_planner", False)):
            return f"{context}\n\n{block}" if context else block
        combined = f"{context}\n\n{block}" if context else block
        markers = (
            "\n\nVALIDATION ERROR",
            "\n\nREPETITION",
            "\n\nPREVIOUS ATTEMPT CORRECTION",
            "\n\nREPEATED ERROR",
            "\n\nINVESTIGATION PERFORMED",
        )
        starts: list[int] = []
        for marker in markers:
            pos = 0
            while True:
                idx = combined.find(marker, pos)
                if idx == -1:
                    break
                starts.append(idx + 2)
                pos = idx + 2
        starts.sort()
        if len(starts) <= self.MAX_ERROR_TAIL_BLOCKS:
            return combined
        keep_from = starts[-(self.MAX_ERROR_TAIL_BLOCKS):][0]
        dropped = len(starts) - self.MAX_ERROR_TAIL_BLOCKS
        prefix = combined[:starts[0]]
        tail = combined[keep_from:]
        return (
            f"{prefix}"
            f"...[{dropped} earlier error block(s) omitted]\n\n"
            f"{tail}"
        )

    def _build_memory_block(
        self, project_name: str, summary: str, files_text: str | None = None
    ) -> str:
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

        files_text: seção de arquivos pré-computada (ex.:
        linha compacta "sem alterações"). None = listagem integral.
        """

        error_block = self.error_checklist.render()

        if files_text is None:
            operational_text = self.operational_memory.render(
                project_name)
        else:
            operational_text = self.operational_memory.render(
                project_name, files_text)

        return (
            f"PROJECT SUMMARY:\n"
            f"{summary}\n\n"
            f"{self.checklist.render()}\n\n"
            + (f"{error_block}\n\n" if error_block else "")
            + f"{self.planner_error_memory.render()}\n\n"
            + f"{operational_text}"
        )

    def _build_memory_block_compact(
        self, project_name: str, summary: str, files_text: str | None = None
    ) -> str:
        """Bloco de memória compacto p/ o Planner.

        Mesmas seções do legado (resumo, checklist, erros, memória),
        com: resumo capped em 2000 (head), histórico em janela de 8 +
        falhas preservadas, erros proibidos nos últimos 5. Checklist
        do objetivo e de erros vão integrais (pequenos e decisivos).
        Doubles legados sem render_compact caem para o legado.
        """
        error_block = self.error_checklist.render()

        render_compact_fn = getattr(
            self.operational_memory, "render_compact", None)
        if files_text is None:
            if render_compact_fn is not None:
                try:
                    operational_text = render_compact_fn(project_name)
                except TypeError:
                    operational_text = self.operational_memory.render(
                        project_name)
            else:
                operational_text = self.operational_memory.render(
                    project_name)
        else:
            if render_compact_fn is not None:
                try:
                    operational_text = render_compact_fn(
                        project_name, files_text)
                except TypeError:
                    operational_text = self.operational_memory.render(
                        project_name, files_text)
            else:
                try:
                    operational_text = self.operational_memory.render(
                        project_name, files_text)
                except TypeError:
                    operational_text = self.operational_memory.render(
                        project_name)

        render_err_fn = getattr(
            self.planner_error_memory, "render_compact", None)
        if render_err_fn is not None:
            try:
                planner_errors_text = render_err_fn()
            except Exception:
                planner_errors_text = self.planner_error_memory.render()
        else:
            planner_errors_text = self.planner_error_memory.render()

        return (
            f"PROJECT SUMMARY:\n"
            f"{self._truncate_summary_compact(summary)}\n\n"
            f"{self.checklist.render()}\n\n"
            + (f"{error_block}\n\n" if error_block else "")
            + f"{planner_errors_text}\n\n"
            + f"{operational_text}"
        )

    def _build_memory_block_tracked(
        self, project_name: str, summary: str
    ) -> str:
        """Bloco de memória com listagem estrutural.

        Quando AIDEV_COMPACT_CONTEXT=1, a lista integral de arquivos só
        é reenviada se a estrutura mudou desde a última verificação
        (hash da listagem, mantido em `self._file_list_state` e zerado
        por run); senão vai uma linha explícita "sem alterações".
        Comportamento legado caso contrário. Nunca altera decisões -
        só o volume reenviado. O bloco compacto ou o integral segue
        a flag AIDEV_COMPACT_PLANNER.
        Doubles legados sem os métodos novos caem para o legado.
        """

        use_compact = bool(getattr(Config, "compact_planner", False))

        def _block(project_name: str, summary: str, files_text=None) -> str:
            if use_compact:
                try:
                    if files_text is None:
                        return self._build_memory_block_compact(
                            project_name, summary)
                    return self._build_memory_block_compact(
                        project_name, summary, files_text)
                except Exception as error:
                    logger.warning(
                        "Bloco compacto falhou (usando integral): %s",
                        error,
                    )
            if files_text is None:
                return self._build_memory_block(project_name, summary)
            try:
                return self._build_memory_block(
                    project_name, summary, files_text)
            except TypeError:
                return self._build_memory_block(project_name, summary)

        if not Config.compact_context:
            return _block(project_name, summary)
        render_state_fn = getattr(
            self.operational_memory, "render_files_state", None)
        if render_state_fn is None:
            return _block(project_name, summary)
        last_state = getattr(self, "_file_list_state", None)
        files_text, new_state = (
            self.operational_memory.render_files_state(
                project_name, last_state)
        )
        self._file_list_state = new_state
        return _block(project_name, summary, files_text)

    def _note_tool_execution(self, tool_name: str) -> None:
        """Invalida o cache do finish-check após escrita/efeito.

        Leituras puras preservam o cache: o veredito do check_project é
        função determinística dos fontes no disco. O próprio check_project
        também não invalida (sintaxe/compilação só-leitura; artefatos vão
        para diretórios ignorados como __pycache__/target).
        """

        if tool_name not in self.READ_ONLY_TOOLS:
            self._finish_check_cache = self._FINISH_CHECK_MISS

    def _cached_finish_check(self, project_name: str) -> str | None:
        """check_project com cache entre finishs consecutivos.

        Um finish bloqueado dá `continue` sem executar nada; o próximo
        finish revalidaria arquivos idênticos (N subprocess/containers).
        A verificação LLM (FinalVerification, estocástica) continua
        sempre fresca - só o veredito determinístico é reutilizado.
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
            success="FAILED" not in result,
        )

        if "FAILED" in result:
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
            self._last_final_verification_status = "error"
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
            self._last_final_verification_status = "unavailable"
            return None

        if result.is_problems:
            self._last_final_verification_status = "problems_found"
            return result.report

        self._last_final_verification_status = "ok"
        return None

    def _run_unnecessary_files_cleanup(
        self, project_name: str, iteration,
    ) -> str | None:
        """Remove arquivos claramente desnecessários antes do finish.

        Dimensão "arquivos desnecessários" da verificação final:
        análise determinística (sem LLM obrigatória) classifica
        candidatos em SAFE / UNCERTAIN / KEEP. Apenas SAFE são
        removidos via delete_file (que aplica suas próprias proteções);
        UNCERTAIN permanecem. Após remover, revalida com check_project
        (fresco, sem cache): se a remoção quebrou algo, bloqueia o
        finish para correção. Uma única passada por finish (sem loops).

        Retorna o relatório de erro se algo falhou, ou None se OK /
        nada a fazer / detecção indisponível (nunca bloqueia por causa
        da ferramenta de detecção em si).
        """
        from app.agent.context.unnecessary_files import (
            MAX_SAFE_DELETIONS_PER_SCAN,
        )

        trace = self._trace_or_null()

        try:
            detect = getattr(
                self.final_verification,
                "detect_unnecessary_files", None,
            )
            if detect is None:
                return None
            report = detect(
                project_name=project_name,
                tools_execute=self.tools.execute,
            )
        except Exception as error:
            logger.warning(
                "Limpeza de arquivos desnecessários indisponível: %s",
                error,
            )
            return None

        try:
            safe_paths = list(getattr(report, "safe_paths", []) or [])
            files_scanned = int(getattr(report, "files_scanned", 0) or 0)
            candidates = list(getattr(report, "candidates", []) or [])
            uncertain_count = sum(
                1 for c in candidates
                if getattr(c, "verdict", "") == "uncertain"
            )
        except Exception:
            return None

        try:
            trace.record(
                "unnecessary_files_scan",
                iteration=iteration,
                files_scanned=files_scanned,
                safe_paths=safe_paths[:MAX_SAFE_DELETIONS_PER_SCAN],
                safe_count=len(safe_paths),
                uncertain_count=uncertain_count,
                candidates=[
                    {
                        "path": getattr(c, "path", "?"),
                        "verdict": getattr(c, "verdict", "?"),
                        "evidences": list(
                            getattr(c, "evidences", []) or [])[:8],
                    }
                    for c in candidates
                ][:30],
            )
        except Exception as error:
            logger.warning("Trace do scan falhou: %s", error)

        self._note_state_verification(
            "unnecessary_files",
            True,
            (f"{files_scanned} files, {len(safe_paths)} safe, "
             f"{uncertain_count} uncertain"),
            iteration)

        if not safe_paths:
            return None

        removed: list[str] = []
        for file_path in safe_paths[:MAX_SAFE_DELETIONS_PER_SCAN]:
            arguments = {
                "project_name": project_name,
                "file_path": file_path,
            }
            self._mark_correction_attempt(
                iteration, "delete_file", arguments)
            self._emit(
                "tool_start", name="delete_file", arguments=arguments)
            try:
                result = self.tools.execute("delete_file", arguments)
            except Exception as error:
                result = (
                    "TOOL EXECUTION ERROR:\n"
                    f"{type(error).__name__}: {error}"
                )
                succeeded = False
                self._emit(
                    "tool_error", name="delete_file",
                    error=str(error))
                self._mark_attempt_failed("delete_file", arguments)
            else:
                succeeded = self._command_succeeded(
                    "delete_file", result)
                self._emit(
                    "tool_end", name="delete_file", success=succeeded)
                if succeeded:
                    removed.append(file_path)
                    self._note_correction_applied(
                        iteration, "delete_file", arguments)
            try:
                self.operational_memory.record(
                    iteration=iteration,
                    tool="delete_file",
                    arguments=arguments,
                    result=result,
                    success=succeeded,
                    dependency=False,
                )
            except Exception as error:
                logger.warning("Memória da limpeza falhou: %s", error)
            self._note_tool_execution("delete_file")
            self._trace_tool_result(
                iteration=iteration,
                tool="delete_file",
                arguments=arguments,
                result=result,
                success=succeeded,
                dependency=False,
            )
            try:
                state = getattr(self, "task_state", None)
                if state is not None:
                    state.record_progress(
                        iteration, "delete_file", succeeded,
                        file_path=file_path,
                        note=("cleanup: unnecessary file (SAFE) "
                              "removed in final verification"
                              if succeeded else
                              "cleanup: failed to remove file"),
                    )
                    self._save_task_state("cleanup")
            except Exception as error:
                logger.warning("TaskState da limpeza falhou: %s", error)
            try:
                trace.record(
                    "unnecessary_file_removed" if succeeded
                    else "unnecessary_file_remove_failed",
                    iteration=iteration,
                    file_path=file_path,
                    success=succeeded,
                )
            except Exception as error:
                logger.warning("Trace da limpeza falhou: %s", error)

        if not removed:
            return None

        check_error = self._run_finish_check(project_name)
        self._finish_check_cache = check_error
        self._note_state_verification(
            "unnecessary_files_recheck",
            check_error is None,
            str(check_error)[:200] if check_error else (
                f"{len(removed)} file(s) removed, "
                "check_project OK"),
            iteration)
        trace.record(
            "finish_gate",
            iteration=iteration,
            gate="unnecessary_files_recheck",
            passed=check_error is None,
            removed=removed,
        )
        return check_error

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
        self._last_final_verification_status = "not_run"
        self._file_list_state = None
        run_start = time.monotonic()

        trace = self.execution_trace
        if trace is None:
            try:
                trace = ExecutionTrace()
            except Exception as error:
                logger.warning("Trace desativado (falha ao criar): %s",
                               error)
                trace = NullTrace()
        self._trace = trace
        trace.record(
            "run_start",
            objective=truncate_text(objective, 2000),
            project_name=project_name,
            max_iterations=self.max_iterations,
        )

        try:
            summary = self.project_context.initialize(
                project_name
            )
        except Exception as error:
            error_type, error_signature = error_type_and_signature(error)
            trace.record(
                "run_error",
                outcome="failure",
                phase="init",
                error_type=error_type,
                error_signature=error_signature,
            )
            raise

        self.operational_memory.reset()
        self.checklist.reset()
        self.error_checklist.reset()
        self.planner_error_memory.reset()
        self._task_state_project = project_name
        self._task_state_saved_sig = None
        self.task_state = self._init_task_state(objective,
                                                project_name)
        try:
            trace.record("task_state_init",
                         state=self.task_state.to_dict())
        except Exception as error:
            logger.warning("Trace do TaskState falhou: %s", error)
        self._save_task_state("init", force=True)
        downstream_objective = self._downstream_objective(objective)
        self._current_objective = downstream_objective
        self._sync_task_plan()

        try:
            self.checklist.generate(downstream_objective)
        except Exception as error:
            logger.warning(
                "Falha ao gerar checklist do objetivo: %s", error
            )
        self._sync_task_plan()

        context = (
            f"{self._build_memory_block_tracked(project_name, summary)}\n\n"
            f"{context}"
        )

        task_history = deque(maxlen=self.MAX_TASK_HISTORY)
        succeeded_mutations: dict = {}
        progress_epoch = 0
        stagnant_iterations = 0
        planner_retry_context = ""
        iteration = 0

        can_short_repair = (
            hasattr(self.planner, "plan_with_prompt")
            and hasattr(self.planner, "build_repair_prompt")
        )

        while True:
            iteration += 1

            trace.record("iteration_start", iteration=iteration)

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

                trace.record(
                    "run_error",
                    outcome="failure",
                    phase="max_iterations",
                    iteration=iteration,
                    error_type="RuntimeError",
                    error_signature=truncate_text(error, 500),
                )

                raise RuntimeError(error)

            planner_attempts = 0
            allow_investigation = False
            used_investigation_budget = False
            planner_retry_context = ""
            short_repair = None

            while True:
                planner_attempts += 1
                decision = None

                self._emit("planner_start")
                self._stats.planner_calls += 1

                try:
                    if short_repair is not None and can_short_repair:
                        decision = self.planner.plan_with_prompt(
                            prompt=short_repair["prompt"],
                            iteration=iteration,
                            request_type="short_repair",
                        )
                    else:
                        retry_context = (
                            f"{context}\n\n"
                            f"{planner_retry_context}"
                            if planner_retry_context
                            else context
                        )
                        decision = self.planner.plan(
                            objective=downstream_objective,
                            context=self._with_task_state(
                                retry_context),
                            iteration=iteration,
                            request_type=(
                                "normal" if planner_attempts == 1
                                else "full_retry"
                            ),
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

                    error_type, error_sig = error_type_and_signature(error)
                    trace.record(
                        "planner_error",
                        iteration=iteration,
                        planner_attempt=planner_attempts,
                        error_type=error_type,
                        error_signature=error_sig,
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
                            trace.record(
                                "planner_recovery",
                                iteration=iteration,
                                tool=(decision.task.tool
                                      if decision is not None
                                      and decision.task is not None
                                      else None),
                                file_path=(
                                    extract_file_path(
                                        decision.task.arguments)
                                    if decision is not None
                                    and decision.task is not None
                                    else None),
                            )
                            continue

                        trace.record(
                            "run_error",
                            outcome="failure",
                            phase="planner_attempts_exceeded",
                            iteration=iteration,
                            error_type="RuntimeError",
                            error_signature=(
                                "O Planner excedeu o limite de tentativas."
                            ),
                        )
                        raise RuntimeError(
                            "O Planner excedeu o limite de tentativas."
                        ) from error

                    failed_was_short = short_repair is not None
                    short_repair = None

                    if can_short_repair and not failed_was_short:
                        repair_prompt = None
                        try:
                            failed_tool = (
                                decision.task.tool
                                if decision is not None
                                and decision.task is not None
                                else None
                            )
                            repair_prompt = (
                                self.planner.build_repair_prompt(
                                    error=error_signature,
                                    raw_response=getattr(
                                        self.planner,
                                        "last_raw_response",
                                        None,
                                    ),
                                    tool_name=failed_tool,
                                )
                            )
                        except Exception as repair_error:
                            logger.warning(
                                "Falha ao montar short repair prompt "
                                "(usando retry completo): %s",
                                repair_error,
                            )
                            repair_prompt = None

                        if repair_prompt is not None:
                            short_repair = {"prompt": repair_prompt}
                            trace.record(
                                "planner_retry",
                                iteration=iteration,
                                planner_attempt=planner_attempts,
                                retry_type="short_repair",
                                reason=error_sig,
                                prompt_chars=len(repair_prompt),
                            )
                            continue

                    if already_forbidden:
                        planner_retry_context = (
                            "REPEATED ERROR — READ CAREFULLY:\n"
                            f"{type(error).__name__}: {error}\n\n"
                            "You ALREADY made this EXACT mistake before "
                            "in this run and are forbidden from repeating "
                            "it. The previous fix attempt did not "
                            "work — do not repeat the same decision "
                            "again. Change the tool, the arguments, or "
                            "the approach differently from the previous "
                            "attempt.\n"
                            "Do not execute tools.\n"
                            "Return only the decision JSON "
                            "expected by the Planner."
                        )
                    else:
                        planner_retry_context = (
                            "PREVIOUS ATTEMPT CORRECTION:\n"
                            f"{type(error).__name__}: {error}\n\n"
                            "The previous response was invalid.\n"
                            "Do not execute tools.\n"
                            "Return only the decision JSON expected by the Planner."
                        )

                    if can_short_repair:
                        full_chars = self.planner.full_prompt_chars(
                            downstream_objective,
                            self._with_task_state(
                                f"{context}\n\n{planner_retry_context}"
                                if planner_retry_context
                                else context
                            ),
                        )
                    else:
                        full_chars = len(downstream_objective or "") + len(
                            context or "") + len(planner_retry_context)
                    trace.record(
                        "planner_retry",
                        iteration=iteration,
                        planner_attempt=planner_attempts,
                        retry_type="full_context",
                        reason=error_sig,
                        prompt_chars=full_chars,
                    )

                    continue

                self._emit(
                    "planner_end",
                    action=decision.action.value,
                )

                if decision.action == DecisionAction.TASK:
                    trace.record(
                        "planner_decision",
                        iteration=iteration,
                        planner_attempt=planner_attempts,
                        decision=decision.action.value,
                        tool=decision.task.tool,
                        file_path=extract_file_path(
                            decision.task.arguments),
                        dependencies=len(decision.task.dependencies),
                        checklist_progress=decision.checklist_progress,
                    )
                else:
                    trace.record(
                        "planner_decision",
                        iteration=iteration,
                        planner_attempt=planner_attempts,
                        decision=decision.action.value,
                        checklist_progress=decision.checklist_progress,
                    )

                self._note_task_decision(iteration, decision)

                break

            marked = self.checklist.mark_done(decision.checklist_progress)
            self._sync_task_plan()

            if marked:
                self._emit("checklist_updated", marked=marked)

            if decision.action == DecisionAction.TASK:
                gate_message = self._check_test_gate(
                    decision.task, iteration)
                if gate_message is not None:
                    self._emit(
                        "planner_error",
                        error=(
                            "Teste bloqueado: há problemas conhecidos "
                            "ainda não corrigidos."
                        ),
                    )
                    trace.record(
                        "test_blocked_pending_problems",
                        iteration=iteration,
                        tool=decision.task.tool,
                        command=self._gate_command(decision.task),
                        problems=self._blocking_ids(),
                    )
                    self.operational_memory.record(
                        iteration=iteration,
                        tool=decision.task.tool,
                        arguments=decision.task.arguments,
                        result=gate_message,
                        success=False,
                        dependency=False,
                    )
                    self._stats.summary_skipped += 1
                    trace.record(
                        "summary_skipped",
                        iteration=iteration,
                        tool=decision.task.tool,
                        reason="test_blocked_no_state_change",
                        file_path=extract_file_path(
                            decision.task.arguments),
                    )
                    context = self._append_context_error(
                        context, gate_message)
                    continue

            if decision.action == DecisionAction.FINISH:
                trace.record("finish_requested", iteration=iteration)
                check_error = self._cached_finish_check(project_name)
                trace.record(
                    "finish_gate",
                    iteration=iteration,
                    gate="check_project",
                    passed=check_error is None,
                )
                self._note_state_verification(
                    "check_project", check_error is None,
                    str(check_error)[:200] if check_error else "",
                    iteration)

                if check_error:
                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas o "
                            "check_project encontrou problemas."
                        ),
                    )

                    context = self._append_context_error(
                        context,
                        f"VALIDATION ERROR BEFORE FINISH:\n"
                        "You tried to finish, but the check_project "
                        "check found problems that must be fixed "
                        "first:\n\n"
                        f"{self._truncate_for_planner(check_error)}\n\n"
                        "Fix the problems above before trying to "
                        "finish again.",
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    trace.record(
                        "finish_block",
                        iteration=iteration,
                        gate="check_project",
                        reason=(
                            "check_project encontrou problemas"
                        ),
                        report_summary=sanitize_result(check_error),
                    )

                    continue

                error_pending = self.error_checklist.pending_count
                trace.record(
                    "finish_gate",
                    iteration=iteration,
                    gate="error_checklist",
                    passed=error_pending == 0,
                    pending_count=error_pending,
                )
                self._note_state_verification(
                    "error_checklist", error_pending == 0,
                    f"{error_pending} pending" if error_pending else "",
                    iteration)

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

                    context = self._append_context_error(
                        context,
                        f"VALIDATION ERROR BEFORE FINISH:\n"
                        "You tried to finish, but the last "
                        "test/build run is still failing:\n\n"
                        f"{self._truncate_for_planner(error_block)}\n\n"
                        "Fix those failures and re-run the test/build, "
                        "confirming it passes, before trying to "
                        "finish again.",
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    trace.record(
                        "finish_block",
                        iteration=iteration,
                        gate="error_checklist",
                        reason=(
                            "falhas de teste/build ainda não corrigidas"
                        ),
                        pending_count=error_pending,
                    )

                    continue

                current_summary = self.project_context.summary.read(
                    project_name
                )
                final_check_error = self._run_final_verification(
                    project_name, current_summary,
                )
                trace.record(
                    "final_verification",
                    iteration=iteration,
                    status=self._last_final_verification_status,
                )
                trace.record(
                    "finish_gate",
                    iteration=iteration,
                    gate="final_verification",
                    passed=final_check_error is None,
                )
                self._note_state_verification(
                    "final_verification",
                    final_check_error is None,
                    str(final_check_error)[:200]
                    if final_check_error else "",
                    iteration)

                if final_check_error:
                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas a "
                            "verificação final encontrou problemas."
                        ),
                    )

                    context = self._append_context_error(
                        context,
                        f"VALIDATION ERROR BEFORE FINISH:\n"
                        "You tried to finish, but the final project "
                        "verification found problems that must be "
                        "fixed first:\n\n"
                        f"{self._truncate_for_planner(final_check_error)}\n\n"
                        "Fix the problems above before trying to "
                        "finish again.",
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    trace.record(
                        "finish_block",
                        iteration=iteration,
                        gate="final_verification",
                        reason=(
                            "verificação final encontrou problemas"
                        ),
                        report_summary=sanitize_result(
                            final_check_error),
                    )

                    continue

                cleanup_error = self._run_unnecessary_files_cleanup(
                    project_name, iteration,
                )

                if cleanup_error:
                    self._emit(
                        "planner_error",
                        error=(
                            "O Planner tentou finalizar, mas a remoção "
                            "de arquivos desnecessários quebrou a "
                            "checagem do projeto."
                        ),
                    )

                    context = self._append_context_error(
                        context,
                        f"VALIDATION ERROR BEFORE FINISH:\n"
                        "You tried to finish and the verification removed "
                        "clearly unnecessary file(s), but the re-check "
                        "(check_project) after the removal found problems "
                        "that must be fixed first:\n\n"
                        f"{self._truncate_for_planner(cleanup_error)}\n\n"
                        "Fix the problems above before trying to "
                        "finish again.",
                    )

                    task_history.clear()
                    stagnant_iterations = 0
                    self._stats.finish_blocks += 1

                    trace.record(
                        "finish_block",
                        iteration=iteration,
                        gate="unnecessary_files_recheck",
                        reason=(
                            "revalidação falhou após remover arquivos "
                            "desnecessários"
                        ),
                        report_summary=sanitize_result(
                            cleanup_error),
                    )

                    continue

                if self.checklist.pending_count:
                    pending_items = self.checklist.pending_items
                    trace.record(
                        "finish_checklist_pending",
                        iteration=iteration,
                        pending_count=self.checklist.pending_count,
                        pending_items=[
                            truncate_text(item.description, 200)
                            for item in pending_items
                        ],
                    )
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
                trace.record(
                    "run_end",
                    outcome="success",
                    iterations=iteration,
                    result_summary=truncate_text(
                        decision.content, 2000),
                )
                self._emit(
                    "agent_done",
                    usage=self.planner.llm.usage.summary(),
                    usage_breakdown=self.planner.llm.usage.breakdown(),
                    updater_summary=updater_summary_fn()
                    if updater_summary_fn
                    else None,
                    perf_summary=perf_summary,
                )
                self._save_task_state("run_end", force=True)
                return decision.content

            if decision.action == DecisionAction.FAIL:
                self._emit(
                    "agent_error",
                    error=decision.reason,
                )
                trace.record(
                    "run_error",
                    outcome="failure",
                    phase="agent_fail",
                    iteration=iteration,
                    error_type="RuntimeError",
                    error_signature=truncate_text(
                        decision.reason, 500),
                )
                self._save_task_state("agent_fail", force=True)
                raise RuntimeError(decision.reason)

            if decision.action != DecisionAction.TASK:
                trace.record(
                    "run_error",
                    outcome="failure",
                    phase="unknown_action",
                    iteration=iteration,
                    error_type="ValueError",
                    error_signature=truncate_text(
                        f"Ação desconhecida: {decision.action}", 500),
                )
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

            task_signature = self._task_signature(task)
            task_history.append(task_signature)

            loop_signatures = self._detect_loop(task_history)
            is_stagnant = stagnant_iterations > self.MAX_STAGNANT_ITERATIONS

            if loop_signatures is not None or is_stagnant:
                if is_stagnant and loop_signatures is None:
                    error = (
                        f"The Planner ran {stagnant_iterations} tasks "
                        "in a row without producing any real change."
                    )
                    detail = (
                        "The agent is only reading files and running "
                        "verification commands repeatedly, without "
                        "writing any new change. Stop investigating "
                        "and take the next action that is actually "
                        "needed, or use finish/fail if no progress is "
                        "possible."
                    )
                else:
                    error = (
                        "The Planner is stuck in a pattern of repeated "
                        "tasks without making progress."
                    )
                    tasks_desc = "\n".join(
                        f"- tool={tool}, args={self._truncate_for_planner(args)}"
                        for tool, args in (loop_signatures or [])
                    )
                    detail = (
                        f"Tasks involved in the loop:\n{tasks_desc}\n\n"
                        "Do not repeat any of these actions. Analyze "
                        "the current project state and choose a "
                        "different action, or use finish if the "
                        "objective is already done."
                    )

                self._emit("planner_error", error=error)

                trace.record(
                    "loop_detected",
                    iteration=iteration,
                    kind=("stagnant"
                          if is_stagnant and loop_signatures is None
                          else "repetition"),
                    detail=truncate_text(error, 500),
                )

                context = self._append_context_error(
                    context,
                    f"REPETITION/STAGNATION ERROR:\n"
                    f"{error}\n\n"
                    f"{detail}",
                )

                task_history.clear()
                stagnant_iterations = 0
                self._stats.loop_hits += 1

                continue

            dependency_results = []

            deps = list(task.dependencies)
            parallel_outcomes = None
            if (
                len(deps) >= 2
                and Config.parallel_tools
                and all_pure_read(d.tool for d in deps)
            ):
                try:
                    batch, batch_wall = run_concurrent(
                        [
                            (lambda d=d: self._execute_dependency(d))
                            for d in deps
                        ]
                    )
                    parallel_outcomes = [
                        (
                            item.value[0],
                            item.value[1],
                            item.value[2],
                        )
                        if item.success
                        else (
                            "PARALLEL BATCH ERROR:\n"
                            f"{type(item.error).__name__}: {item.error}",
                            False,
                            item.error,
                        )
                        for item in batch
                    ]
                    saved_ms = estimated_saved_ms(batch, batch_wall)
                    stats = getattr(self, "_stats", None)
                    if stats is not None:
                        stats.parallel_batches += 1
                        stats.parallel_ops += len(deps)
                        stats.parallel_saved_ms += saved_ms
                    trace.record(
                        "parallel_batch",
                        iteration=iteration,
                        context="dependencies",
                        size=len(deps),
                        wall_ms=round(batch_wall, 1),
                        saved_ms=round(saved_ms, 1),
                    )
                except Exception as batch_error:
                    logger.warning(
                        "Batch paralelo de dependencies falhou "
                        "(usando sequencial): %s",
                        batch_error,
                    )
                    parallel_outcomes = None

            if parallel_outcomes is None:
                stats = getattr(self, "_stats", None)
                if stats is not None:
                    stats.sequential_ops += len(deps)

            for index, dependency in enumerate(deps):

                self._emit(
                    "tool_start",
                    name=dependency.tool,
                    arguments=dependency.arguments,
                    dependency=True,
                )

                if parallel_outcomes is not None:
                    result, dependency_succeeded, error = (
                        parallel_outcomes[index]
                    )
                else:
                    result, dependency_succeeded, error = (
                        self._execute_dependency(dependency)
                    )

                if error is None:
                    self._emit(
                        "tool_end",
                        name=dependency.tool,
                        dependency=True,
                        success=dependency_succeeded,
                    )
                else:
                    self._emit(
                        "tool_error",
                        name=dependency.tool,
                        error=str(error),
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
                self._trace_tool_result(
                    iteration=iteration,
                    tool=dependency.tool,
                    arguments=dependency.arguments,
                    result=result,
                    success=dependency_succeeded,
                    dependency=True,
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
                    executor_context = self._with_task_state(
                        task_context, problem_detail=False)
                    problems_block = self._known_problems_block()
                    if problems_block:
                        executor_context = (
                            f"{executor_context}\n\n{problems_block}"
                            if executor_context.strip()
                            else problems_block
                        )
                    execution = self.task_decision_maker.decide(
                        objective=downstream_objective,
                        task=task,
                        context=executor_context,
                        iteration=iteration,
                    )

                except (ValueError, LLMInvalidResponseError) as error:
                    self._emit(
                        "executor_error",
                        error=str(error),
                    )

                    error_type, error_sig = error_type_and_signature(
                        error)
                    trace.record(
                        "executor_error",
                        iteration=iteration,
                        executor_attempt=executor_attempts,
                        error_type=error_type,
                        error_signature=error_sig,
                    )

                    if executor_attempts >= self.MAX_EXECUTOR_ATTEMPTS:
                        trace.record(
                            "run_error",
                            outcome="failure",
                            phase="executor_attempts_exceeded",
                            iteration=iteration,
                            error_type="RuntimeError",
                            error_signature=(
                                "O Executor excedeu o limite de "
                                "tentativas."
                            ),
                        )
                        raise RuntimeError(
                            "O Executor excedeu o limite de tentativas."
                        ) from error

                    task_context = (
                        f"{task_context}\n\n"
                        f"EXECUTION DECISION ERROR:\n"
                        f"{type(error).__name__}: {error}\n\n"
                        "The previous execution decision was invalid. "
                        "Analyze the error and try again, strictly "
                        "following the task rules."
                    )

                    continue

                if execution.tool != task.tool:
                    error = (
                        "The LLM tried to execute a different tool "
                        "from the one defined in the task."
                    )

                    self._emit(
                        "executor_error",
                        error=error,
                    )

                    trace.record(
                        "executor_error",
                        iteration=iteration,
                        executor_attempt=executor_attempts,
                        error_type="ToolMismatchError",
                        error_signature=truncate_text(error, 500),
                    )

                    if executor_attempts >= self.MAX_EXECUTOR_ATTEMPTS:
                        trace.record(
                            "run_error",
                            outcome="failure",
                            phase="executor_attempts_exceeded",
                            iteration=iteration,
                            error_type="RuntimeError",
                            error_signature=(
                                "O Executor excedeu o limite de "
                                "tentativas."
                            ),
                        )
                        raise RuntimeError(
                            "O Executor excedeu o limite de tentativas."
                        )

                    task_context = (
                        f"{task_context}\n\n"
                        f"EXECUTION DECISION ERROR:\n"
                        f"{error}\n\n"
                        f"Expected tool: {task.tool}\n"
                        f"Received tool: {execution.tool}\n\n"
                        "Fix the decision and try again."
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

                    error_type, error_sig = error_type_and_signature(
                        error)
                    trace.record(
                        "executor_error",
                        iteration=iteration,
                        executor_attempt=executor_attempts,
                        error_type=error_type,
                        error_signature=error_sig,
                    )

                    if executor_attempts >= self.MAX_EXECUTOR_ATTEMPTS:
                        trace.record(
                            "run_error",
                            outcome="failure",
                            phase="executor_attempts_exceeded",
                            iteration=iteration,
                            error_type="RuntimeError",
                            error_signature=(
                                "O Executor excedeu o limite de "
                                "tentativas."
                            ),
                        )
                        raise RuntimeError(
                            "O Executor excedeu o limite de tentativas."
                        ) from error

                    task_context = (
                        f"{task_context}\n\n"
                        f"ARGUMENT VALIDATION ERROR:\n"
                        f"{error}\n\n"
                        "Fix the arguments using exactly the parameter "
                        "names expected by the tool and try again."
                    )

                    continue

                self._emit(
                    "executor_end",
                    tool=execution.tool,
                )

                trace.record(
                    "executor_decision",
                    iteration=iteration,
                    executor_attempt=executor_attempts,
                    tool=execution.tool,
                    file_path=extract_file_path(execution.arguments),
                )

                break

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
                    "The Executor tried to repeat an operation that was "
                    "already executed successfully before."
                )

                self._emit("executor_error", error=error)

                trace.record(
                    "executor_error",
                    iteration=iteration,
                    executor_attempt=executor_attempts,
                    error_type="RepeatedMutationError",
                    error_signature=truncate_text(error, 500),
                )
                trace.record(
                    "loop_detected",
                    iteration=iteration,
                    kind="mutation_repeat",
                    detail=truncate_text(error, 500),
                )

                context = self._append_context_error(
                    context,
                    f"REPETITION ERROR:\n"
                    f"{error}\n\n"
                    f"Tool: {execution.tool}\n"
                    f"Arguments: {self._truncate_for_planner(execution.arguments)}\n\n"
                    "That exact operation was already executed "
                    "successfully before; redoing it is useless, the "
                    "changes already exist. Check the current state "
                    "(list_files/read_file) and choose the next action "
                    "that is really needed, or use finish if the "
                    "objective is already done.",
                )

                task_history.clear()
                stagnant_iterations = 0
                self._stats.loop_hits += 1

                continue

            self._mark_correction_attempt(
                iteration, execution.tool, execution.arguments)

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
                    f"TOOL EXECUTION ERROR:\n"
                    f"{type(error).__name__}: {error}"
                )
                execution_succeeded = False

                self._emit(
                    "tool_error",
                    name=execution.tool,
                    error=str(error),
                )

                self._mark_attempt_failed(
                    execution.tool, execution.arguments)

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
            self._trace_tool_result(
                iteration=iteration,
                tool=execution.tool,
                arguments=execution.arguments,
                result=result,
                success=execution_succeeded,
                dependency=False,
            )

            if not (
                execution_succeeded
                and self._is_mutating(execution.tool, execution.arguments)
            ):
                progress_epoch += 1

            if execution_succeeded:
                self._note_correction_applied(
                    iteration, execution.tool, execution.arguments)

            self._update_error_checklist(
                execution.tool,
                execution.arguments,
                result,
                execution_succeeded,
                iteration,
            )
            self._note_task_progress(
                iteration, execution.tool, execution.arguments,
                result, execution_succeeded)

            try:
                skip_reason = self._summary_skip_reason(
                    execution.tool, execution.arguments
                )
                if skip_reason is not None:
                    self._stats.summary_skipped += 1
                    trace.record(
                        "summary_skipped",
                        iteration=iteration,
                        tool=execution.tool,
                        reason=skip_reason,
                        file_path=extract_file_path(
                            execution.arguments),
                    )
                else:
                    summary = self.project_summary_updater.update(
                        objective=downstream_objective,
                        project_name=project_name,
                        task=task,
                        result=result,
                        iteration=iteration,
                    )
                    trace.record(
                        "summary_updated",
                        iteration=iteration,
                        tool=execution.tool,
                        file_path=extract_file_path(
                            execution.arguments),
                    )
            except Exception as error:
                error_type, error_sig = error_type_and_signature(error)
                trace.record(
                    "run_error",
                    outcome="failure",
                    phase="summary_updater",
                    iteration=iteration,
                    error_type=error_type,
                    error_signature=error_sig,
                )
                raise

            context = (
                f"{self._build_memory_block_tracked(project_name, summary)}\n\n"
                f"{self._truncate_for_planner(task_context)}\n\n"
                f"EXECUTION RESULT:\n"
                f"{self._truncate_for_planner(result)}"
            )
