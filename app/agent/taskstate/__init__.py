"""Fase 4 — estado estruturado da tarefa (TaskState).

Pipeline: prompt do usuário → canonicalização (EN) → TaskInterpreter
→ TaskState → fluxo atual do AIDev (inalterado nesta fase).

Divisão deliberada com as memórias existentes (sem duplicação):
- OperationalMemory: histórico cru das execuções (tool/args/resultado).
  O TaskState guarda só deltas compactos (1 linha por decisão/progresso)
  + referências (file_path, exit code) — nunca conteúdos integrais.
- ProjectChecklist: plano fixo em texto. O TaskState referencia o
  progresso (done/total) sem copiar as descrições.
- ProjectSummary: narrativa LLM por projeto. O TaskState não copia o
  resumo; guarda fatos observados/verificados estruturados.
- ErrorChecklist/PlannerErrorMemory: evidência detalhada. O TaskState
  guarda problemas (descrição curta + origem + resolved?) e correções.
- Trace: eventos crus. O TaskState é fotografado no trace (init),
  não o contrário.

Nesta fase o TaskState é construído, mantido e auditado (trace), mas
NÃO é injetado nos prompts do Planner/Executor — migração futura.
"""

from app.agent.taskstate.task_state import (
    BLOCKING_PROBLEM_STATUSES,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_COMPLETED,
    PLAN_STATUS_IN_PROGRESS,
    PLAN_STATUS_PENDING,
    PROBLEM_STATUS_BLOCKED,
    PROBLEM_STATUS_IN_PROGRESS,
    PROBLEM_STATUS_INVALIDATED,
    PROBLEM_STATUS_PENDING,
    PROBLEM_STATUS_PENDING_VERIFICATION,
    PROBLEM_STATUS_RESOLVED,
    Ambiguity,
    Correction,
    DecisionRecord,
    PlanItem,
    Problem,
    ProgressEntry,
    Requirement,
    TaskConstraint,
    TaskState,
    VerificationRecord,
)

__all__ = [
    "BLOCKING_PROBLEM_STATUSES",
    "PLAN_STATUS_BLOCKED",
    "PLAN_STATUS_COMPLETED",
    "PLAN_STATUS_IN_PROGRESS",
    "PLAN_STATUS_PENDING",
    "PROBLEM_STATUS_BLOCKED",
    "PROBLEM_STATUS_IN_PROGRESS",
    "PROBLEM_STATUS_INVALIDATED",
    "PROBLEM_STATUS_PENDING",
    "PROBLEM_STATUS_PENDING_VERIFICATION",
    "PROBLEM_STATUS_RESOLVED",
    "Ambiguity",
    "Correction",
    "DecisionRecord",
    "PlanItem",
    "Problem",
    "ProgressEntry",
    "Requirement",
    "TaskConstraint",
    "TaskState",
    "VerificationRecord",
]
