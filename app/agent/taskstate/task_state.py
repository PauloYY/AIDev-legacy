"""Fase 4 — TaskState: representação estruturada e persistente da tarefa.

Proveniência (nunca misturar):
- "user":       texto/span verbatim do prompt original do usuário.
- "canonical":  texto da versão canônica em inglês (tradução fiel).
- "interpreted": estrutura extraída deterministicamente do canônico
  (requisitos/restrições/ambiguidades detectáveis — sem invenção).
- "observed":   fato visto durante a execução (decisão, resultado,
  arquivo tocado, exit code).
- "verified":   fato confirmado por execução real (teste verde, gates
  de finish). Nada é "verified" por decisão do Planner.
"""

from dataclasses import asdict, dataclass, field


# Origens válidas (documentação executável; métodos validam).
ORIGINS = ("user", "canonical", "interpreted", "observed", "verified")


def _safe_iteration(value) -> int:
    """Iteração defensiva (None/lixo → 0, nunca levanta)."""
    try:
        return int(value)
    except Exception:
        return 0


def _safe_str(value, limit: int) -> str:
    """str() defensivo com teto (nunca levanta, nunca estoura)."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unrepresentable>"
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [+{omitted} chars]"


@dataclass
class Requirement:
    """Um requisito explícito do prompt (não inferido)."""

    text: str
    origin: str = "interpreted"
    # id estável dentro da run ("R1", "R2", ...).
    req_id: str = ""


@dataclass
class TaskConstraint:
    """Uma restrição explícita do prompt (não inferida)."""

    text: str
    origin: str = "interpreted"
    constraint_id: str = ""


@dataclass
class Ambiguity:
    """Ambiguidade detectável (marcador concreto, não palpite)."""

    text: str
    reason: str = ""
    origin: str = "interpreted"


# Status de item de plano (ciclo de vida; ver TaskState.sync_plan).
PLAN_STATUS_PENDING = "pending"
PLAN_STATUS_IN_PROGRESS = "in_progress"
PLAN_STATUS_COMPLETED = "completed"
PLAN_STATUS_BLOCKED = "blocked"
PLAN_STATUSES = (
    PLAN_STATUS_PENDING,
    PLAN_STATUS_IN_PROGRESS,
    PLAN_STATUS_COMPLETED,
    PLAN_STATUS_BLOCKED,
)


@dataclass
class PlanItem:
    """Item de plano (espelha progresso do checklist, sem copiar texto).

    Identidade estável: `index` (id numérico do checklist) + `item_id`
    (string estável, ex. "checklist-2"). `status` é a fonte de verdade;
    `done` espelha status == completed (compatibilidade).
    """

    index: int
    done: bool = False
    # Referência curta (ex.: "checklist #2"); a descrição canônica
    # continua no ProjectChecklist.
    ref: str = ""
    status: str = PLAN_STATUS_PENDING
    item_id: str = ""


@dataclass
class DecisionRecord:
    """Decisão do Planner (compacta: sem conteúdos de arquivo)."""

    iteration: int
    action: str = "task"
    tool: str | None = None
    file_path: str | None = None
    summary: str = ""
    origin: str = "observed"


@dataclass
class ProgressEntry:
    """Progresso de UMA execução (1 linha; detalhe no histórico/trace)."""

    iteration: int
    tool: str = ""
    success: bool = True
    file_path: str | None = None
    note: str = ""
    origin: str = "observed"


# Status de problema (Fase 5). `resolved` (bool legado) espelha
# status == resolved; `open` p/ gate = pending/in_progress/blocked.
PROBLEM_STATUS_PENDING = "pending"
PROBLEM_STATUS_IN_PROGRESS = "in_progress"
PROBLEM_STATUS_PENDING_VERIFICATION = "pending_verification"
PROBLEM_STATUS_RESOLVED = "resolved"
PROBLEM_STATUS_BLOCKED = "blocked"
PROBLEM_STATUS_INVALIDATED = "invalidated"
PROBLEM_STATUSES = (
    PROBLEM_STATUS_PENDING,
    PROBLEM_STATUS_IN_PROGRESS,
    PROBLEM_STATUS_PENDING_VERIFICATION,
    PROBLEM_STATUS_RESOLVED,
    PROBLEM_STATUS_BLOCKED,
    PROBLEM_STATUS_INVALIDATED,
)
# Gate de retest: bloqueia enquanto houver algum destes.
BLOCKING_PROBLEM_STATUSES = frozenset({
    PROBLEM_STATUS_PENDING,
    PROBLEM_STATUS_IN_PROGRESS,
    PROBLEM_STATUS_BLOCKED,
})


@dataclass
class Problem:
    """Problema estruturado (Fase 5: hipótese do Analyzer + estado).

    `description` resume o erro (compat Fase 4); `error` é a linha
    observada; causa/solução são HIPÓTESES ("unknown"/"investigate"
    quando sem evidência). `test` = identificador do teste p/ dedup
    conservadora. `resolved` espelha status == resolved.
    """

    description: str
    kind: str = "test_failure"
    iteration: int | None = None
    resolved: bool = False
    origin: str = "observed"
    # --- Fase 5 (defaults seguros; ditados antigos carregam) ---
    problem_id: str = ""
    source: str = ""
    error: str = ""
    probable_cause: str = "unknown"
    suggested_solution: str = "investigate"
    affected_files: list[str] = field(default_factory=list)
    status: str = PROBLEM_STATUS_PENDING
    correction_id: str = ""
    test: str = ""

    def __post_init__(self):
        try:
            if not isinstance(self.affected_files, list):
                self.affected_files = list(self.affected_files or [])
        except Exception:
            self.affected_files = []
        try:
            if self.status not in PROBLEM_STATUSES:
                self.status = PROBLEM_STATUS_PENDING
            # Compat nos dois sentidos.
            if self.status == PROBLEM_STATUS_RESOLVED:
                self.resolved = True
            elif self.resolved and self.status == PROBLEM_STATUS_PENDING:
                self.status = PROBLEM_STATUS_RESOLVED
        except Exception:
            pass


@dataclass
class Correction:
    """Correção aplicada pelo Executor, vinculada a problema(s).

    "applied" = Executor aplicou (NÃO significa teste verde; a
    confirmação vem da verificação). `problem_ids` = vinculados;
    `problem_id` = primeiro (compat).
    """

    description: str
    iteration: int | None = None
    origin: str = "observed"
    # --- Fase 5 (defaults seguros) ---
    correction_id: str = ""
    problem_id: str = ""
    problem_ids: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    status: str = "applied"

    def __post_init__(self):
        try:
            if not isinstance(self.problem_ids, list):
                self.problem_ids = list(self.problem_ids or [])
            if not self.problem_id and self.problem_ids:
                first = self.problem_ids[0]
                self.problem_id = first if isinstance(
                    first, str) else str(first)
            if self.problem_id and self.problem_id not in (
                    self.problem_ids):
                self.problem_ids = [self.problem_id] + self.problem_ids
        except Exception:
            self.problem_ids = []
        try:
            if not isinstance(self.files_changed, list):
                self.files_changed = list(self.files_changed or [])
        except Exception:
            self.files_changed = []


@dataclass
class VerificationRecord:
    """Veredicto de verificação (gates de finish)."""

    gate: str = ""
    passed: bool = False
    detail: str = ""
    iteration: int | None = None
    origin: str = "verified"


@dataclass
class TaskState:
    """Estado estruturado da tarefa (Fase 4).

    Campos de entrada (prompt → canonical → interpreter) são escritos
    uma vez na construção; campos de execução evoluem via métodos
    record_* centralizados (chamados pelo Runner). Todos os métodos
    são defensivos: nunca levantam por entrada ruim.
    """

    # --- entrada (escritos uma vez) ---
    original_prompt: str = ""
    canonical_prompt: str = ""
    language: str = "unknown"
    translation_applied: bool = False
    # task_id: sha1 do canonical_prompt (recuperação segura: mesma
    # tarefa → pode recuperar; tarefa diferente → estado novo).
    task_id: str = ""
    objective: str = ""
    requirements: list[Requirement] = field(default_factory=list)
    constraints: list[TaskConstraint] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)
    plan: list[PlanItem] = field(default_factory=list)

    # --- execução (evoluem via record_*) ---
    decisions: list[DecisionRecord] = field(default_factory=list)
    progress: list[ProgressEntry] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    verification: list[VerificationRecord] = field(default_factory=list)

    # Tetos da renderização compacta (só representação, não os dados).
    MAX_ITEMS_PER_SECTION = 10
    MAX_TEXT_CHARS = 200

    # ----- atualizações centralizadas (Runner chama estes) -----

    def record_decision(
        self,
        iteration: int,
        action: str,
        tool: str | None = None,
        file_path: str | None = None,
        summary: str = "",
    ) -> None:
        try:
            self.decisions.append(DecisionRecord(
                iteration=_safe_iteration(iteration),
                action=_safe_str(action, 20),
                tool=_safe_str(tool, 60) if tool else None,
                file_path=_safe_str(file_path, 200) if file_path else None,
                summary=_safe_str(summary, self.MAX_TEXT_CHARS),
            ))
        except Exception:
            pass

    def record_progress(
        self,
        iteration: int,
        tool: str,
        success: bool,
        file_path: str | None = None,
        note: str = "",
    ) -> None:
        try:
            self.progress.append(ProgressEntry(
                iteration=_safe_iteration(iteration),
                tool=_safe_str(tool, 60),
                success=bool(success),
                file_path=_safe_str(file_path, 200) if file_path else None,
                note=_safe_str(note, self.MAX_TEXT_CHARS),
            ))
        except Exception:
            pass

    def record_problem(
        self,
        description: str,
        kind: str = "test_failure",
        iteration: int | None = None,
    ) -> None:
        try:
            if description is None:
                return
            text = _safe_str(description, self.MAX_TEXT_CHARS)
            if not text.strip():
                return
            self.problems.append(Problem(
                description=text,
                kind=_safe_str(kind, 40),
                iteration=(_safe_iteration(iteration)
                           if iteration is not None else None),
            ))
        except Exception:
            pass

    def record_correction(
        self,
        description: str,
        iteration: int | None = None,
    ) -> None:
        try:
            if description is None:
                return
            text = _safe_str(description, self.MAX_TEXT_CHARS)
            if not text.strip():
                return
            self.corrections.append(Correction(
                description=text,
                iteration=(_safe_iteration(iteration)
                           if iteration is not None else None),
            ))
        except Exception:
            pass

    def resolve_problems(self, iteration: int | None = None) -> int:
        """Marca problemas abertos como resolvidos. Retorna quantos."""
        resolved = 0
        try:
            for problem in self.problems:
                if problem.status in BLOCKING_PROBLEM_STATUSES or (
                        problem.status
                        == PROBLEM_STATUS_PENDING_VERIFICATION):
                    problem.status = PROBLEM_STATUS_RESOLVED
                    problem.resolved = True
                    resolved += 1
        except Exception:
            pass
        return resolved

    # ----- Fase 5: ciclo de correção (métodos centralizados) -----

    @staticmethod
    def _problem_key(source: str, test: str,
                     error: str) -> tuple[str, str, str]:
        from app.agent.errors.error_analyzer import normalize_error

        try:
            src = (source if isinstance(source, str) else str(
                source or ""))[:120]
        except Exception:
            src = ""
        try:
            tst = (test if isinstance(test, str) else str(
                test or ""))[:200]
        except Exception:
            tst = ""
        return (src, tst, normalize_error(error))

    def _next_problem_id(self) -> str:
        try:
            return f"p{len(self.problems) + 1:03d}"
        except Exception:
            return "p000"

    def _next_correction_id(self) -> str:
        try:
            return f"c{len(self.corrections) + 1:03d}"
        except Exception:
            return "c000"

    def add_analyzed_problems(
        self,
        items: list | None,
        source: str = "",
        iteration: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """Incorpora problemas do Analyzer (dedup conservadora).

        Mesma (source, test, erro normalizado) aberta → mantém (sem
        duplicar). Mesmo teste com erro DIFERENTE → invalida o antigo
        e cria novo. Retorna (ids criados, ids invalidados). Nunca
        levanta.
        """
        created: list[str] = []
        invalidated: list[str] = []
        try:
            if not items:
                return (created, invalidated)
            try:
                it = (_safe_iteration(iteration)
                      if iteration is not None else None)
            except Exception:
                it = None
            for item in list(items)[:20]:
                try:
                    error = getattr(item, "error", "") or ""
                    test = getattr(item, "test", "") or ""
                    key = self._problem_key(source, test, error)
                    if not key[2]:
                        continue
                    tracked_statuses = (
                        PROBLEM_STATUS_PENDING,
                        PROBLEM_STATUS_IN_PROGRESS,
                        PROBLEM_STATUS_BLOCKED,
                        PROBLEM_STATUS_PENDING_VERIFICATION,
                    )
                    tracked = [
                        p for p in self.problems
                        if p.status in tracked_statuses
                    ]
                    # Re-evidência exata → mantém (reabre se aguardava).
                    exact = [p for p in tracked
                             if self._problem_key(
                                 p.source, p.test, p.error) == key]
                    if exact:
                        for problem in exact:
                            if problem.status == (
                                    PROBLEM_STATUS_PENDING_VERIFICATION):
                                problem.status = PROBLEM_STATUS_PENDING
                                problem.correction_id = ""
                        continue
                    # Mesmo teste (identificado), erro diferente →
                    # diagnóstico antigo não vale mais.
                    if key[1]:
                        for problem in tracked:
                            if problem.test == key[1]:
                                problem.status = (
                                    PROBLEM_STATUS_INVALIDATED)
                                if problem.problem_id:
                                    invalidated.append(
                                        problem.problem_id)
                    description = _safe_str(
                        error, self.MAX_TEXT_CHARS)
                    problem = Problem(
                        description=description,
                        kind="test_failure",
                        iteration=it,
                        problem_id=self._next_problem_id(),
                        source=_safe_str(source, 120),
                        error=description,
                        probable_cause=_safe_str(
                            getattr(item, "probable_cause",
                                    "unknown") or "unknown",
                            self.MAX_TEXT_CHARS),
                        suggested_solution=_safe_str(
                            getattr(item, "suggested_solution",
                                    "investigate") or "investigate",
                            self.MAX_TEXT_CHARS),
                        affected_files=[
                            f for f in (
                                getattr(item, "affected_files", None)
                                or []) if isinstance(f, str)][:5],
                        status=PROBLEM_STATUS_PENDING,
                        test=_safe_str(test, 200),
                    )
                    self.problems.append(problem)
                    created.append(problem.problem_id)
                except Exception:
                    continue
        except Exception:
            pass
        return (created, invalidated)

    def open_tracked_problems(self) -> list:
        """Problemas não-finais (para gate e Executor)."""
        try:
            return [p for p in self.problems if p.status in (
                PROBLEM_STATUS_PENDING,
                PROBLEM_STATUS_IN_PROGRESS,
                PROBLEM_STATUS_BLOCKED,
                PROBLEM_STATUS_PENDING_VERIFICATION,
            )]
        except Exception:
            return []

    def has_blocking_problems(self) -> bool:
        """Há pending/in_progress/blocked? (gate de retest)."""
        try:
            return any(p.status in BLOCKING_PROBLEM_STATUSES
                       for p in self.problems)
        except Exception:
            return False

    def blocking_problem_ids(self) -> list[str]:
        try:
            return [p.problem_id or p.test or p.description[:40]
                    for p in self.problems
                    if p.status in BLOCKING_PROBLEM_STATUSES]
        except Exception:
            return []

    @staticmethod
    def _files_overlap(files_changed: list,
                       affected: list) -> bool:
        """Interseção por basename (evidência, não adivinhação)."""
        try:
            changed = {str(f).replace("\\", "/").split("/")[-1]
                       for f in files_changed or [] if f}
            known = {str(f).replace("\\", "/").split("/")[-1]
                     for f in affected or [] if f}
            if not changed or not known:
                return False
            return bool(changed & known)
        except Exception:
            return False

    def mark_attempt_started(
        self,
        files_changed: list | None,
        iteration: int | None = None,
    ) -> list[str]:
        """Tentativa de correção começou → candidatos a in_progress.

        Candidatos: pending/pending_verification/blocked com overlap
        de arquivo, ou todos esses quando os arquivos são
        desconhecidos (tentativa sem alvo identificável). Nova
        tentativa reabre blocked. Retorna ids marcados. Nunca levanta.
        """
        marked: list[str] = []
        try:
            known_files = [f for f in (files_changed or [])
                           if isinstance(f, str) and f]
            for problem in self.problems:
                if problem.status not in (
                        PROBLEM_STATUS_PENDING,
                        PROBLEM_STATUS_PENDING_VERIFICATION,
                        PROBLEM_STATUS_BLOCKED):
                    continue
                if known_files and problem.affected_files and (
                        not self._files_overlap(
                            known_files, problem.affected_files)):
                    continue
                problem.status = PROBLEM_STATUS_IN_PROGRESS
                marked.append(problem.problem_id or problem.test)
        except Exception:
            pass
        return marked

    def apply_correction(
        self,
        description: str,
        files_changed: list | None,
        iteration: int | None = None,
    ):
        """Registra Correction e move vinculados → pending_verification.

        Vinculados: mesma regra de overlap de mark_attempt_started.
        Retorna a Correction (ou None se nada a vincular? — cria
        mesmo assim quando há abertos? NÃO: sem abertos, sem Correction
        — correção sem problema conhecido é só progresso normal).
        Nunca levanta.
        """
        try:
            if description is None:
                return None
            text = _safe_str(description, self.MAX_TEXT_CHARS)
            if not text.strip():
                return None
            known_files = [str(f)[:200] for f in (
                files_changed or []) if f]
            linked: list[str] = []
            for problem in self.problems:
                if problem.status not in (
                        PROBLEM_STATUS_PENDING,
                        PROBLEM_STATUS_IN_PROGRESS,
                        PROBLEM_STATUS_BLOCKED):
                    continue
                if known_files and problem.affected_files and (
                        not self._files_overlap(
                            known_files, problem.affected_files)):
                    continue
                linked.append(problem.problem_id)
            if not linked:
                return None
            try:
                it = (_safe_iteration(iteration)
                      if iteration is not None else None)
            except Exception:
                it = None
            correction = Correction(
                description=text,
                iteration=it,
                correction_id=self._next_correction_id(),
                problem_id=linked[0],
                problem_ids=list(linked),
                files_changed=known_files,
                status="applied",
            )
            self.corrections.append(correction)
            for problem in self.problems:
                if problem.problem_id in linked:
                    problem.status = (
                        PROBLEM_STATUS_PENDING_VERIFICATION)
                    problem.correction_id = correction.correction_id
            return correction
        except Exception:
            return None

    def mark_attempt_failed(
        self,
        files_changed: list | None,
    ) -> list[str]:
        """Tentativa falhou (erro de tool) → in_progress vira blocked."""
        marked: list[str] = []
        try:
            known_files = [f for f in (files_changed or [])
                           if isinstance(f, str) and f]
            for problem in self.problems:
                if problem.status != PROBLEM_STATUS_IN_PROGRESS:
                    continue
                if known_files and problem.affected_files and (
                        not self._files_overlap(
                            known_files, problem.affected_files)):
                    continue
                problem.status = PROBLEM_STATUS_BLOCKED
                marked.append(problem.problem_id or problem.test)
        except Exception:
            pass
        return marked

    def render_known_problems(
        self,
        max_items: int = 8,
        max_chars: int = 2000,
    ) -> str:
        """Bloco KNOWN PROBLEMS p/ o Executor (determinístico, capped)."""
        try:
            tracked = self.open_tracked_problems()
            if not tracked:
                return ""
            lines = ["KNOWN PROBLEMS (working hypotheses — verify "
                     "with read_file before fixing):"]
            for problem in tracked[:max_items]:
                files = ", ".join(problem.affected_files[:4]) or "-"
                lines.append(
                    f"{problem.problem_id or '?'} [{problem.status}]\n"
                    f"  Error: {_safe_str(problem.error, 200)}\n"
                    f"  Probable cause: "
                    f"{_safe_str(problem.probable_cause, 160)}\n"
                    f"  Suggested solution: "
                    f"{_safe_str(problem.suggested_solution, 160)}\n"
                    f"  Files: {files}")
            omitted = len(tracked) - min(len(tracked), max_items)
            if omitted:
                lines.append(f"... [+{omitted} more]")
            text = "\n".join(lines)
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "\n...[truncated]"
        except Exception:
            return ""

    def record_verification(
        self,
        gate: str,
        passed: bool,
        detail: str = "",
        iteration: int | None = None,
    ) -> None:
        try:
            self.verification.append(VerificationRecord(
                gate=_safe_str(gate, 60),
                passed=bool(passed),
                detail=_safe_str(detail, self.MAX_TEXT_CHARS),
                iteration=(_safe_iteration(iteration)
                           if iteration is not None else None),
                origin="verified" if passed else "observed",
            ))
        except Exception:
            pass

    @property
    def canonical_objective(self) -> str:
        """Objective de trabalho derivado do prompt canônico.

        É o que o Planner/Executor devem usar como objetivo principal
        (integração Fase 4); o texto integral está em
        `canonical_prompt` e o original do usuário em
        `original_prompt` (nunca descartado).
        """
        try:
            return self.objective or self.canonical_prompt[:300]
        except Exception:
            return ""

    def sync_plan(
        self,
        items: list[tuple[int, bool]] | None,
        blocked: bool = False,
        started: bool = False,
    ) -> None:
        """Espelha o checklist em `plan` (idempotente, sem duplicar texto).

        `items`: [(id, done)] do checklist (ids estáveis por run).
        `blocked`: há pendência bloqueante (ex.: error checklist).
        `started`: alguma decisão/execução já ocorreu.
        Transições: pending → in_progress (primeiro pendente, após
        início) → completed (done) ; in_progress → blocked (falha);
        blocked → in_progress (desbloqueou) ou → completed (done).
        Nunca levanta.
        """
        try:
            if not items:
                return
            first_open: int | None = None
            for item_id, done in items:
                try:
                    index = int(item_id)
                except Exception:
                    continue
                entry = None
                for existing in self.plan:
                    if existing.index == index:
                        entry = existing
                        break
                if entry is None:
                    entry = PlanItem(
                        index=index,
                        ref=f"checklist #{index}",
                        item_id=f"checklist-{index}",
                    )
                    self.plan.append(entry)
                if done:
                    entry.status = PLAN_STATUS_COMPLETED
                    entry.done = True
                else:
                    entry.done = False
                    if first_open is None:
                        first_open = index
                    if entry.status not in PLAN_STATUSES:
                        entry.status = PLAN_STATUS_PENDING
                    if entry.status == PLAN_STATUS_COMPLETED:
                        entry.status = PLAN_STATUS_PENDING
                    if blocked and (entry.status
                                    in (PLAN_STATUS_IN_PROGRESS,
                                        PLAN_STATUS_BLOCKED)
                                    or index == first_open):
                        entry.status = PLAN_STATUS_BLOCKED
                    elif not blocked and entry.status == (
                            PLAN_STATUS_BLOCKED):
                        entry.status = (
                            PLAN_STATUS_IN_PROGRESS if started
                            else PLAN_STATUS_PENDING)
            # Primeiro pendente vira in_progress após o início.
            if started and not blocked and first_open is not None:
                for entry in self.plan:
                    if entry.index == first_open and entry.status == (
                            PLAN_STATUS_PENDING):
                        entry.status = PLAN_STATUS_IN_PROGRESS
                        break
            self.plan.sort(key=lambda entry: entry.index)
        except Exception:
            pass

    # ----- consultas -----

    @property
    def open_problems(self) -> list[Problem]:
        """Abertos no sentido amplo (exclui resolved/invalidated)."""
        try:
            return [p for p in self.problems if p.status in (
                PROBLEM_STATUS_PENDING,
                PROBLEM_STATUS_IN_PROGRESS,
                PROBLEM_STATUS_BLOCKED,
                PROBLEM_STATUS_PENDING_VERIFICATION,
            )]
        except Exception:
            return []

    @property
    def progress_summary(self) -> dict[str, int]:
        try:
            done = sum(1 for e in self.progress if e.success)
            return {"total": len(self.progress), "succeeded": done,
                    "failed": len(self.progress) - done,
                    "decisions": len(self.decisions)}
        except Exception:
            return {"total": 0, "succeeded": 0, "failed": 0,
                    "decisions": 0}

    # ----- serialização determinística -----

    def to_dict(self) -> dict:
        """Dict JSON-serializável (ordem de campos estável)."""
        try:
            return asdict(self)
        except Exception:
            return {"original_prompt": str(
                getattr(self, "original_prompt", ""))}

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """Reconstrói o estado (tolerante a campos ausentes/extras)."""
        if not isinstance(data, dict):
            return cls()
        try:
            def _items(key: str, factory):
                raw = data.get(key) or []
                items = []
                if isinstance(raw, list):
                    for entry in raw:
                        if isinstance(entry, dict):
                            try:
                                known = {
                                    k: entry[k]
                                    for k in factory.__dataclass_fields__
                                    if k in entry
                                }
                                items.append(factory(**known))
                            except Exception:
                                continue
                return items

            return cls(
                original_prompt=str(data.get("original_prompt", "")),
                canonical_prompt=str(data.get("canonical_prompt", "")),
                language=str(data.get("language", "unknown")),
                translation_applied=bool(
                    data.get("translation_applied", False)),
                task_id=str(data.get("task_id", "")),
                objective=str(data.get("objective", "")),
                requirements=_items("requirements", Requirement),
                constraints=_items("constraints", TaskConstraint),
                ambiguities=_items("ambiguities", Ambiguity),
                plan=_items("plan", PlanItem),
                decisions=_items("decisions", DecisionRecord),
                progress=_items("progress", ProgressEntry),
                problems=_items("problems", Problem),
                corrections=_items("corrections", Correction),
                verification=_items("verification", VerificationRecord),
            )
        except Exception:
            return cls()

    # ----- renderização compacta (consumo futuro) -----

    # Problemas por render no Planner (detalhe) — Executor usa
    # KNOWN PROBLEMS + contagens (sem duplicar o detalhe).
    MAX_RENDER_PROBLEM_DETAIL = 8

    def render_compact(
        self,
        include_objective: bool = True,
        original_ref_chars: int = 0,
        problem_detail: bool = True,
    ) -> str:
        """Representação determinística, legível e limitada.

        Separa planejado (requirements/constraints/plan) de executado
        (progress/decisions) e verificado (problems resolved?,
        verification). Nunca inventa: seções vazias dizem "none".
        `include_objective=False` evita duplicar a seção OBJECTIVE
        quando o bloco é injetado logo após ela. `original_ref_chars`
        anexa referência truncada ao prompt original (acesso, não
        duplicação — o original integral segue no estado/trace).
        `problem_detail=False` resume problemas a contagens (p/ o
        Executor, que recebe o detalhe via KNOWN PROBLEMS).
        Defaults preservam a saída anterior byte a byte.
        """
        try:
            lines = []
            if include_objective:
                lines.append(
                    f"TASK: {_safe_str(self.objective, 300) or '(empty)'}")
            if original_ref_chars and self.original_prompt:
                lines.append(
                    "ORIGINAL: "
                    + _safe_str(self.original_prompt,
                                original_ref_chars))
            lines.append(
                f"LANGUAGE: {self.language}"
                + (" (translated)" if self.translation_applied else ""),
            )
            lines.append(self._render_items(
                "REQUIREMENTS", self.requirements,
                lambda r: r.text))
            lines.append(self._render_items(
                "CONSTRAINTS", self.constraints,
                lambda c: c.text))
            if self.ambiguities:
                lines.append(self._render_items(
                    "AMBIGUITIES", self.ambiguities,
                    lambda a: (f"{a.text} ({a.reason})"
                               if a.reason else a.text)))
            if self.plan:
                total = len(self.plan)
                done = sum(1 for item in self.plan
                           if item.status == PLAN_STATUS_COMPLETED)
                extra = []
                in_prog = sum(1 for item in self.plan
                              if item.status == PLAN_STATUS_IN_PROGRESS)
                blocked_n = sum(1 for item in self.plan
                                if item.status == PLAN_STATUS_BLOCKED)
                if in_prog:
                    extra.append(f"{in_prog} in progress")
                if blocked_n:
                    extra.append(f"{blocked_n} blocked")
                suffix = f" ({', '.join(extra)})" if extra else ""
                lines.append(f"PLAN: {done}/{total} done{suffix}")
            summary = self.progress_summary
            lines.append(
                f"PROGRESS: {summary['succeeded']}/{summary['total']} "
                f"succeeded ({summary['decisions']} decisions)")
            opened = self.open_problems
            if self.problems:
                lines.append(
                    f"PROBLEMS: {len(opened)} open / "
                    f"{len(self.problems)} total")
                if problem_detail:
                    shown = opened[:self.MAX_RENDER_PROBLEM_DETAIL]
                    for problem in shown:
                        lines.append(self._render_problem_line(problem))
                    omitted_open = len(opened) - len(shown)
                    if omitted_open:
                        lines.append(f"... [+{omitted_open} more]")
            else:
                lines.append("PROBLEMS: none")
            if self.corrections:
                lines.append(f"CORRECTIONS: {len(self.corrections)}")
                for correction in self.corrections[
                        -self.MAX_ITEMS_PER_SECTION:]:
                    lines.append(
                        f"- {_safe_str(correction.description, 120)}")
            if self.verification:
                verdicts = ", ".join(
                    f"{v.gate}: {'PASS' if v.passed else 'BLOCKED'}"
                    for v in self.verification)
                lines.append(f"VERIFICATION: {verdicts}")
            else:
                lines.append("VERIFICATION: none")
            return "\n".join(lines)
        except Exception:
            return "TASK: (unavailable)"

    @staticmethod
    def _render_problem_line(problem) -> str:
        """Uma linha rica por problema aberto (Fase 6, Planner).

        Formato: `- [status] id descrição | files: .. | cause: .. |
        fix: ..` — descrição primeiro (compat com parsers simples).
        Só fatos/hipóteses do estado; nada de output bruto.
        """
        try:
            ident = getattr(problem, "problem_id", "") or "?"
            status = getattr(problem, "status", "pending") or "pending"
            desc = _safe_str(
                getattr(problem, "description", ""), 120)
            parts = [f"- [{status}] {ident} {desc}".rstrip()]
            try:
                files = [f for f in (
                    getattr(problem, "affected_files", None) or [])
                    if isinstance(f, str)][:4]
            except Exception:
                files = []
            if files:
                parts.append(f"files: {', '.join(files)}")
            cause = _safe_str(
                getattr(problem, "probable_cause", ""), 100)
            if cause and cause != "unknown":
                parts.append(f"cause: {cause}")
            fix = _safe_str(
                getattr(problem, "suggested_solution", ""), 100)
            if fix and fix != "investigate":
                parts.append(f"fix: {fix}")
            return " | ".join(parts)
        except Exception:
            return "- [pending] ?"

    def _render_items(self, title: str, items: list, pick) -> str:
        if not items:
            return f"{title}: none"
        lines = [f"{title} ({len(items)}):"]
        for item in items[:self.MAX_ITEMS_PER_SECTION]:
            try:
                lines.append(
                    f"- {_safe_str(pick(item), self.MAX_TEXT_CHARS)}")
            except Exception:
                continue
        omitted = len(items) - min(len(items),
                                   self.MAX_ITEMS_PER_SECTION)
        if omitted:
            lines.append(f"... [+{omitted} more]")
        return "\n".join(lines)
