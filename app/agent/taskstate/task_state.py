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


@dataclass
class PlanItem:
    """Item de plano (espelha progresso do checklist, sem copiar texto)."""

    index: int
    done: bool = False
    # Referência curta (ex.: "checklist #2"); a descrição canônica
    # continua no ProjectChecklist.
    ref: str = ""


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


@dataclass
class Problem:
    """Problema visto em execução (falha de teste/build, erro)."""

    description: str
    kind: str = "test_failure"
    iteration: int | None = None
    resolved: bool = False
    origin: str = "observed"


@dataclass
class Correction:
    """Correção confirmada (só existe após evidência de resolução)."""

    description: str
    iteration: int | None = None
    origin: str = "observed"


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
                if not problem.resolved:
                    problem.resolved = True
                    resolved += 1
        except Exception:
            pass
        return resolved

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

    # ----- consultas -----

    @property
    def open_problems(self) -> list[Problem]:
        try:
            return [p for p in self.problems if not p.resolved]
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

    def render_compact(self) -> str:
        """Representação determinística, legível e limitada.

        Separa planejado (requirements/constraints/plan) de executado
        (progress/decisions) e verificado (problems resolved?,
        verification). Nunca inventa: seções vazias dizem "none".
        """
        try:
            lines = [
                f"TASK: {_safe_str(self.objective, 300) or '(empty)'}",
                f"LANGUAGE: {self.language}"
                + (" (translated)" if self.translation_applied else ""),
            ]
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
                done = sum(1 for item in self.plan if item.done)
                lines.append(f"PLAN: {done}/{total} done")
            summary = self.progress_summary
            lines.append(
                f"PROGRESS: {summary['succeeded']}/{summary['total']} "
                f"succeeded ({summary['decisions']} decisions)")
            opened = self.open_problems
            if self.problems:
                lines.append(
                    f"PROBLEMS: {len(opened)} open / "
                    f"{len(self.problems)} total")
                for problem in opened[:self.MAX_ITEMS_PER_SECTION]:
                    lines.append(
                        f"- [open] {_safe_str(problem.description, 120)}")
                omitted_open = (len(opened)
                                - min(len(opened),
                                      self.MAX_ITEMS_PER_SECTION))
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
