"""Fase 4 — TaskInterpreter determinístico (zero chamadas LLM).

Extrai ESTRUTURA explícita do prompt canônico: objetivo (primeira
sentença), requisitos (itens de lista explícitos), restrições
(sentenças com marcadores de restrição EN/PT) e ambiguidades
(marcadores concretos como "etc.", "?", "TBD").

Regra de ouro: nunca inventar. Sem item de lista → requirements == [].
Sem marcador → sem constraint/ambiguity. O Planner continua sendo quem
interpreta semanticamente a cada iteração; aqui só se organiza o que já
está dito, com proveniência "interpreted".
"""

import re
from dataclasses import dataclass, field

from app.agent.taskstate.task_state import (
    Ambiguity,
    Requirement,
    TaskConstraint,
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.*\S)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Marcadores de restrição (palavra inteira; EN + PT).
_CONSTRAINT_WORDS = frozenset({
    "only", "must", "never", "without", "before", "after",
    "apenas", "somente", "nunca", "sem", "antes", "depois",
})
_CONSTRAINT_PHRASES = (
    "must not", "mustn't", "cannot", "can't", "at least", "at most",
    "up to", "no more than", "ao menos", "no máximo", "limite",
    "não pode", "nao pode",
)
_WORD_RE = re.compile(r"[a-zà-úâêôãõç]+", re.IGNORECASE)

# Marcadores de ambiguidade (concretos, não palpites).
_AMBIGUITY_PHRASES = (
    "etc.", "etc", "something", "somehow", "tbd", "todo",
    "talvez", "algo assim",
)
_AMBIGUITY_WORDS = frozenset({
    "something", "somehow", "appropriate", "relevant", "etc",
    "talvez",
})

MAX_SENTENCE_CHARS = 300
MAX_REQUIREMENTS = 20
MAX_CONSTRAINTS = 20
MAX_AMBIGUITIES = 10


def split_sentences(text: str) -> list[str]:
    """Sentenças por pontuação/quebra de linha (determinístico)."""
    try:
        parts = _SENTENCE_SPLIT_RE.split(text)
    except Exception:
        return []
    return [part.strip() for part in parts if part and part.strip()]


def split_bullets(text: str) -> list[str]:
    """Itens de lista explícitos (-, *, •, 1. 2) ...)."""
    items = []
    try:
        for line in text.splitlines():
            match = _BULLET_RE.match(line)
            if match:
                items.append(match.group(1).strip())
    except Exception:
        pass
    return items


def _has_constraint_markers(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(phrase in lowered for phrase in _CONSTRAINT_PHRASES):
        return True
    words = set(_WORD_RE.findall(lowered))
    return bool(words & _CONSTRAINT_WORDS)


def _ambiguity_reason(sentence: str) -> str | None:
    lowered = sentence.lower()
    if "?" in sentence:
        return "open_question"
    for phrase in _AMBIGUITY_PHRASES:
        if phrase in lowered:
            return f"vague_marker:{phrase}"
    words = set(_WORD_RE.findall(lowered))
    vague = words & _AMBIGUITY_WORDS
    if vague:
        return f"vague_word:{sorted(vague)[0]}"
    return None


@dataclass
class Interpretation:
    objective: str = ""
    requirements: list[Requirement] = field(default_factory=list)
    constraints: list[TaskConstraint] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)


class TaskInterpreter:
    """Interpretação estrutural determinística do prompt canônico."""

    def interpret(
        self, canonical_prompt, language: str = "en",
    ) -> Interpretation:
        try:
            if canonical_prompt is None:
                text = ""
            else:
                text = (canonical_prompt
                        if isinstance(canonical_prompt, str)
                        else str(canonical_prompt))
        except Exception:
            text = ""
        text = text.strip()
        if not text:
            return Interpretation()

        sentences = split_sentences(text)
        objective = ""
        if sentences:
            objective = sentences[0][:MAX_SENTENCE_CHARS].strip()

        requirements: list[Requirement] = []
        for item in split_bullets(text)[:MAX_REQUIREMENTS]:
            # Item com marcador de restrição é restrição, não requisito.
            if _has_constraint_markers(item):
                continue
            requirements.append(Requirement(
                text=item[:MAX_SENTENCE_CHARS],
                req_id=f"R{len(requirements) + 1}",
            ))

        constraints: list[TaskConstraint] = []
        counter = 0
        # Bullets restritivos primeiro (ordem do texto), depois sentenças.
        for item in split_bullets(text):
            if len(constraints) >= MAX_CONSTRAINTS:
                break
            if _has_constraint_markers(item):
                counter += 1
                constraints.append(TaskConstraint(
                    text=item[:MAX_SENTENCE_CHARS],
                    constraint_id=f"C{counter}",
                ))
        for sentence in sentences:
            if len(constraints) >= MAX_CONSTRAINTS:
                break
            # Sentença que já veio de bullet: compara/armazena sem o
            # marcador (evita duplicar a mesma restrição).
            bullet = _BULLET_RE.match(sentence)
            short = (bullet.group(1) if bullet else sentence)
            short = short[:MAX_SENTENCE_CHARS].strip()
            if not short or any(c.text == short for c in constraints):
                continue
            if _has_constraint_markers(sentence):
                counter += 1
                constraints.append(TaskConstraint(
                    text=short, constraint_id=f"C{counter}"))

        ambiguities: list[Ambiguity] = []
        for sentence in sentences:
            if len(ambiguities) >= MAX_AMBIGUITIES:
                break
            reason = _ambiguity_reason(sentence)
            if reason:
                ambiguities.append(Ambiguity(
                    text=sentence[:MAX_SENTENCE_CHARS].strip(),
                    reason=reason,
                ))

        return Interpretation(
            objective=objective,
            requirements=requirements,
            constraints=constraints,
            ambiguities=ambiguities,
        )
