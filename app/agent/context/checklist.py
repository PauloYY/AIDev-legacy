from dataclasses import dataclass

from app.llm.client import LLMClient
from app.llm.json_extraction import parse_json_object
from app.llm.models import Message


@dataclass
class ChecklistItem:
    id: int
    description: str
    done: bool = False


class ProjectChecklist:
    """Checklist estruturado do objetivo, gerado UMA VEZ no início da
    run e mantido em Python puro dali em diante.

    Diferente do RESUMO DO PROJETO (reescrito por LLM a cada
    iteração, sujeito a deriva e esquecimento ao longo de uma run
    longa), o texto de cada item aqui nunca é reescrito ou resumido
    de novo — só muda quando o Planner explicitamente marca um id
    como concluído. Isso dá ao Planner uma sequência fixa de etapas
    para seguir, em vez de precisar redescobrir "o que falta fazer"
    a cada iteração só a partir de um resumo em prosa, que é uma das
    causas de loop (fica reinvestigando o que já foi decidido).
    """

    MIN_ITEMS = 3
    MAX_ITEMS = 15

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._items: list[ChecklistItem] = []

    def reset(self) -> None:
        self._items = []

    def generate(self, objective: str) -> None:
        prompt = f"""
You are responsible for breaking the objective below into a checklist of
concrete, verifiable steps for an autonomous software development
agent to follow in order.

OBJECTIVE:
{objective}

RULES:
- Return ONLY JSON in the format:
  {{"items": ["first step", "second step", ...]}}
- Each item must be a concrete, verifiable step (e.g. "Create the
  Client model with id, name and status fields", "Implement the
  POST /clients endpoint", "Write tests for the priority queue").
  Avoid vague items like "plan" or "review the code".
- Order the items in the logical implementation sequence.
- Include, as the last item, validating the result by running the
  tests/build/program before finishing.
- Use between {self.MIN_ITEMS} and {self.MAX_ITEMS} items — prefer the
  smallest number that fully covers the objective.
"""

        response = self.llm.generate(
            messages=[Message(role="user", content=prompt)],
            component="ProjectChecklist",
        )

        data = parse_json_object(response.content)
        descriptions = data.get("items")

        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError(
                "O checklist gerado está vazio ou em formato inválido."
            )

        self._items = [
            ChecklistItem(id=index, description=str(description))
            for index, description in enumerate(descriptions, start=1)
        ]

    def mark_done(self, ids) -> list[int]:
        """Marca os ids informados como concluídos.

        Retorna a lista de ids que foram efetivamente marcados (ids
        desconhecidos ou já concluídos são ignorados silenciosamente
        — a LLM não precisa acertar o estado exato, só sinalizar
        progresso).
        """

        if not ids:
            return []

        valid_ids = {item.id for item in self._items}
        marked = []

        for item in self._items:
            if item.id in ids and not item.done:
                item.done = True
                marked.append(item.id)

        return marked

    def render(self) -> str:
        if not self._items:
            return "(checklist not defined yet)"

        lines = [
            "OBJECTIVE CHECKLIST (defined once at the start of "
            "this run; each item description never changes — only the "
            "done/pending state. When you complete an item, include "
            '"checklist_progress": [ids] in your decision):',
        ]

        for item in self._items:
            mark = "x" if item.done else " "
            lines.append(f"[{mark}] {item.id}. {item.description}")

        return "\n".join(lines)

    @property
    def statuses(self) -> list[tuple[int, bool]]:
        """[(id, done)] para sincronizar o TaskState.plan (Fase 4).

        Só identidade + estado — as descrições continuam aqui (o
        TaskState referencia, não copia).
        """
        try:
            return [(item.id, bool(item.done)) for item in self._items]
        except Exception:
            return []

    @property
    def pending_items(self) -> list[ChecklistItem]:
        return [item for item in self._items if not item.done]

    @property
    def pending_count(self) -> int:
        return len(self.pending_items)