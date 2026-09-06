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
Você é responsável por quebrar o objetivo abaixo em um checklist de
etapas concretas e verificáveis para um agente autônomo de
desenvolvimento de software seguir em ordem.

OBJETIVO:
{objective}

REGRAS:
- Retorne SOMENTE um JSON no formato:
  {{"items": ["primeira etapa", "segunda etapa", ...]}}
- Cada item deve ser uma etapa concreta e verificável (ex.: "Criar
  modelo Client com os campos id, nome e status", "Implementar o
  endpoint POST /clients", "Escrever testes para a fila de
  prioridade"). Evite itens vagos como "planejar" ou "revisar o
  código".
- Ordene os itens na sequência lógica de implementação.
- Inclua, como último item, validar o resultado executando os
  testes/build/programa antes de finalizar.
- Use entre {self.MIN_ITEMS} e {self.MAX_ITEMS} itens — prefira o
  menor número que cobre o objetivo por completo.
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
            return "(checklist ainda não definido)"

        lines = [
            "CHECKLIST DO OBJETIVO (definido uma única vez no início "
            "desta run; a descrição de cada item nunca muda — só o "
            "estado concluído/pendente. Ao concluir um item, inclua "
            '"checklist_progress": [ids] na sua decisão):',
        ]

        for item in self._items:
            mark = "x" if item.done else " "
            lines.append(f"[{mark}] {item.id}. {item.description}")

        return "\n".join(lines)

    @property
    def pending_items(self) -> list[ChecklistItem]:
        return [item for item in self._items if not item.done]

    @property
    def pending_count(self) -> int:
        return len(self.pending_items)