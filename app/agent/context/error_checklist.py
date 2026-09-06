from dataclasses import dataclass

from app.llm.client import LLMClient
from app.llm.json_extraction import parse_json_object
from app.llm.models import Message


@dataclass
class ErrorChecklistItem:
    id: int
    description: str


class ErrorChecklist:
    """Checklist das falhas de teste/build ATUAIS, extraído por LLM a
    partir da saída bruta de um run_command que falhou.

    Diferente do ProjectChecklist (fixo a run inteira, definido uma
    única vez), este é efêmero e evidência-based:

    - É gerado do zero sempre que um run_command de teste/build
      falha, a partir da saída real desse comando.
    - É limpo automaticamente assim que um run_command de teste/build
      volta a ter sucesso — sem precisar de nenhuma marcação manual
      da LLM. O progresso aqui é sempre validado por execução real,
      nunca por autoavaliação: se ainda falhar (mesmo que só
      parcialmente), o checklist é regerado do zero a partir da nova
      saída, então itens já corrigidos simplesmente somem (os testes
      deles já passam) e ficam só os que ainda quebram + eventuais
      falhas novas.
    """

    MIN_ITEMS = 0
    MAX_ITEMS = 20
    MAX_INPUT_CHARS = 6000

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._items: list[ErrorChecklistItem] = []

    def reset(self) -> None:
        self._items = []

    def clear(self) -> None:
        self._items = []

    def generate(self, command: str, output: str) -> None:
        truncated = str(output)

        if len(truncated) > self.MAX_INPUT_CHARS:
            omitted = len(truncated) - self.MAX_INPUT_CHARS
            truncated = (
                f"{truncated[:self.MAX_INPUT_CHARS]}\n"
                f"...[truncado, {omitted} caracteres omitidos]"
            )

        prompt = f"""
Você é responsável por extrair, da saída bruta de um comando de
teste/build que falhou, uma lista objetiva das falhas reais.

COMANDO EXECUTADO:
{command}

SAÍDA DO COMANDO:
{truncated}

REGRAS:
- Retorne SOMENTE um JSON no formato:
  {{"items": ["descrição da falha 1", "descrição da falha 2", ...]}}
- Cada item deve descrever UMA falha concreta e específica: o
  teste/arquivo envolvido, o que era esperado e o que aconteceu
  (ex.: "RideService.test.js: 'deve falhar quando motorista estiver
  ocupado' esperava toThrow('Motorista está ocupado'), mas a função
  não lançou exceção").
- Ignore testes que passaram. Ignore avisos que não são falhas.
- Se a saída for de um erro de compilação/build (não testes),
  extraia cada erro de compilação como um item separado.
- Se não houver nenhuma falha real identificável no texto (ex.: a
  saída não tem relação com teste/build, ou o motivo da falha é
  só infraestrutura, tipo timeout do sandbox), retorne
  {{"items": []}}.
- Use no máximo {self.MAX_ITEMS} itens — se houver mais falhas que
  isso, priorize as mais informativas/distintas.
"""

        response = self.llm.generate(
            messages=[Message(role="user", content=prompt)]
        )

        data = parse_json_object(response.content)
        descriptions = data.get("items")

        if not isinstance(descriptions, list):
            raise ValueError(
                "A extração de erros retornou um formato inválido."
            )

        self._items = [
            ErrorChecklistItem(id=index, description=str(description))
            for index, description in enumerate(descriptions, start=1)
        ]

    def render(self) -> str | None:
        """Retorna o bloco de contexto, ou None quando não há nenhuma
        falha pendente (nesse caso o Runner simplesmente omite o
        bloco — não faz sentido mostrar uma seção vazia toda
        iteração)."""

        if not self._items:
            return None

        lines = [
            "CHECKLIST DE ERROS ATUAIS (extraído automaticamente da "
            "última falha de teste/build; some sozinho quando os "
            "testes voltarem a passar — não precisa marcar nada "
            "manualmente, só corrigir e rodar o teste de novo):",
        ]

        lines += [
            f"- {item.id}. {item.description}" for item in self._items
        ]

        return "\n".join(lines)

    @property
    def pending_count(self) -> int:
        return len(self._items)