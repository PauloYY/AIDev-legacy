from dataclasses import dataclass, field

from app.llm.client import LLMClient
from app.llm.json_extraction import parse_json_object
from app.llm.models import Message


@dataclass
class ErrorChecklistItem:
    id: int
    description: str
    # --- Fase 6: projeção da análise unificada (defaults seguros;
    # itens legados/sintéticos continuam válidos sem eles) ---
    test_id: str = ""
    affected_files: list[str] = field(default_factory=list)
    # Comando e iteração de origem (rastreabilidade da evidência).
    command: str = ""
    iteration: int | None = None


class ErrorChecklist:
    """Checklist das falhas de teste/build ATUAIS.

    Fase 6: projeção operacional da análise unificada
    (`apply_unified`, sem LLM próprio). O legado `generate` (LLM
    própria) segue disponível p/ compatibilidade e benchmark, mas o
    hot path não o usa mais — uma análise, vários consumidores.

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

    def generate(
        self, command: str, output: str, iteration: int | None = None
    ) -> None:
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
            messages=[Message(role="user", content=prompt)],
            component="ErrorChecklist",
            iteration=iteration,
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

    def apply_unified(self, analysis) -> int:
        """Projeção da análise unificada (Fase 6, SEM LLM).

        Substitui os itens pelos da análise (hipóteses primeiro;
        fatos determinísticos quando só eles existirem) — mesma
        semântica de regenerate-from-scratch do generate(). Também
        serve como fallback determinístico (análise sem problemas e
        sem fatos → checklist vazio, como o legado com []). Retorna
        quantos itens foram projetados. Nunca levanta.
        """
        try:
            try:
                command = getattr(analysis, "command", "") or ""
            except Exception:
                command = ""
            entries: list[tuple[str, str, list]] = []
            problems = list(getattr(analysis, "problems", None) or [])
            for problem in problems[:self.MAX_ITEMS]:
                try:
                    error = getattr(problem, "error", "") or ""
                    test = getattr(problem, "test", "") or ""
                    files = [f for f in (
                        getattr(problem, "affected_files", None)
                        or []) if isinstance(f, str)][:4]
                    from app.agent.errors.error_analyzer import (
                        checklist_description,
                    )

                    entries.append((checklist_description(test, error),
                                    test, files))
                except Exception:
                    continue
            if not entries:
                failures = list(getattr(analysis, "failures", None)
                                or [])
                for fact in failures[:self.MAX_ITEMS]:
                    try:
                        from app.agent.errors.error_analyzer import (
                            checklist_description,
                        )

                        entries.append((checklist_description(
                            getattr(fact, "test", ""),
                            getattr(fact, "error", "")),
                            getattr(fact, "test", "") or "",
                            [f for f in (
                                getattr(fact, "files", None) or [])
                             if isinstance(f, str)][:4]))
                    except Exception:
                        continue
            self._items = [
                ErrorChecklistItem(id=index, description=description,
                                   test_id=test_id,
                                   affected_files=list(files),
                                   command=str(command)[:200])
                for index, (description, test_id,
                            files) in enumerate(entries, start=1)
            ]
            return len(self._items)
        except Exception:
            return 0

    def note_infra_failure(self, command: str, reason: str) -> None:
        """Registra falha de infra/timeout sem descartar itens reais.

        Timeout ou erro de execução não é evidência de que os testes
        passam — então os itens anteriores são preservados e, se não
        houver nenhum, um item sintético garante o bloqueio do finish
        até um run com sucesso (que limpa tudo via clear()).
        """

        description = (
            f"Comando '{command}' {reason}; rode-o novamente e "
            "confirme que passa antes de finalizar."
        )

        if any(item.description == description for item in self._items):
            return

        self._items.append(
            ErrorChecklistItem(
                id=len(self._items) + 1, description=description
            )
        )

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

        for item in self._items:
            line = f"- {item.id}. {item.description}"
            try:
                files = [f for f in (
                    getattr(item, "affected_files", None) or [])
                    if isinstance(f, str)][:4]
            except Exception:
                files = []
            if files:
                line += f" [files: {', '.join(files)}]"
            lines.append(line)

        return "\n".join(lines)

    @property
    def pending_count(self) -> int:
        return len(self._items)