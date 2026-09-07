from typing import Any, Callable

import logging
import time

from app.agent.context.project_summary import ProjectSummary
from app.agent.execution.task import Task
from app.exceptions import LLMConnectionError
from app.llm.client import LLMClient
from app.llm.models import Message


logger = logging.getLogger(__name__)


class ProjectSummaryUpdater:

    # Etapa 2E: retry limitado — 1 tentativa inicial + até 2 retries
    # (máximo 3 chamadas). Nunca retry infinito.
    MAX_SUMMARY_ATTEMPTS = 3
    # Backoff determinístico entre retries (s); sleep injetável p/ testes.
    RETRY_BACKOFF_SECONDS = (1.0, 2.0)

    # Etapa 2D: limites da REPRESENTAÇÃO usada no prompt (nunca dos dados
    # reais). Escolhas baseadas no código existente:
    # - MAX_ARG_VALUE_CHARS = 200: mesmo valor da compactação do
    #   action_history (OperationalMemory, Etapa 2C), para consistência.
    # - MAX_RESULT_CHARS = 2000: maior que o resumo de resultado do
    #   histórico (300, que só sinaliza OK/falha) porque atualizar o
    #   resumo narrativo precisa de mais evidência; menor que o bloco de
    #   contexto do Planner (Runner.MAX_CONTEXT_CHARS = 4000), porque
    #   aqui basta o desfecho + trechos relevantes, não a evidência
    #   completa (logs integrais seguem disponíveis em outras vias).
    MAX_ARG_VALUE_CHARS = 200
    MAX_RESULT_CHARS = 2000

    def __init__(
        self,
        llm: LLMClient,
        summary: ProjectSummary,
    ):
        self.llm = llm
        self.summary = summary

    def __init__(
        self,
        llm: LLMClient,
        summary: ProjectSummary,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.llm = llm
        self.summary = summary
        self._sleep = sleep_fn or time.sleep

    def update(
        self,
        objective: str,
        project_name: str,
        task: Task,
        result: str,
        iteration: int | None = None,
    ) -> str:

        current_summary = self.summary.read(
            project_name
        )

        prompt = self._build_prompt(
            objective=objective,
            current_summary=current_summary,
            task=task,
            result=result,
        )
        prompt_chars = len(prompt)

        for attempt in range(1, self.MAX_SUMMARY_ATTEMPTS + 1):
            try:
                response = self._generate(prompt, iteration, attempt)
            except LLMConnectionError as error:
                # Transitório (rede/timeout de leitura): retry limitado.
                # Erros permanentes (LLMAPIError, LLMRateLimitError,
                # LLMInvalidResponseError, config, programação) propagam
                # sem retry.
                self._log_transient(
                    iteration, attempt, prompt_chars, error
                )
                if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                    raise
                self._sleep(
                    self.RETRY_BACKOFF_SECONDS[attempt - 1]
                )
                continue

            content = response.content
            if content is not None and content.strip():
                updated_summary = content.strip()
                self.summary.write(
                    project_name,
                    updated_summary,
                )
                return updated_summary

            # Resposta vazia ("" / whitespace / None): plausivelmente
            # transitória (caso Agnes HTTP 200 + content="") — retry
            # limitado, nunca aceitar resumo vazio em silêncio.
            self._log_empty(
                iteration, attempt, prompt_chars, content
            )
            if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                raise ValueError(
                    "A LLM retornou um resumo vazio após "
                    f"{self.MAX_SUMMARY_ATTEMPTS} tentativas "
                    "(component=ProjectSummaryUpdater, "
                    f"iteration={iteration})."
                )
            self._sleep(self.RETRY_BACKOFF_SECONDS[attempt - 1])

    def _generate(self, prompt: str, iteration: int | None, attempt: int):
        try:
            return self.llm.generate(
                messages=[
                    Message(
                        role="user",
                        content=prompt,
                    )
                ],
                component="ProjectSummaryUpdater",
                iteration=iteration,
                attempt=attempt,
            )
        except TypeError:
            # Compatibilidade com fakes antigos sem o kwarg `attempt`.
            return self.llm.generate(
                messages=[
                    Message(
                        role="user",
                        content=prompt,
                    )
                ],
                component="ProjectSummaryUpdater",
                iteration=iteration,
            )

    def _log_empty(
        self,
        iteration: int | None,
        attempt: int,
        prompt_chars: int,
        content: str | None,
    ) -> None:
        logger.warning(
            "ProjectSummaryUpdater: iteration=%s attempt=%d/%d "
            "prompt_chars=%d estimated_prompt_tokens=%d "
            "response_chars=%d reason=empty_response duration_ms=%.1f",
            iteration,
            attempt,
            self.MAX_SUMMARY_ATTEMPTS,
            prompt_chars,
            prompt_chars // 4,  # mesma aproximação de estimate_tokens
            len(content or ""),
            self._last_duration_ms(),
        )

    def _log_transient(
        self,
        iteration: int | None,
        attempt: int,
        prompt_chars: int,
        error: Exception,
    ) -> None:
        logger.warning(
            "ProjectSummaryUpdater: iteration=%s attempt=%d/%d "
            "prompt_chars=%d estimated_prompt_tokens=%d error=%s "
            "duration_ms=%.1f",
            iteration,
            attempt,
            self.MAX_SUMMARY_ATTEMPTS,
            prompt_chars,
            prompt_chars // 4,
            type(error).__name__,
            self._last_duration_ms(),
        )

    def _last_duration_ms(self) -> float:
        """Duração do último attempt registrada no tracker (defensivo)."""

        usage = getattr(self.llm, "usage", None)
        if usage is None:
            return 0.0
        try:
            records = usage.get_records()
        except Exception:
            return 0.0
        if not records:
            return 0.0
        return records[-1].duration_ms

    def _build_prompt(
        self,
        objective: str,
        current_summary: str,
        task: Task,
        result: str,
    ) -> str:
        arguments_repr = self._format_arguments_for_prompt(
            task.tool, task.arguments
        )
        result_repr = self._format_result_for_prompt(result)

        return f"""
Você é responsável por manter o resumo persistente
de um projeto de desenvolvimento.

Atualize o resumo do projeto utilizando as informações
da tarefa que acabou de ser executada.

OBJETIVO:
{objective}

RESUMO ATUAL:
{current_summary}

TASK EXECUTADA:
Tool: {task.tool}

ARGUMENTOS:
{arguments_repr}

RESULTADO:
{result_repr}

REGRAS:
- Retorne SOMENTE o conteúdo completo do novo resumo.
- Não retorne JSON.
- Preserve informações importantes do resumo atual.
- Incorpore as novas informações descobertas.
- Não invente informações.
- Remova informações que comprovadamente estejam incorretas.
- O resumo deve representar o estado atual conhecido do projeto.
- Seja conciso.
- Limite o resumo a no máximo 300 palavras.
- Não liste nomes de arquivos nem a estrutura de pastas — isso já é
  fornecido separadamente como memória operacional determinística.
  Foque em decisões de design, regras de negócio, convenções
  adotadas e o que ainda falta fazer.
"""

    def _format_arguments_for_prompt(
        self, tool: str, arguments: dict[str, Any] | None
    ) -> str:
        """Representação compacta dos argumentos SÓ para o prompt.

        Mesma ideia da Etapa 2C (o prompt registra O QUE aconteceu, não
        repete o conteúdo escrito), ajustada a este prompt:
        - write_file.content NUNCA vai integral (vira "<omitted: N chars>").
        - strings maiores que MAX_ARG_VALUE_CHARS são truncadas com
          indicador de caracteres omitidos.
        - nunca muta o dict original (constrói um novo).
        """
        if arguments is None:
            return "{}"
        if not isinstance(arguments, dict):
            return self._truncate_text(str(arguments))

        compact: dict[str, Any] = {}
        for key, value in arguments.items():
            if tool == "write_file" and key == "content":
                if value is None:
                    compact[key] = "<empty>"
                else:
                    text = value if isinstance(value, str) else str(value)
                    if len(text) == 0:
                        compact[key] = "<empty>"
                    else:
                        compact[key] = f"<omitted: {len(text)} chars>"
                continue

            if isinstance(value, str):
                compact[key] = self._truncate_text(value)
                continue

            try:
                rep = repr(value)
            except Exception:
                rep = str(value)
            if len(rep) > self.MAX_ARG_VALUE_CHARS + 20:
                compact[key] = self._truncate_text(str(value))
            else:
                compact[key] = value

        return repr(compact)

    def _format_result_for_prompt(self, result: Any) -> str:
        """Representação limitada do resultado SÓ para o prompt.

        Preserva o início (onde está o desfecho: STATUS, mensagem de
        sucesso/erro) e indica o total quando há truncamento. Não altera
        o resultado real armazenado pelo sistema.
        """
        text = result if isinstance(result, str) else str(result)
        if len(text) <= self.MAX_RESULT_CHARS:
            return text
        omitted = len(text) - self.MAX_RESULT_CHARS
        return (
            f"[result truncated: {len(text)} chars total]\n"
            f"{text[:self.MAX_RESULT_CHARS]}"
            f"\n... [+{omitted} chars omitted]"
        )

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.MAX_ARG_VALUE_CHARS:
            return text
        omitted = len(text) - self.MAX_ARG_VALUE_CHARS
        return f"{text[:self.MAX_ARG_VALUE_CHARS]}... [+{omitted} chars]"