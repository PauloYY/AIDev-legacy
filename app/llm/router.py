import logging
import time

from app.exceptions import LLMAPIError, LLMConnectionError, LLMRateLimitError
from app.llm.models import LLMResponse, Message
from app.llm.providers.base import LLMProvider


logger = logging.getLogger(__name__)


class LLMRouter(LLMProvider):
    """Distribui chamadas entre providers, com fallback e espera automática.

    Faz fallback para o próximo provider quando o erro é considerado
    transitório: rate limit, falha de conexão, ou erro 5xx da API. Erros
    definitivos (ex.: 4xx de request malformada) não disparam fallback.

    Se TODOS os providers falharem com erros transitórios numa mesma
    rodada, em vez de desistir na hora, o router espera um tempo (usando
    o header Retry-After quando disponível, senão backoff exponencial) e
    tenta de novo, até `max_wait_rounds` vezes, antes de desistir de vez.
    """

    DEFAULT_MAX_WAIT_ROUNDS = 3
    DEFAULT_BASE_WAIT_SECONDS = 10.0
    DEFAULT_MAX_WAIT_SECONDS = 90.0

    def __init__(
        self,
        providers: list[LLMProvider],
        max_wait_rounds: int | None = None,
        base_wait_seconds: float | None = None,
        max_wait_seconds: float | None = None,
    ):
        self.providers = providers
        self.current_index = 0

        self.max_wait_rounds = (
            self.DEFAULT_MAX_WAIT_ROUNDS
            if max_wait_rounds is None
            else max_wait_rounds
        )
        self.base_wait_seconds = (
            self.DEFAULT_BASE_WAIT_SECONDS
            if base_wait_seconds is None
            else base_wait_seconds
        )
        self.max_wait_seconds = (
            self.DEFAULT_MAX_WAIT_SECONDS
            if max_wait_seconds is None
            else max_wait_seconds
        )

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        if not self.providers:
            raise RuntimeError("Nenhum provider disponível.")

        last_error: Exception | None = None
        retry_after: float | None = None

        for wait_round in range(self.max_wait_rounds + 1):

            for _ in range(len(self.providers)):
                provider = self.providers[self.current_index]
                provider_name = getattr(
                    provider, "name", provider.__class__.__name__
                )

                try:
                    return provider.generate(
                        messages,
                        tools,
                    )

                except LLMRateLimitError as error:
                    last_error = error
                    retry_after = getattr(error, "retry_after", None) or retry_after

                    logger.info(
                        "Rate limit em %s, tentando próximo provider.",
                        provider_name,
                    )
                    self._advance()

                except LLMConnectionError as error:
                    last_error = error

                    logger.warning(
                        "Falha de conexão com %s, tentando próximo provider.",
                        provider_name,
                    )
                    self._advance()

                except LLMAPIError as error:
                    last_error = error

                    if not error.is_transient:
                        raise

                    logger.warning(
                        "Erro transitório (status=%s) em %s, "
                        "tentando próximo provider.",
                        error.status_code,
                        provider_name,
                    )
                    self._advance()

            if wait_round >= self.max_wait_rounds:
                break

            wait_seconds = self._compute_wait_seconds(
                wait_round,
                retry_after,
            )

            logger.warning(
                "Todos os %d providers falharam com erros transitórios. "
                "Aguardando %.0fs antes de tentar novamente "
                "(tentativa %d/%d)...",
                len(self.providers),
                wait_seconds,
                wait_round + 1,
                self.max_wait_rounds,
            )

            time.sleep(wait_seconds)
            retry_after = None

        raise LLMRateLimitError(
            "Todos os providers falharam com erros transitórios mesmo "
            f"após aguardar (último erro: {last_error})."
        )

    def _compute_wait_seconds(
        self,
        wait_round: int,
        retry_after: float | None,
    ) -> float:
        if retry_after:
            return min(retry_after, self.max_wait_seconds)

        wait = self.base_wait_seconds * (2 ** wait_round)

        return min(wait, self.max_wait_seconds)

    def _advance(self) -> None:
        self.current_index = (
            self.current_index + 1
        ) % len(self.providers)