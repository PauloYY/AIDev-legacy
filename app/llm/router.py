import logging

from app.exceptions import LLMAPIError, LLMConnectionError, LLMRateLimitError
from app.llm.models import LLMResponse, Message
from app.llm.providers.base import LLMProvider


logger = logging.getLogger(__name__)


class LLMRouter(LLMProvider):
    """Distribui chamadas entre providers, com fallback automático.

    Faz fallback para o próximo provider quando o erro é considerado
    transitório: rate limit, falha de conexão, ou erro 5xx da API.
    Erros definitivos (ex.: 4xx de request malformada, resposta inválida)
    não disparam fallback — trocar de provider não resolveria o problema.
    """

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self.current_index = 0

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:

        if not self.providers:
            raise RuntimeError("Nenhum provider disponível.")

        last_error: Exception | None = None

        for _ in range(len(self.providers)):
            provider = self.providers[self.current_index]
            provider_name = getattr(provider, "name", provider.__class__.__name__)

            try:
                return provider.generate(
                    messages,
                    tools,
                )

            except LLMRateLimitError as error:
                last_error = error
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

        raise LLMRateLimitError(
            "Todos os providers falharam com erros transitórios "
            f"(último erro: {last_error})."
        )

    def _advance(self) -> None:
        self.current_index = (
            self.current_index + 1
        ) % len(self.providers)
