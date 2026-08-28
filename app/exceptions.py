class AIDevError(Exception):

    """Erro base do AIDev."""


class ConfigurationError(AIDevError):

    """Configuração inválida ou incompleta (ex.: variáveis de ambiente ausentes)."""


class LLMError(AIDevError):

    """Erro relacionado à LLM."""


class LLMRateLimitError(LLMError):

    """Limite de requisições atingido."""


class LLMAPIError(LLMError):

    """Erro retornado pela API da LLM.

    ``status_code`` é usado pelo LLMRouter para decidir se o erro é
    transitório (ex.: 5xx, indisponibilidade momentânea) e portanto vale
    tentar outro provider, ou se é um erro definitivo (ex.: 4xx de request
    malformada) que não seria resolvido trocando de provider.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_transient(self) -> bool:
        if self.status_code is None:
            return True

        return self.status_code >= 500


class LLMConnectionError(LLMError):

    """Falha de rede/timeout ao comunicar com a API da LLM."""


class LLMInvalidResponseError(LLMError):

    """Resposta da LLM inválida ou incompatível com o contexto esperado."""
