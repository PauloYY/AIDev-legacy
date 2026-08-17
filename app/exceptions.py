class AIDevError(Exception):
    """Erro base do AIDev."""


class LLMError(AIDevError):
    """Erro relacionado à LLM."""


class LLMRateLimitError(LLMError):
    """Limite de requisições atingido."""


class LLMAPIError(LLMError):
    """Erro retornado pela API da LLM."""