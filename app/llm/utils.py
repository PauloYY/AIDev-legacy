def estimate_tokens(text: str) -> int:
    """Estima o número de tokens a partir do tamanho em caracteres.

    Esta é uma estimativa grosseira: para português/inglês, 1 token
    corresponde aproximadamente a 4 caracteres. Para outros idiomas
    ou textos com muitos símbolos especiais, a precisão pode variar.

    Esta função serve apenas para instrumentação e estimativa de
    tamanho de prompt antes do envio. Os tokens reais vêm sempre
    da resposta da API via UsageTracker.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)
