import logging
import sys


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "aidev.log",
) -> None:
    """Configura o logging da aplicação.

    Mensagens de progresso voltadas ao usuário (eventos do agente) continuam
    sendo impressas via ``print`` em app.cli.events — isso é UI, não log.
    O logging aqui cobre diagnóstico interno: fallback de providers, retries
    de rede, avisos de configuração, stack traces de erros inesperados etc.
    """

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # httpx é bem verboso em DEBUG (loga corpo de requests); mantemos em WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
