import logging
import os
import sys
from logging.handlers import RotatingFileHandler


# P6: rotação do log da aplicação (stdlib, sem dependências novas).
# Defaults seguros: 10 MB por arquivo + 3 backups (aidev.log.1..3).
# Sobrescrevíveis por env; valores inválidos caem para o default.
DEFAULT_LOG_MAX_MB = 10
DEFAULT_LOG_BACKUPS = 3


def _positive_int_env(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except (TypeError, ValueError):
        return default


def log_rotation_limits() -> tuple[int, int]:
    """(max_bytes_por_arquivo, nº_de_backups) vigentes (P6, só observa)."""
    max_mb = _positive_int_env("AIDEV_LOG_MAX_MB", DEFAULT_LOG_MAX_MB)
    backups = _positive_int_env("AIDEV_LOG_BACKUPS", DEFAULT_LOG_BACKUPS)
    return max(1, max_mb) * 1024 * 1024, max(0, backups)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "aidev.log",
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Configura o logging da aplicação.

    Mensagens de progresso voltadas ao usuário (eventos do agente) continuam
    sendo impressas via ``print`` em app.cli.events — isso é UI, não log.
    O logging aqui cobre diagnóstico interno: fallback de providers, retries
    de rede, avisos de configuração, stack traces de erros inesperados etc.

    P6: o arquivo de log usa rotação por tamanho (stdlib
    RotatingFileHandler): ao exceder `max_bytes`, o atual vira
    `<log>.1` (até `backup_count` backups). `max_bytes`/`backup_count`
    None → vigentes via `log_rotation_limits()`. Formato inalterado.
    """

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        if max_bytes is None or backup_count is None:
            default_bytes, default_backups = log_rotation_limits()
            if max_bytes is None:
                max_bytes = default_bytes
            if backup_count is None:
                backup_count = default_backups
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8",
        ))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # httpx é bem verboso em DEBUG (loga corpo de requests); mantemos em WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
