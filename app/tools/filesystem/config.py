from pathlib import Path

from app.config import Config


def get_projects_dir() -> Path:
    """Resolve o diretório-raiz de projetos a cada chamada.

    Usar uma função (em vez de uma constante calculada em tempo de import)
    permite configurar AIDEV_PROJECTS_DIR dinamicamente e também facilita
    testes, que podem sobrescrever app.config.Config.projects_dir sem
    precisar recarregar módulos.
    """

    return Path(Config.projects_dir).resolve()
