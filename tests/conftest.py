import pytest

from app.config import Config


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    """Isola AIDEV_PROJECTS_DIR num diretório temporário para cada teste."""

    monkeypatch.setattr(Config, "projects_dir", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _default_sandbox_mode(monkeypatch):
    """Por padrão, testes rodam código diretamente (sem exigir Docker).

    Testes que precisam validar especificamente o caminho do Docker devem
    sobrescrever isso explicitamente (veja tests/test_sandbox.py).
    """

    monkeypatch.setattr(Config, "sandbox_mode", "none")