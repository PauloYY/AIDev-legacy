import pytest

from app.config import Config


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    """Isola AIDEV_PROJECTS_DIR num diretório temporário para cada teste."""

    monkeypatch.setattr(Config, "projects_dir", str(tmp_path))
    return tmp_path
