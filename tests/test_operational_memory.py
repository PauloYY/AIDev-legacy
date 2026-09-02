import pytest

from app.agent.context.operational_memory import OperationalMemory
from app.tools.registry import ToolRegistry


@pytest.fixture
def memory(projects_root):
    tools = ToolRegistry()
    tools.load_defaults()
    return OperationalMemory(tools)


def test_render_history_empty(memory):
    assert "Nenhuma ação" in memory.render_history()


def test_record_appears_in_history(memory):
    memory.record(
        iteration=1,
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
        result="Arquivo escrito com sucesso: a.py",
        success=True,
    )

    history = memory.render_history()

    assert "write_file" in history
    assert "OK" in history
    assert "a.py" in history


def test_failed_action_marked_as_falhou(memory):
    memory.record(
        iteration=2,
        tool="read_file",
        arguments={"project_name": "p", "file_path": "missing.py"},
        result="ERRO: arquivo não encontrado",
        success=False,
    )

    assert "FALHOU" in memory.render_history()


def test_history_respects_max_actions(memory):
    for i in range(OperationalMemory.MAX_ACTIONS + 5):
        memory.record(
            iteration=i,
            tool="read_file",
            arguments={},
            result="ok",
            success=True,
        )

    lines = memory.render_history().splitlines()
    assert len(lines) == OperationalMemory.MAX_ACTIONS


def test_reset_clears_history(memory):
    memory.record(
        iteration=1,
        tool="read_file",
        arguments={},
        result="ok",
        success=True,
    )
    memory.reset()

    assert "Nenhuma ação" in memory.render_history()


def test_render_files_reflects_real_disk_state(memory, projects_root):
    project_dir = projects_root / "demo"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("print(1)")

    files = memory.render_files("demo")

    assert "main.py" in files


def test_render_combines_history_and_files(memory, projects_root):
    (projects_root / "demo").mkdir()

    output = memory.render("demo")

    assert "HISTÓRICO DE AÇÕES" in output
    assert "ARQUIVOS ATUAIS DO PROJETO" in output