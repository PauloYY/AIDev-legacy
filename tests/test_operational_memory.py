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

def test_known_test_command_starts_undiscovered(memory):
    assert "ainda não descoberto" in memory.render_known_commands()


def test_records_successful_test_command(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="STATUS: sucesso (exit code 0)",
        success=True,
    )

    rendered = memory.render_known_commands()

    assert "- Teste: npm test" in rendered
    assert "Build: ainda não descoberto" in rendered


def test_only_successful_commands_are_remembered(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="STATUS: falha (exit code 1)",
        success=False,
    )

    assert "Teste: ainda não descoberto" in memory.render_known_commands()


def test_later_successful_test_command_overwrites_earlier_one(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="ok",
        success=True,
    )
    memory.record(
        iteration=5,
        tool="run_command",
        arguments={
            "project_name": "p",
            "command": "node --test tests/validation.test.js tests/expenseService.test.js",
        },
        result="ok",
        success=True,
    )

    rendered = memory.render_known_commands()

    assert "node --test tests/validation.test.js tests/expenseService.test.js" in rendered
    assert "npm test" not in rendered


def test_records_successful_build_command_separately(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm run build"},
        result="ok",
        success=True,
    )

    rendered = memory.render_known_commands()

    assert "- Build: npm run build" in rendered
    assert "Teste: ainda não descoberto" in rendered


def test_non_run_command_tools_do_not_affect_known_commands(memory):
    memory.record(
        iteration=1,
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
        result="ok",
        success=True,
    )

    assert "ainda não descoberto" in memory.render_known_commands()


def test_reset_clears_known_commands(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="ok",
        success=True,
    )
    memory.reset()

    assert "ainda não descoberto" in memory.render_known_commands()


def test_render_includes_known_commands_section(memory, projects_root):
    (projects_root / "demo").mkdir()

    output = memory.render("demo")

    assert "COMANDOS CONHECIDOS QUE JÁ FUNCIONARAM" in output


def test_jest_command_is_recognized_as_test_command(memory):
    # "jest" não contém a substring "test" — precisa estar na lista
    # de keywords explicitamente, senão o comando nunca é lembrado
    # nem reconhecido como gatilho de investigação após falha.
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npx jest --no-coverage"},
        result="STATUS: sucesso (exit code 0)",
        success=True,
    )

    assert "npx jest --no-coverage" in memory.render_known_commands()


def test_last_action_empty_history_is_not_failed_test(memory):
    assert memory.last_run_command_failed_test_or_build() is False


def test_last_action_failed_jest_command_is_detected(memory):
    # Caso real que motivou a checagem: 'npm test' rodando jest por
    # baixo dos panos, retornando exit code 1 com testes falhando.
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result=(
            "STATUS: falha (exit code 1)\n\n"
            "STDOUT:\n> jest --detectOpenHandles\n\n"
            "STDERR:\nFAIL tests/RideService.test.js"
        ),
        success=False,
    )

    assert memory.last_run_command_failed_test_or_build() is True


def test_last_action_failed_build_command_is_detected(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm run build"},
        result="STATUS: falha (exit code 1)",
        success=False,
    )

    assert memory.last_run_command_failed_test_or_build() is True


def test_last_action_successful_test_command_is_not_a_trigger(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="STATUS: sucesso (exit code 0)",
        success=True,
    )

    assert memory.last_run_command_failed_test_or_build() is False


def test_last_action_failed_non_test_command_is_not_a_trigger(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "ls -la"},
        result="STATUS: falha (exit code 1)",
        success=False,
    )

    assert memory.last_run_command_failed_test_or_build() is False


def test_last_action_not_run_command_is_not_a_trigger(memory):
    memory.record(
        iteration=1,
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        result="ERRO: arquivo não encontrado",
        success=False,
    )

    assert memory.last_run_command_failed_test_or_build() is False


def test_only_the_most_recent_action_counts_as_trigger(memory):
    memory.record(
        iteration=1,
        tool="run_command",
        arguments={"project_name": "p", "command": "npm test"},
        result="STATUS: falha (exit code 1)",
        success=False,
    )
    memory.record(
        iteration=2,
        tool="read_file",
        arguments={"project_name": "p", "file_path": "a.py"},
        result="conteúdo do arquivo",
        success=True,
    )

    # O gatilho só vale logo em seguida ao teste que falhou — depois
    # que outra ação já foi registrada, a exceção não se aplica mais.
    assert memory.last_run_command_failed_test_or_build() is False

def test_investigation_budget_available_when_never_used(memory):
    assert memory.investigation_budget_available(iteration=1) is True


def test_investigation_budget_unavailable_right_after_consumption(memory):
    memory.consume_investigation_budget(iteration=10)

    assert memory.investigation_budget_available(iteration=11) is False
    assert memory.investigation_budget_available(iteration=14) is False


def test_investigation_budget_available_again_after_period(memory):
    memory.consume_investigation_budget(iteration=10)

    assert memory.investigation_budget_available(iteration=15) is True
    assert memory.investigation_budget_available(iteration=20) is True


def test_reset_clears_investigation_budget_state(memory):
    memory.consume_investigation_budget(iteration=10)
    memory.reset()

    assert memory.investigation_budget_available(iteration=11) is True