import subprocess

import pytest

from app.config import Config
from app.tools.execution import sandbox
from app.tools.execution.run_command import run_command
from app.tools.filesystem.write_file import write_file


def test_run_command_success(projects_root):
    write_file("proj", "main.py", "print('hello')")

    result = run_command("proj", "python3 main.py")

    assert "STATUS: sucesso (exit code 0)" in result
    assert "hello" in result


def test_run_command_failure_captures_output(projects_root):
    write_file("proj", "boom.py", "raise ValueError('deu ruim')")

    result = run_command("proj", "python3 boom.py")

    assert "STATUS: falha" in result
    assert "deu ruim" in result


def test_run_command_with_stdin(projects_root):
    write_file("proj", "menu.py", "nome = input()\nprint(f'ola {nome}')")

    result = run_command("proj", "python3 menu.py", stdin="Paulo\n")

    assert "ola Paulo" in result


def test_run_command_timeout(projects_root):
    write_file("proj", "loop.py", "while True:\n    pass\n")

    result = run_command("proj", "python3 loop.py", timeout_seconds=1)

    assert "TIMEOUT" in result


def test_run_command_supports_shell_chaining(projects_root):
    write_file("proj", "a.py", "print('a')")

    result = run_command("proj", "python3 a.py && echo done")

    assert "STATUS: sucesso" in result
    assert "done" in result


def test_run_command_unknown_binary_is_a_failure_not_an_exception(projects_root):
    write_file("proj", "placeholder.txt", "x")

    result = run_command("proj", "comando-que-nao-existe-123")

    assert "STATUS: falha" in result


def test_run_command_empty_raises(projects_root):
    write_file("proj", "placeholder.txt", "x")

    with pytest.raises(ValueError):
        run_command("proj", "   ")


def test_run_command_project_not_found(projects_root):
    with pytest.raises(FileNotFoundError):
        run_command("nao_existe", "echo oi")


def test_run_command_blocks_path_traversal(projects_root):
    with pytest.raises(PermissionError):
        run_command("../fora", "echo oi")


def test_run_command_uses_docker_sandbox_when_enabled(projects_root, monkeypatch):
    monkeypatch.setattr(Config, "sandbox_mode", "docker")

    write_file("proj", "main.py", "print('hello')")

    captured = {}

    def fake_run_sandboxed(project_dir, command, timeout, stdin=None):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="hello\n", stderr="")

    monkeypatch.setattr(sandbox, "run_sandboxed", fake_run_sandboxed)

    result = run_command("proj", "python main.py")

    assert "STATUS: sucesso" in result
    assert captured["command"] == ["/bin/sh", "-c", "python main.py"]