import os
import subprocess
import time

import pytest

from app.config import Config
from app.tools.execution import sandbox
from app.tools.execution.run_command import run_command
from app.tools.filesystem.write_file import write_file


def _pids_by_script(*names: str) -> set[int]:
    """PIDs vivos cujo cmdline é `python* <name>` (só stdlib, via /proc).

    Compara argv[1] exato para não confundir com outros testes/processos.
    """

    found: set[int] = set()

    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                parts = handle.read().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue

        if (
            len(parts) >= 2
            and parts[0].split(b"/")[-1] in (b"python", b"python3")
            and parts[1].decode(errors="replace") in names
        ):
            found.add(int(pid))

    return found


def _assert_no_new_pids(names: tuple[str, ...], before: set[int], timeout_s: float = 5.0):
    """Aguarda (poll) os PIDs novos sumirem; falha listando sobreviventes."""

    deadline = time.time() + timeout_s
    survivors = (_pids_by_script(*names) - before) or set()

    while survivors and time.time() < deadline:
        time.sleep(0.2)
        survivors = _pids_by_script(*names) - before

    assert not survivors, f"processos sobreviventes após timeout: {sorted(survivors)}"


def test_run_command_success(projects_root):
    write_file("proj", "main.py", "print('hello')")

    result = run_command("proj", "python3 main.py")

    assert "STATUS: success (exit code 0)" in result
    assert "hello" in result


def test_run_command_failure_captures_output(projects_root):
    write_file("proj", "boom.py", "raise ValueError('deu ruim')")

    result = run_command("proj", "python3 boom.py")

    assert "STATUS: failure" in result
    assert "deu ruim" in result


def test_run_command_with_stdin(projects_root):
    write_file("proj", "menu.py", "nome = input()\nprint(f'ola {nome}')")

    result = run_command("proj", "python3 menu.py", stdin="Paulo\n")

    assert "ola Paulo" in result


def test_run_command_timeout(projects_root):
    write_file("proj", "loop.py", "while True:\n    pass\n")

    before = _pids_by_script("loop.py")

    result = run_command("proj", "python3 loop.py", timeout_seconds=1)

    assert "TIMEOUT" in result
    _assert_no_new_pids(("loop.py",), before)


def test_run_command_timeout_kills_grandchild_processes(projects_root):
    write_file("proj", "child_loop.py", "while True:\n    pass\n")
    write_file(
        "proj",
        "parent_loop.py",
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, 'child_loop.py'])\n"
        "while True:\n    pass\n",
    )

    before = _pids_by_script("parent_loop.py", "child_loop.py")

    result = run_command("proj", "python3 parent_loop.py", timeout_seconds=1)

    assert "TIMEOUT" in result
    _assert_no_new_pids(("parent_loop.py", "child_loop.py"), before)


def test_run_command_consecutive_timeouts_leave_no_processes(projects_root):
    write_file("proj", "loop.py", "while True:\n    pass\n")

    before = _pids_by_script("loop.py")

    for _ in range(5):
        result = run_command("proj", "python3 loop.py", timeout_seconds=1)
        assert "TIMEOUT" in result

    _assert_no_new_pids(("loop.py",), before)


def test_run_command_supports_shell_chaining(projects_root):
    write_file("proj", "a.py", "print('a')")

    result = run_command("proj", "python3 a.py && echo done")

    assert "STATUS: success" in result
    assert "done" in result


def test_run_command_unknown_binary_is_a_failure_not_an_exception(projects_root):
    write_file("proj", "placeholder.txt", "x")

    result = run_command("proj", "comando-que-nao-existe-123")

    assert "STATUS: failure" in result


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

    assert "STATUS: success" in result
    assert captured["command"] == ["/bin/sh", "-c", "python main.py"]