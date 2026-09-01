import subprocess

import pytest

from app.config import Config
from app.tools.execution import sandbox


def test_sandbox_available_when_docker_missing(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    available, reason = sandbox.sandbox_available()

    assert available is False
    assert "docker" in reason.lower()


def test_sandbox_available_when_daemon_down(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")

    class FakeResult:
        returncode = 1

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: FakeResult())

    available, reason = sandbox.sandbox_available()

    assert available is False
    assert reason is not None


def test_sandbox_available_ok(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")

    class FakeResult:
        returncode = 0

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: FakeResult())

    available, reason = sandbox.sandbox_available()

    assert available is True
    assert reason is None


def test_run_sandboxed_builds_expected_docker_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    result = sandbox.run_sandboxed(tmp_path, ["python", "main.py"], timeout=5)

    assert result.stdout == "ok"

    command = captured["command"]

    assert command[0:2] == ["docker", "run"]
    assert "--rm" in command
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert f"{tmp_path}:/workspace:rw" in command
    assert Config.sandbox_docker_image in command
    assert command[-2:] == ["python", "main.py"]


def test_run_sandboxed_kills_container_on_timeout(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[0:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=command, timeout=1)

        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        sandbox.run_sandboxed(tmp_path, ["python", "loop.py"], timeout=1)

    kill_calls = [c for c in calls if c[0:2] == ["docker", "kill"]]
    assert len(kill_calls) == 1