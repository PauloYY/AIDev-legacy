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


def test_run_sandboxed_uses_restricted_network_and_proxy_env(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Config, "sandbox_network_mode", "restricted")
    monkeypatch.setattr(Config, "sandbox_network_name", "test-net")
    monkeypatch.setattr(Config, "sandbox_proxy_container_name", "test-proxy")
    monkeypatch.setattr(Config, "sandbox_proxy_port", 3128)

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox.run_sandboxed(tmp_path, ["npm", "install"], timeout=5)

    command = captured["command"]

    assert command[command.index("--network") + 1] == "test-net"
    assert "none" not in command
    assert "HTTP_PROXY=http://test-proxy:3128" in command
    assert "HTTPS_PROXY=http://test-proxy:3128" in command
    assert "http_proxy=http://test-proxy:3128" in command
    assert "https_proxy=http://test-proxy:3128" in command


def test_ensure_restricted_network_ready_creates_missing_network_and_proxy(
    monkeypatch,
):
    monkeypatch.setattr(Config, "sandbox_network_mode", "restricted")
    monkeypatch.setattr(Config, "sandbox_network_name", "test-net")
    monkeypatch.setattr(Config, "sandbox_proxy_container_name", "test-proxy")
    monkeypatch.setattr(Config, "sandbox_proxy_image", "test-proxy-image")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(command, 1)

        if command[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(command, 1, stdout="")

        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")

    available, reason = sandbox.sandbox_available()

    assert available is True
    assert reason is None

    create_network_calls = [
        c for c in calls
        if c[:3] == ["docker", "network", "create"]
    ]
    assert create_network_calls == [
        ["docker", "network", "create", "--internal", "test-net"]
    ]

    run_proxy_calls = [c for c in calls if c[0:2] == ["docker", "run"]]
    assert len(run_proxy_calls) == 1
    assert "test-proxy-image" in run_proxy_calls[0]

    connect_calls = [
        c for c in calls if c[:3] == ["docker", "network", "connect"]
    ]
    assert connect_calls == [["docker", "network", "connect", "bridge", "test-proxy"]]


def test_ensure_restricted_network_ready_skips_when_already_up(monkeypatch):
    monkeypatch.setattr(Config, "sandbox_network_mode", "restricted")
    monkeypatch.setattr(Config, "sandbox_network_name", "test-net")
    monkeypatch.setattr(Config, "sandbox_proxy_container_name", "test-proxy")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(command, 0)

        if command[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(command, 0, stdout="true")

        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")

    available, reason = sandbox.sandbox_available()

    assert available is True
    assert reason is None
    assert not any(c[0:2] == ["docker", "run"] for c in calls)
    assert not any(c[:3] == ["docker", "network", "create"] for c in calls)


def test_sandbox_available_reports_error_when_proxy_setup_fails(monkeypatch):
    monkeypatch.setattr(Config, "sandbox_network_mode", "restricted")
    monkeypatch.setattr(Config, "sandbox_network_name", "test-net")
    monkeypatch.setattr(Config, "sandbox_proxy_container_name", "test-proxy")
    monkeypatch.setattr(Config, "sandbox_proxy_image", "test-proxy-image")

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(command, 1)

        if command[:3] == ["docker", "network", "create"]:
            raise subprocess.CalledProcessError(1, command)

        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")

    available, reason = sandbox.sandbox_available()

    assert available is False
    assert "test-net" in reason
    assert "test-proxy-image" in reason

def test_run_sandboxed_includes_host_uid_gid_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(sandbox.os, "getgid", lambda: 1000, raising=False)

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox.run_sandboxed(tmp_path, ["npm", "install"], timeout=5)

    command = captured["command"]

    assert "--user" in command
    assert command[command.index("--user") + 1] == "1000:1000"
    assert "HOME=/tmp" in command


def test_run_sandboxed_skips_user_flag_on_windows(monkeypatch, tmp_path):
    monkeypatch.delattr(sandbox.os, "getuid", raising=False)

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox.run_sandboxed(tmp_path, ["npm", "install"], timeout=5)

    assert "--user" not in captured["command"]
