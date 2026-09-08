import os
import signal
import subprocess

from app.config import Config
from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir
from app.tools.execution import sandbox


def _run_direct_without_sandbox(
    command: str,
    project_dir,
    stdin: str | None,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Executa `command` via shell fora do Docker, com kill da árvore.

    Com `shell=True` existe um intermediário (`/bin/sh -c ...`) e o
    trabalho real roda como neto. O `subprocess.run()` padrão mata só
    o filho direto no timeout, orfanando o resto (ex.: `python3 loop.py`
    com PPID=1 a 100% CPU). Por isso o processo é iniciado como líder
    de uma nova sessão/grupo (`start_new_session=True`, logo o PGID do
    grupo é o próprio PID criado aqui) e, no timeout, o GRUPO inteiro
    recebe SIGKILL — nunca o grupo do AIDev.
    """

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=project_dir,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        # Drena os pipes e faz reap do filho direto (evita zumbi).
        try:
            proc.communicate()
        except Exception:
            pass
        raise

    return subprocess.CompletedProcess(
        command, proc.returncode, stdout, stderr
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Mata somente o grupo criado para `proc` (PGID == PID do filho).

    Silencioso quando o processo já terminou sozinho na corrida entre
    o timeout e o kill (ProcessLookupError) ou sem permissão.
    """

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    # Trava de segurança: só mata se o grupo for o do próprio filho
    # (líder de sessão criado com start_new_session=True). Nunca o
    # grupo do processo atual.
    if pgid is not None and pgid == proc.pid and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass

    try:
        proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_command(
    project_name: str,
    command: str,
    stdin: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Executa um comando de shell dentro da pasta do projeto.

    Generaliza a validação do agente para QUALQUER linguagem ou framework
    (ex.: "python main.py", "npm test", "node app.js", "go test ./...",
    "cargo run"), desde que o runtime necessário esteja disponível na
    imagem de sandbox (veja docker/Dockerfile). Se o runtime não estiver
    instalado, o comando simplesmente falha com exit code diferente de
    zero — isso é reportado normalmente, não é uma exceção.

    Por padrão (AIDEV_SANDBOX_MODE=docker), a execução acontece isolada
    num container descartável, sem rede e com limites de recursos.
    """

    projects_dir = get_projects_dir()
    project_dir = (projects_dir / project_name).resolve()

    if not project_dir.is_relative_to(projects_dir):
        raise PermissionError(
            "Access outside the projects directory is not allowed."
        )

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Project not found: {project_name}"
        )

    if not command or not command.strip():
        raise ValueError("Command cannot be empty.")

    timeout = timeout_seconds or Config.execution_timeout_seconds

    try:
        if Config.sandbox_mode == "docker":
            result = sandbox.run_sandboxed(
                project_dir,
                ["/bin/sh", "-c", command],
                timeout,
                stdin,
            )

        else:
            result = _run_direct_without_sandbox(
                command,
                project_dir,
                stdin,
                timeout,
            )

    except subprocess.TimeoutExpired:
        return (
            f"TIMEOUT: execution exceeded {timeout}s and was interrupted.\n"
            "Possible causes: infinite loop, the process waited for an "
            "input that was not provided via 'stdin', or the command "
            "starts a process that never terminates on its own (e.g. "
            "'npm start' or a web server keep running indefinitely — "
            "prefer commands that terminate, such as 'npm test' or 'npm "
            "run build')."
        )

    return _format_result(result)


def _format_result(result: subprocess.CompletedProcess) -> str:
    status = "success" if result.returncode == 0 else "failure"

    return (
        f"STATUS: {status} (exit code {result.returncode})\n\n"
        f"STDOUT:\n{result.stdout.strip() or '(empty)'}\n\n"
        f"STDERR:\n{result.stderr.strip() or '(empty)'}"
    )


definition = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Runs a shell command inside the project folder, to "
            "validate code in ANY language or framework available in "
            "the sandbox image (e.g. 'python main.py', "
            "'python -m pytest -q', 'npm test', 'npm run build', "
            "'node app.js', 'gcc main.c -o main && ./main', "
            "'g++ main.cpp -o main && ./main', "
            "'javac Main.java && java Main', 'go run .', "
            "'go test ./...', 'cargo run', 'cargo test', "
            "'ruby main.rb', 'php main.php'). Use this tool to VERIFY "
            "that the code actually works before considering a task "
            "or the objective done — do not assume it is correct just "
            "because you wrote it. Avoid commands that never terminate "
            "on their own (servers like 'npm start', 'flask run') — "
            "they will hit the timeout. For interactive programs, use "
            "'stdin' to simulate user inputs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project name.",
                },
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run inside the project folder "
                        "(e.g. 'npm test', 'python main.py')."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Simulated user inputs, separated by newlines "
                        "(optional)."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": (
                        "Maximum execution time in seconds (optional, "
                        f"default: {Config.execution_timeout_seconds})."
                    ),
                },
            },
            "required": [
                "project_name",
                "command",
            ],
        },
    },
}


# P4: ANALYSIS aqui significa apenas "pode ser dependency" (ex.:
# rodar testes para coletar evidência antes de agir) — NÃO significa
# "sem efeitos colaterais". Qualquer comando shell pode mutar arquivos
# via redirect/heredoc/etc., então run_command é sempre não-puro
# (pure=False, o default): nunca entra em batch paralelo, mesmo para
# comandos com cara de leitura como "--version" ou "ls".
tool = Tool(
    name="run_command",
    function=run_command,
    definition=definition,
    type=ToolType.ANALYSIS,
)