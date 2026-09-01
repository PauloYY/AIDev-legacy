import shutil
import subprocess
import uuid
from pathlib import Path

from app.config import Config


def sandbox_available() -> tuple[bool, str | None]:
    """Verifica se o sandbox Docker está pronto para uso.

    Retorna (True, None) se estiver tudo certo, ou (False, motivo) caso
    contrário. Usado tanto na checagem de startup (main.py) quanto para
    diagnosticar problemas de forma clara em vez de deixar o agente
    descobrir isso no meio de uma execução.
    """

    if shutil.which("docker") is None:
        return False, (
            "O comando 'docker' não foi encontrado no PATH. Instale o "
            "Docker, ou defina AIDEV_SANDBOX_MODE=none no .env para "
            "rodar código diretamente no sistema (sem isolamento)."
        )

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )

    except (subprocess.TimeoutExpired, OSError) as error:
        return False, f"Não foi possível falar com o Docker: {error}"

    if result.returncode != 0:
        return False, (
            "O daemon do Docker não parece estar rodando. Inicie o "
            "Docker (ex.: 'sudo systemctl start docker') e tente de novo."
        )

    return True, None


def run_sandboxed(
    project_dir: Path,
    command: list[str],
    timeout: int,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Executa `command` isolado num container Docker descartável.

    Proteções aplicadas:
    - Sem acesso à rede (`--network none`), evitando exfiltração de dados
      ou downloads não autorizados pelo código gerado.
    - Sem capacidades de root e sem escalonamento de privilégio.
    - Limites de memória, CPU e número de processos (evita o código
      gerado esgotar os recursos do host).
    - Nenhuma variável de ambiente do host chega ao container — as API
      keys do AIDev, por exemplo, nunca ficam visíveis para o código
      sendo executado.
    - Container é sempre removido ao final (`--rm`), inclusive em caso
      de timeout.
    """

    container_name = f"aidev-{uuid.uuid4().hex[:12]}"

    docker_command = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", Config.sandbox_memory_limit,
        "--cpus", Config.sandbox_cpu_limit,
        "--pids-limit", Config.sandbox_pids_limit,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{project_dir}:/workspace:rw",
        "-w", "/workspace",
        Config.sandbox_docker_image,
        *command,
    ]

    try:
        return subprocess.run(
            docker_command,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:
        # Garante que o container não continue rodando em segundo plano
        # mesmo depois do processo cliente do docker ter sido encerrado.
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=10,
        )
        raise