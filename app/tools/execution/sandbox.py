import os
import shutil
import subprocess
import uuid
from pathlib import Path

from app.config import Config


NETWORK_MODE_NONE = "none"
NETWORK_MODE_RESTRICTED = "restricted"

PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def sandbox_available() -> tuple[bool, str | None]:
    """Verifica se o sandbox Docker está pronto para uso.

    Retorna (True, None) se estiver tudo certo, ou (False, motivo) caso
    contrário. Usado tanto na checagem de startup (main.py) quanto para
    diagnosticar problemas de forma clara em vez de deixar o agente
    descobrir isso no meio de uma execução.

    Quando AIDEV_SANDBOX_NETWORK_MODE=restricted, também garante (de
    forma idempotente) que a rede interna e o container do proxy de
    egress existam e estejam rodando — sem isso, containers do sandbox
    ficariam sem rede nenhuma mesmo em modo "restricted".
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

    if Config.sandbox_network_mode == NETWORK_MODE_RESTRICTED:
        try:
            _ensure_restricted_network_ready()

        except subprocess.CalledProcessError as error:
            return False, (
                "Não foi possível preparar a rede restrita do sandbox "
                f"(rede: {Config.sandbox_network_name}, proxy: "
                f"{Config.sandbox_proxy_container_name}): {error}\n"
                "Verifique se a imagem do proxy foi construída: "
                f"'docker build -t {Config.sandbox_proxy_image} "
                "docker/squid'."
            )

    return True, None


def _ensure_restricted_network_ready() -> None:
    """Garante rede interna + container de proxy prontos (idempotente).

    Chamado toda vez que o sandbox é verificado, então precisa ser
    seguro de rodar repetidamente — se a rede/proxy já existem e estão
    de pé, não faz nada.
    """

    _ensure_network(Config.sandbox_network_name)
    _ensure_proxy_container(
        network_name=Config.sandbox_network_name,
        container_name=Config.sandbox_proxy_container_name,
        image=Config.sandbox_proxy_image,
    )


def _network_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "network", "inspect", name],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def _ensure_network(name: str) -> None:
    if _network_exists(name):
        return

    subprocess.run(
        ["docker", "network", "create", "--internal", name],
        capture_output=True,
        timeout=10,
        check=True,
    )


def _proxy_container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _ensure_proxy_container(
    network_name: str,
    container_name: str,
    image: str,
) -> None:
    if _proxy_container_running(container_name):
        return

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=10,
    )

    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", network_name,
            "--restart", "unless-stopped",
            image,
        ],
        capture_output=True,
        timeout=30,
        check=True,
    )

    subprocess.run(
        ["docker", "network", "connect", "bridge", container_name],
        capture_output=True,
        timeout=10,
    )


def _network_arguments() -> list[str]:
    if Config.sandbox_network_mode == NETWORK_MODE_RESTRICTED:
        return ["--network", Config.sandbox_network_name]

    return ["--network", NETWORK_MODE_NONE]


def _proxy_env_arguments() -> list[str]:
    if Config.sandbox_network_mode != NETWORK_MODE_RESTRICTED:
        return []

    proxy_url = (
        f"http://{Config.sandbox_proxy_container_name}:"
        f"{Config.sandbox_proxy_port}"
    )

    arguments = []

    for var in PROXY_ENV_VARS:
        arguments += ["-e", f"{var}={proxy_url}"]

    return arguments


def _user_arguments() -> list[str]:
    """Faz o container rodar com o mesmo UID/GID do usuário do host.

    Sem isso, o container roda como root (padrão da imagem base) mas,
    por causa do --cap-drop ALL, sem CAP_DAC_OVERRIDE — ou seja, sem o
    privilégio que normalmente deixaria o root ignorar permissões de
    arquivo. Como o diretório do projeto no host pertence ao usuário
    normal (não ao root), isso causava EACCES em qualquer escrita nova
    dentro do sandbox (ex.: 'npm install' criando package-lock.json).

    Rodar como o mesmo UID/GID do host resolve isso sem precisar abrir
    mão do --cap-drop ALL: as permissões já batem naturalmente, do
    mesmo jeito que os arquivos escritos pelo write_file (que roda
    direto no host, fora do Docker) já pertencem a esse usuário.

    HOME=/tmp evita um problema secundário: com um UID arbitrário sem
    entrada em /etc/passwd, ferramentas como o npm tentariam escrever
    cache em /root (que não pertenceria a esse UID) e falhariam do
    mesmo jeito. /tmp tem permissão de escrita para qualquer UID.

    No Windows (sem os.getuid), isso não se aplica.
    """

    if not hasattr(os, "getuid"):
        return []

    return [
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
    ]


def run_sandboxed(
    project_dir: Path,
    command: list[str],
    timeout: int,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Executa `command` isolado num container Docker descartável.

    Proteções aplicadas:
    - Rede: por padrão nenhum acesso (`--network none`), evitando
      exfiltração de dados ou downloads não autorizados pelo código
      gerado. Com AIDEV_SANDBOX_NETWORK_MODE=restricted, o container
      entra numa rede interna (sem rota direta pra internet) e só
      alcança a rede através de um proxy com allowlist de domínios
      (ver docker/squid/) — o suficiente para `npm install`/`pip
      install`/etc. em pacotes de registries oficiais, sem abrir
      acesso à internet de forma irrestrita.
    - Sem capacidades de root e sem escalonamento de privilégio.
    - Limites de memória, CPU e número de processos (evita o código
      gerado esgotar os recursos do host).
    - Nenhuma variável de ambiente do host chega ao container — as API
      keys do AIDev, por exemplo, nunca ficam visíveis para o código
      sendo executado (as únicas variáveis injetadas são as de proxy
      acima, e só em modo restricted).
    - Container é sempre removido ao final (`--rm`), inclusive em caso
      de timeout.
    """

    container_name = f"aidev-{uuid.uuid4().hex[:12]}"

    docker_command = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        *_network_arguments(),
        *_proxy_env_arguments(),
        *_user_arguments(),
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
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=10,
        )
        raise
