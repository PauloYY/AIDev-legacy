import subprocess

from app.config import Config
from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir
from app.tools.execution import sandbox


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
            "Acesso fora do diretório de projetos não permitido."
        )

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Projeto não encontrado: {project_name}"
        )

    if not command or not command.strip():
        raise ValueError("O comando não pode ser vazio.")

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
            result = subprocess.run(
                command,
                shell=True,
                cwd=project_dir,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

    except subprocess.TimeoutExpired:
        return (
            f"TIMEOUT: a execução excedeu {timeout}s e foi interrompida.\n"
            "Possíveis causas: loop infinito, o processo esperou uma "
            "entrada que não foi fornecida via 'stdin', ou o comando "
            "inicia um processo que não termina sozinho (ex.: 'npm "
            "start' ou um servidor web ficam rodando indefinidamente — "
            "prefira comandos que terminam, como 'npm test' ou 'npm "
            "run build')."
        )

    return _format_result(result)


def _format_result(result: subprocess.CompletedProcess) -> str:
    status = "sucesso" if result.returncode == 0 else "falha"

    return (
        f"STATUS: {status} (exit code {result.returncode})\n\n"
        f"STDOUT:\n{result.stdout.strip() or '(vazio)'}\n\n"
        f"STDERR:\n{result.stderr.strip() or '(vazio)'}"
    )


definition = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Executa um comando de shell dentro da pasta do projeto, "
            "para validar código em QUALQUER linguagem ou framework "
            "disponível na imagem de sandbox (ex.: 'python main.py', "
            "'python -m pytest -q', 'npm test', 'npm run build', "
            "'node app.js', 'gcc main.c -o main && ./main', "
            "'g++ main.cpp -o main && ./main', "
            "'javac Main.java && java Main', 'go run .', "
            "'go test ./...', 'cargo run', 'cargo test', "
            "'ruby main.rb', 'php main.php'). Use esta tool para "
            "VALIDAR que o código realmente funciona antes de "
            "considerar uma task ou o objetivo concluído — não assuma "
            "que está correto apenas por tê-lo escrito. Evite comandos "
            "que não terminam sozinhos (servidores como 'npm start', "
            "'flask run') — eles vão estourar o timeout. Para "
            "programas interativos, use 'stdin' para simular entradas "
            "do usuário."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto.",
                },
                "command": {
                    "type": "string",
                    "description": (
                        "Comando de shell a executar dentro da pasta do "
                        "projeto (ex.: 'npm test', 'python main.py')."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Entradas simuladas de usuário, separadas por "
                        "quebra de linha (opcional)."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": (
                        "Tempo máximo de execução em segundos (opcional, "
                        f"padrão: {Config.execution_timeout_seconds})."
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


tool = Tool(
    name="run_command",
    function=run_command,
    definition=definition,
    type=ToolType.ANALYSIS,
)