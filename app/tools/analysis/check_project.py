import subprocess
from pathlib import Path

from app.config import Config
from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir
from app.tools.execution import sandbox


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "target",
    "build",
    "dist",
}

JAVAC_OUT_DIR = "/tmp/aidev_check_java"


def _run(project_dir: Path, command: list[str], timeout: int):
    try:
        if Config.sandbox_mode == "docker":
            return sandbox.run_sandboxed(project_dir, command, timeout)

        return subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr=f"TIMEOUT: excedeu {timeout}s.",
        )

    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr=(
                f"Comando '{command[0]}' não encontrado na imagem de "
                "sandbox — instale a ferramenta ou pule esta checagem."
            ),
        )


def _iter_files(project_dir: Path, *suffixes: str):
    for suffix in suffixes:
        for path in project_dir.rglob(f"*{suffix}"):
            if not path.is_file():
                continue

            if any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue

            yield path


def _format(label: str, result: subprocess.CompletedProcess) -> str:
    status = "OK" if result.returncode == 0 else "FALHOU"

    return (
        f"{label}: {status}\n"
        f"STDOUT:\n{result.stdout.strip() or '(vazio)'}\n"
        f"STDERR:\n{result.stderr.strip() or '(vazio)'}"
    )


def _check_per_file(
    project_dir: Path,
    timeout: int,
    files: list[Path],
    build_command,
    label: str,
    syntax_only: bool,
) -> str:
    """Roda `build_command(relative_path)` em cada arquivo e agrega os
    resultados num único relatório."""

    if not files:
        return f"{label}: nenhum arquivo encontrado."

    failures = []

    for path in files:
        relative = path.relative_to(project_dir)
        result = _run(project_dir, build_command(str(relative)), timeout)

        if result.returncode != 0:
            failures.append(f"{relative}:\n{result.stderr.strip()}")

    if failures:
        return f"{label} — FALHOU em {len(failures)} arquivo(s):\n\n" + (
            "\n\n".join(failures)
        )

    suffix = (
        "\nAVISO: isso valida só sintaxe, NÃO confirma que imports/"
        "requires apontam para módulos que existem de verdade."
        if syntax_only
        else ""
    )

    return f"{label}: {len(files)} arquivo(s) OK.{suffix}"


def _check_go(project_dir: Path, timeout: int) -> str:
    result = _run(project_dir, ["go", "vet", "./..."], timeout)
    return _format("go vet ./... (Go — pega imports/tipos errados)", result)


def _check_node(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".js")),
        lambda relative: ["node", "--check", relative],
        "node --check (JS, apenas sintaxe)",
        syntax_only=True,
    )


def _check_python(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".py")),
        lambda relative: ["python3", "-m", "py_compile", relative],
        "py_compile (Python, apenas sintaxe)",
        syntax_only=True,
    )


def _check_ruby(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".rb")),
        lambda relative: ["ruby", "-c", relative],
        "ruby -c (Ruby, apenas sintaxe)",
        syntax_only=True,
    )


def _check_php(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".php")),
        lambda relative: ["php", "-l", relative],
        "php -l (PHP, apenas sintaxe)",
        syntax_only=True,
    )


def _check_c(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".c")),
        lambda relative: [
            "gcc", "-fsyntax-only", "-Wall", "-Wextra", relative,
        ],
        "gcc -fsyntax-only (C — pega #include errado)",
        syntax_only=False,
    )


def _check_cpp(project_dir: Path, timeout: int) -> str:
    return _check_per_file(
        project_dir,
        timeout,
        list(_iter_files(project_dir, ".cpp", ".cc", ".cxx")),
        lambda relative: [
            "g++", "-fsyntax-only", "-Wall", "-Wextra",
            "-std=c++17", relative,
        ],
        "g++ -fsyntax-only (C++ — pega #include errado)",
        syntax_only=False,
    )


def _check_java(project_dir: Path, timeout: int) -> str:
    java_files = list(_iter_files(project_dir, ".java"))

    if not java_files:
        return "javac (Java): nenhum arquivo .java encontrado."

    relatives = [
        str(path.relative_to(project_dir)) for path in java_files
    ]

    # Compila todos os arquivos juntos numa passada só, em vez de um
    # por um — assim referências entre classes de arquivos diferentes
    # são resolvidas corretamente (não gera falso positivo por causa
    # de dependência cruzada).
    result = _run(
        project_dir,
        ["javac", "-d", JAVAC_OUT_DIR, "-Xlint:all", *relatives],
        timeout,
    )

    return _format(
        f"javac (Java — {len(relatives)} arquivo(s), pega imports/"
        "tipos errados)",
        result,
    )


def _check_rust(project_dir: Path, timeout: int) -> str:
    # --offline porque o sandbox não tem rede; se houver dependências
    # não vendorizadas/cacheadas, essa checagem falha por causa disso
    # e não por um erro real de código — o relatório abaixo deixa isso
    # visível no stderr.
    result = _run(
        project_dir,
        ["cargo", "check", "--offline"],
        timeout,
    )
    return _format(
        "cargo check --offline (Rust — pega imports/tipos errados)",
        result,
    )


def check_project(project_name: str) -> str:
    """Roda uma checagem estática (compilação/sintaxe) em TODOS os
    arquivos do projeto, não só nos que os testes tocam. Detecta a
    linguagem automaticamente pelos arquivos de manifesto/extensões
    presentes. Suporta Go, Node/JS, Python, Java, Rust, C, C++, Ruby
    e PHP.

    Use esta tool depois de escrever ou alterar vários arquivos, antes
    de considerar uma etapa concluída — ela pega erros (import/include
    errado, sintaxe quebrada) em arquivos que os testes existentes
    talvez nunca cheguem a executar.
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

    timeout = Config.execution_timeout_seconds
    reports = []

    if (project_dir / "go.mod").exists():
        reports.append(_check_go(project_dir, timeout))

    if (project_dir / "package.json").exists():
        reports.append(_check_node(project_dir, timeout))

    if (
        (project_dir / "requirements.txt").exists()
        or (project_dir / "pyproject.toml").exists()
        or any(_iter_files(project_dir, ".py"))
    ):
        reports.append(_check_python(project_dir, timeout))

    if (
        (project_dir / "pom.xml").exists()
        or (project_dir / "build.gradle").exists()
        or any(_iter_files(project_dir, ".java"))
    ):
        reports.append(_check_java(project_dir, timeout))

    if (project_dir / "Cargo.toml").exists():
        reports.append(_check_rust(project_dir, timeout))

    if any(_iter_files(project_dir, ".c")):
        reports.append(_check_c(project_dir, timeout))

    if any(_iter_files(project_dir, ".cpp", ".cc", ".cxx")):
        reports.append(_check_cpp(project_dir, timeout))

    if (
        (project_dir / "Gemfile").exists()
        or any(_iter_files(project_dir, ".rb"))
    ):
        reports.append(_check_ruby(project_dir, timeout))

    if (
        (project_dir / "composer.json").exists()
        or any(_iter_files(project_dir, ".php"))
    ):
        reports.append(_check_php(project_dir, timeout))

    if not reports:
        return (
            "Nenhum tipo de projeto reconhecido (procurei manifestos "
            "e extensões de Go, Node/JS, Python, Java, Rust, C, C++, "
            "Ruby e PHP)."
        )

    return "\n\n".join(reports)


definition = {
    "type": "function",
    "function": {
        "name": "check_project",
        "description": (
            "Roda uma checagem estática (compilação/sintaxe) em TODOS "
            "os arquivos do projeto de uma vez, detectando a "
            "linguagem automaticamente (Go, Node/JS, Python, Java, "
            "Rust, C, C++, Ruby, PHP). Diferente de rodar os testes, "
            "isso cobre também arquivos que os testes atuais não "
            "exercitam, então pega erros como imports/includes com "
            "nome errado ou sintaxe quebrada em código ainda não "
            "conectado ao resto do projeto. Em linguagens compiladas "
            "(Go, Java, Rust, C, C++) isso pega import/include errado "
            "de verdade; em linguagens dinâmicas (JS, Python, Ruby, "
            "PHP) só garante sintaxe válida, não que os imports "
            "resolvem. Use depois de escrever ou alterar vários "
            "arquivos, antes de considerar uma etapa concluída."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto.",
                },
            },
            "required": [
                "project_name",
            ],
        },
    },
}


tool = Tool(
    name="check_project",
    function=check_project,
    definition=definition,
    type=ToolType.ANALYSIS,
)