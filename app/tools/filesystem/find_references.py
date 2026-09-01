from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
}


def find_references(
    project_name: str,
    symbol: str,
) -> list[str]:
    projects_dir = get_projects_dir()
    project_path = (projects_dir / project_name).resolve()

    if not project_path.is_relative_to(projects_dir):
        raise PermissionError(
            "Acesso fora do diretório de projetos não permitido."
        )

    if not project_path.exists():
        raise FileNotFoundError(
            f"Projeto não encontrado: {project_name}"
        )

    if not project_path.is_dir():
        raise NotADirectoryError(
            f"O projeto não é um diretório: {project_name}"
        )

    references = []

    for path in project_path.rglob("*"):

        if not path.is_file():
            continue

        if any(
            directory in path.parts
            for directory in IGNORED_DIRECTORIES
        ):
            continue

        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if symbol in line:
                relative_path = path.relative_to(
                    project_path
                )

                references.append(
                    f"{relative_path}:{line_number}: "
                    f"{line.strip()}"
                )

    return references


definition = {
    "type": "function",
    "function": {
        "name": "find_references",
        "description": (
            "Procura referências a um símbolo em todos os arquivos "
            "do projeto. Use esta ferramenta para descobrir onde "
            "classes, funções, atributos ou métodos são utilizados "
            "antes de alterar uma interface existente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto.",
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Nome do símbolo que deve ser procurado."
                    ),
                },
            },
            "required": [
                "project_name",
                "symbol",
            ],
        },
    },
}


tool = Tool(
    name="find_references",
    function=find_references,
    definition=definition,
    type=ToolType.ANALYSIS,
)
