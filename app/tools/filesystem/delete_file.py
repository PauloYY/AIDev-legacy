from pathlib import Path

from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


# Componentes de caminho que nunca podem ser removidos via delete_file.
# Razoável e mínima: controle de versão e estado interno do agente.
# (O .git do próprio AIDev já está fora do workspace; aqui protegemos
# o .git de cada projeto + o estado persistido do TaskState.)
PROTECTED_PATH_PARTS = frozenset({".git", ".aidev"})


def _is_protected(relative_path: str) -> str | None:
    """Retorna o motivo da proteção, ou None se o caminho é permitido."""
    parts = Path(relative_path).parts
    for part in parts:
        if part in PROTECTED_PATH_PARTS:
            return (
                f"DELETE_ERROR: protected path ('{part}'). "
                "Removing version control or internal agent state "
                "via delete_file is not allowed."
            )
    return None


def delete_file(project_name: str, file_path: str) -> str:
    """Remove um arquivo do workspace (sem recursão de diretórios)."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(
            "DELETE_ERROR: 'file_path' cannot be empty."
        )

    projects_dir = get_projects_dir()
    project_dir = (projects_dir / project_name).resolve()
    target = (project_dir / file_path).resolve()

    if not project_dir.is_relative_to(projects_dir):
        raise PermissionError(
            "Access outside the projects directory is not allowed."
        )

    if not target.is_relative_to(project_dir):
        raise PermissionError("Access outside the project is not allowed.")

    protected_reason = _is_protected(file_path)
    if protected_reason is not None:
        raise PermissionError(protected_reason)

    # Nunca remover a raiz do projeto em si.
    if target == project_dir:
        raise PermissionError(
            "DELETE_ERROR: removing the project root is not allowed."
        )

    if not target.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not target.is_file():
        raise IsADirectoryError(
            f"Path is not a file (directories cannot be "
            f"removed via delete_file): {file_path}"
        )

    # Symlinks: remover o link, nunca seguir para fora do projeto.
    target.unlink()

    return f"DELETE_SUCCESS\nfile: {file_path}"


definition = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": (
            "Removes a clearly unnecessary file from the project "
            "(e.g. temporary, duplicated or superseded). Does not "
            "remove directories or protected paths (.git, .aidev). "
            "Returns a controlled error if the file does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Caminho do arquivo dentro do projeto.",
                },
            },
            "required": [
                "project_name",
                "file_path",
            ],
        },
    },
}


tool = Tool(
    name="delete_file",
    function=delete_file,
    definition=definition,
    type=ToolType.EXECUTION,
    pure=False,
)
