from pathlib import Path

from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir

import os


# Fase 3 (OPT-2): diretórios de dependências/build gerados — ruído para
# o Planner (file_list) e custo de walk a cada iteração. Mesmo critério
# já usado por check_project/find_references para ignorar esses dirs.
IGNORED_DIRECTORIES = frozenset({
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "target",
    "build",
    "dist",
})


def list_files(project_name: str) -> list[str]:
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

    visible: list[str] = []

    # os.walk (em vez de rglob) para podar diretórios ignorados antes
    # de descer neles — sem isso, um node_modules com milhares de
    # arquivos seria percorrido integralmente a cada chamada (a tool
    # roda a cada iteração via memória operacional).
    for root, dirnames, filenames in os.walk(project_path):
        dirnames[:] = [
            name for name in dirnames if name not in IGNORED_DIRECTORIES
        ]

        for filename in filenames:
            full_path = Path(root) / filename

            if not full_path.is_file():
                continue

            visible.append(str(full_path.relative_to(project_path)))

    return visible


definition = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Lista todos os arquivos de um projeto.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto.",
                }
            },
            "required": ["project_name"],
        },
    },
}


tool = Tool(
    name="list_files",
    function=list_files,
    definition=definition,
    type=ToolType.ANALYSIS,
    pure=True,
)
