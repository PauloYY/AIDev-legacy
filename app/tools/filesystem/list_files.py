from app.tools.base import Tool
from app.tools.filesystem.config import PROJECTS_DIR


def list_files(project_name: str) -> list[str]:
    project_path = (PROJECTS_DIR / project_name).resolve()

    if not project_path.is_relative_to(PROJECTS_DIR):
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

    return [
        str(path.relative_to(project_path))
        for path in project_path.rglob("*")
        if path.is_file()
    ]


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
)