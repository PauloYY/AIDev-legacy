from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


def read_file(project_name: str, file_path: str) -> str:
    projects_dir = get_projects_dir()
    project_dir = (projects_dir / project_name).resolve()
    target_file = (project_dir / file_path).resolve()

    if not project_dir.is_relative_to(projects_dir):
        raise PermissionError("Acesso fora do diretório de projetos não permitido.")

    if not target_file.is_relative_to(project_dir):
        raise PermissionError("Acesso fora do projeto não permitido.")

    if not target_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if not target_file.is_file():
        raise IsADirectoryError(f"O caminho não é um arquivo: {file_path}")

    return target_file.read_text(encoding="utf-8")


definition = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Lê o conteúdo de um arquivo dentro de um projeto.",
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
    name="read_file",
    function=read_file,
    definition=definition,
    type=ToolType.ANALYSIS,
)
