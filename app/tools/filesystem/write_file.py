from app.tools.base import Tool, ToolType
from app.tools.filesystem.config import PROJECTS_DIR


def write_file(project_name: str, file_path: str, content: str) -> str:
    project_path = PROJECTS_DIR / project_name
    target_path = project_path / file_path

    try:
        target_path = target_path.resolve()
        project_path = project_path.resolve()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Projeto não encontrado: {project_name}"
        )

    if not target_path.is_relative_to(project_path):
        raise PermissionError(
            "Acesso fora do diretório do projeto não permitido."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")

    return f"Arquivo escrito com sucesso: {file_path}"

definition = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Cria ou sobrescreve um arquivo dentro de um projeto.",
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
                "content": {
                    "type": "string",
                    "description": "Conteúdo completo do arquivo.",
                },
            },
            "required": [
                "project_name",
                "file_path",
                "content",
            ],
        },
    },
}


tool = Tool(
    name="write_file",
    function=write_file,
    definition=definition,
    type=ToolType.EXECUTION,
)