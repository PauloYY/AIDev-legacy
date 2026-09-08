from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir
from app.tools.filesystem.naming_convention import check_naming_convention


def write_file(project_name: str, file_path: str, content: str) -> str:
    projects_dir = get_projects_dir()
    project_path = projects_dir / project_name
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

    is_new_file = not target_path.exists()

    if is_new_file:
        convention_error = check_naming_convention(
            target_path.parent,
            target_path.suffix,
            target_path.stem,
        )

        if convention_error:
            raise ValueError(convention_error)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")

    return f"Arquivo escrito com sucesso: {file_path}"


definition = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Cria um arquivo novo ou reconstrói o conteúdo de um "
            "arquivo existente. Para mudanças pequenas e localizadas "
            "num arquivo existente, prefira edit_file."
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