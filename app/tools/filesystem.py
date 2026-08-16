from pathlib import Path

PROJECTS_DIR = Path("projects").resolve()

def list_files(project_name: str) -> list[str]:
    project_dir = (PROJECTS_DIR / project_name).resolve()

    if PROJECTS_DIR not in project_dir.parents:
        raise PermissionError(
            "Acesso fora do diretório de projetos não permitido."
        )

    if not project_dir.exists():
        raise FileNotFoundError(
            f"Projeto não encontrado: {project_name}"
        )

    if not project_dir.is_dir():
        raise NotADirectoryError(
            f"O caminho não é um diretório: {project_name}"
        )

    return [
        str(path.relative_to(project_dir))
        for path in project_dir.rglob("*")
        if path.is_file()
    ]

LIST_FILES_DEFINITION = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Lista todos os arquivos de um projeto.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto."
                }
            },
            "required": ["project_name"]
        }
    }
}