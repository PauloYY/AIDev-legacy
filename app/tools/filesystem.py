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

def read_file(project_name: str, file_path: str) -> str:
    project_dir = (PROJECTS_DIR / project_name).resolve()
    target_file = (project_dir / file_path).resolve()

    if PROJECTS_DIR not in project_dir.parents:
        raise PermissionError("Acesso fora do diretório de projetos não permitido.")

    if project_dir not in target_file.parents:
        raise PermissionError("Acesso fora do projeto não permitido.")

    if not target_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if not target_file.is_file():
        raise IsADirectoryError(f"O caminho não é um arquivo: {file_path}")

    return target_file.read_text(encoding="utf-8")

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

READ_FILE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Lê o conteúdo de um arquivo dentro de um projeto.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto."
                },
                "file_path": {
                    "type": "string",
                    "description": "Caminho do arquivo relativo ao projeto."
                }
            },
            "required": [
                "project_name",
                "file_path"
            ]
        }
    }
}

WRITE_FILE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Cria ou sobrescreve um arquivo dentro de um projeto.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nome do projeto."
                },
                "file_path": {
                    "type": "string",
                    "description": "Caminho do arquivo dentro do projeto."
                },
                "content": {
                    "type": "string",
                    "description": "Conteúdo completo do arquivo."
                }
            },
            "required": [
                "project_name",
                "file_path",
                "content"
            ]
        }
    }
}