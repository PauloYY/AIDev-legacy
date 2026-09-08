from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


def read_file(project_name: str, file_path: str) -> str:
    projects_dir = get_projects_dir()
    project_dir = (projects_dir / project_name).resolve()
    target_file = (project_dir / file_path).resolve()

    if not project_dir.is_relative_to(projects_dir):
        raise PermissionError(
            "Access outside the projects directory is not allowed."
        )

    if not target_file.is_relative_to(project_dir):
        raise PermissionError("Access outside the project is not allowed.")

    if not target_file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not target_file.is_file():
        raise IsADirectoryError(f"Path is not a file: {file_path}")

    return target_file.read_text(encoding="utf-8")


definition = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Reads the content of a file inside a project.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project name.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path of the file inside the project.",
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
    pure=True,
)
