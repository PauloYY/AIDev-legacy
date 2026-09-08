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
            "Access outside the projects directory is not allowed."
        )

    if not project_path.exists():
        raise FileNotFoundError(
            f"Project not found: {project_name}"
        )

    if not project_path.is_dir():
        raise NotADirectoryError(
            f"Project is not a directory: {project_name}"
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
            "Searches for references to a symbol in all project "
            "files. Use this tool to find where classes, functions, "
            "attributes or methods are used before changing an "
            "existing interface."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project name.",
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Name of the symbol to search for."
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
    pure=True,
)
