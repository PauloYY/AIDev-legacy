import os
import tempfile

from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


def edit_file(
    project_name: str, file_path: str, old_text: str, new_text: str
) -> str:
    """Substitui exatamente um trecho de um arquivo (edição por conteúdo).

    Operação atômica na medida do possível: valida tudo antes de
    escrever; se qualquer validação falhar, o arquivo original
    permanece intacto.
    """
    if not isinstance(old_text, str) or old_text == "":
        raise ValueError(
            "EDIT_ERROR: 'old_text' cannot be empty. "
            "Provide the exact snippet to replace."
        )
    if not isinstance(new_text, str):
        raise ValueError(
            "EDIT_ERROR: 'new_text' must be a string."
        )

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

    content = target_file.read_text(encoding="utf-8")

    occurrences = content.count(old_text)

    if occurrences == 0:
        raise ValueError(
            "EDIT_ERROR: snippet not found. "
            f"No occurrence of 'old_text' in '{file_path}'. "
            "Check the current file content (read_file) and "
            "provide the exact snippet."
        )

    if occurrences > 1:
        raise ValueError(
            f"EDIT_ERROR: multiple occurrences ({occurrences}). "
            f"The snippet appears {occurrences} times in '{file_path}' — "
            "the edit was aborted to avoid an ambiguous change. "
            "Provide a more specific snippet (with more context) "
            "that occurs exactly once."
        )

    updated = content.replace(old_text, new_text, 1)

    # Escrita atômica: validações já passaram; escreve em tmp + replace
    # para não deixar o arquivo pela metade em caso de falha de I/O.
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(target_file.parent), prefix=".aidev-edit-"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
        os.replace(tmp_name, target_file)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    # Linha da ocorrência (1-indexed) para o resumo.
    line_number = content[: content.find(old_text)].count("\n") + 1

    return (
        "EDIT_SUCCESS\n"
        f"file: {file_path}\n"
        "replacement: 1 occurrence\n"
        f"line: {line_number}"
    )


definition = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Makes a localized edit to an existing file (the file "
            "must exist — to create one, use write_file), without "
            "rewriting the whole file. Replaces the exact 'old_text' "
            "snippet with 'new_text'; the edit is only applied if the "
            "snippet occurs exactly once (not found or ambiguous "
            "returns an error, modifying nothing). For a substantial "
            "rewrite, use write_file."
        ),
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
                "old_text": {
                    "type": "string",
                    "description": (
                        "Exact snippet to locate (must occur "
                        "exactly once in the file)."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "Text that replaces the snippet.",
                },
            },
            "required": [
                "project_name",
                "file_path",
                "old_text",
                "new_text",
            ],
        },
    },
}


tool = Tool(
    name="edit_file",
    function=edit_file,
    definition=definition,
    type=ToolType.EXECUTION,
    pure=False,
)
