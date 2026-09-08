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
            "EDIT_ERROR: 'old_text' não pode ser vazio. "
            "Forneça o trecho exato a ser substituído."
        )
    if not isinstance(new_text, str):
        raise ValueError(
            "EDIT_ERROR: 'new_text' deve ser uma string."
        )

    projects_dir = get_projects_dir()
    project_dir = (projects_dir / project_name).resolve()
    target_file = (project_dir / file_path).resolve()

    if not project_dir.is_relative_to(projects_dir):
        raise PermissionError(
            "Acesso fora do diretório de projetos não permitido."
        )

    if not target_file.is_relative_to(project_dir):
        raise PermissionError("Acesso fora do projeto não permitido.")

    if not target_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if not target_file.is_file():
        raise IsADirectoryError(f"O caminho não é um arquivo: {file_path}")

    content = target_file.read_text(encoding="utf-8")

    occurrences = content.count(old_text)

    if occurrences == 0:
        raise ValueError(
            "EDIT_ERROR: trecho não encontrado. "
            f"Nenhuma ocorrência de 'old_text' em '{file_path}'. "
            "Verifique o conteúdo atual do arquivo (read_file) e "
            "forneça o trecho exato."
        )

    if occurrences > 1:
        raise ValueError(
            f"EDIT_ERROR: múltiplas ocorrências ({occurrences}). "
            f"O trecho aparece {occurrences} vezes em '{file_path}' — "
            "a edição foi abortada para evitar alteração ambígua. "
            "Forneça um trecho mais específico (com mais contexto) "
            "que ocorra exatamente uma vez."
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
            "Modifica pontualmente um arquivo existente (o arquivo "
            "deve existir — para criar, use write_file). Substitui o "
            "trecho exato 'old_text' por 'new_text'; a edição só é "
            "aplicada se o trecho ocorrer exatamente uma vez "
            "(não encontrado ou ambíguo retorna erro, sem modificar "
            "nada). Indicado para mudanças localizadas; para uma "
            "reescrita substancial, use write_file."
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
                "old_text": {
                    "type": "string",
                    "description": (
                        "Trecho exato a ser localizado (deve ocorrer "
                        "exatamente uma vez no arquivo)."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "Texto que substitui o trecho.",
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
