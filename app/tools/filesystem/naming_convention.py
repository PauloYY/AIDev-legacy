import re
from pathlib import Path

# Segmentos finais que não fazem parte do "nome" em si — sufixos de
# convenção largamente usados (arquivo de teste, tipagem, minificado)
# que, se não removidos antes de classificar o estilo, gerariam falso
# positivo (ex.: "RideService.test.js" pareceria dot.case, mesmo o
# arquivo sendo claramente PascalCase + sufixo de teste).
_MARKER_SEGMENTS = {"test", "tests", "spec", "specs", "min", "d"}


def _strip_marker_segment(stem: str) -> str:
    if "." not in stem:
        return stem

    parts = stem.split(".")

    if len(parts) > 1 and parts[-1].lower() in _MARKER_SEGMENTS:
        return ".".join(parts[:-1])

    return stem


def classify_name_style(stem: str) -> str | None:
    """Classifica o estilo de nomenclatura de `stem` (nome do arquivo
    sem a extensão final).

    Retorna None para nomes de uma palavra só (ex.: "utils",
    "Repository", "index") — são ambíguos, compatíveis com qualquer
    convenção, então não servem nem para estabelecer nem para violar
    um padrão.
    """

    stem = _strip_marker_segment(stem)

    if "-" in stem:
        return "kebab-case"

    if "_" in stem:
        return "snake_case"

    if "." in stem:
        return "dot.case"

    if re.search(r"[a-z][A-Z]", stem):
        return "PascalCase" if stem[:1].isupper() else "camelCase"

    return None


def detect_folder_convention(directory: Path, suffix: str) -> str | None:
    """Olha os arquivos irmãos (mesma pasta, mesma extensão) e
    retorna o estilo de nomenclatura já estabelecido, ou None se não
    houver um estilo único pra impor — pasta inexistente/vazia, só
    nomes ambíguos (uma palavra), ou os irmãos já estão inconsistentes
    entre si (nesse caso não forçamos nada retroativamente).
    """

    if not directory.is_dir():
        return None

    styles = set()

    for sibling in directory.iterdir():
        if not sibling.is_file() or sibling.suffix != suffix:
            continue

        style = classify_name_style(sibling.stem)

        if style is not None:
            styles.add(style)

    if len(styles) == 1:
        return next(iter(styles))

    return None


def check_naming_convention(
    directory: Path,
    suffix: str,
    new_stem: str,
) -> str | None:
    """Retorna uma mensagem de erro se `new_stem` quebra a convenção
    de nomenclatura já estabelecida na pasta, ou None se não há
    problema (sem convenção a impor, nome novo ambíguo, ou já está
    no estilo certo).
    """

    convention = detect_folder_convention(directory, suffix)

    if convention is None:
        return None

    new_style = classify_name_style(new_stem)

    if new_style is None or new_style == convention:
        return None

    return (
        f"Existing files in '{directory.name}/' follow the "
        f"{convention} pattern, but the new file name uses "
        f"{new_style}. Rename the file to keep the convention "
        "already established in that folder."
    )