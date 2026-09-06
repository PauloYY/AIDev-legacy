import ast
import re

from app.tools.base import Tool, ToolType
from app.tools.config import get_projects_dir


# --- Python: via AST real, é preciso -----------------------------------

def _list_python_symbols(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as error:
        return [f"(erro de sintaxe ao analisar o arquivo: {error})"]

    all_names = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    all_names = [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ]

    def is_public(name: str) -> bool:
        if all_names is not None:
            return name in all_names
        return not name.startswith("_")

    symbols = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_public(node.name):
                symbols.append(f"{node.name}  (def)")

        elif isinstance(node, ast.ClassDef):
            if is_public(node.name):
                symbols.append(f"{node.name}  (class)")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id != "__all__"
                    and is_public(target.id)
                ):
                    symbols.append(f"{target.id}  (module-level)")

    return symbols


# --- JS/TS: sem parser disponível, regex sobre os padrões comuns -------

_JS_EXPORT_PATTERNS = [
    (
        re.compile(
            r"^\s*export\s+default\s+(?:async\s+)?function\s*\*?\s*(\w+)",
            re.MULTILINE,
        ),
        "export default function",
    ),
    (
        re.compile(r"^\s*export\s+default\s+class\s+(\w+)", re.MULTILINE),
        "export default class",
    ),
    (
        re.compile(
            r"^\s*export\s+(?:async\s+)?function\s*\*?\s+(\w+)",
            re.MULTILINE,
        ),
        "export function",
    ),
    (
        re.compile(r"^\s*export\s+class\s+(\w+)", re.MULTILINE),
        "export class",
    ),
    (
        re.compile(
            r"^\s*export\s+(?:const|let|var)\s+(\w+)", re.MULTILINE
        ),
        "export const/let/var",
    ),
    (
        re.compile(r"^\s*exports\.(\w+)\s*=", re.MULTILINE),
        "exports.X =",
    ),
    (
        re.compile(r"^\s*module\.exports\.(\w+)\s*=", re.MULTILINE),
        "module.exports.X =",
    ),
]


def _list_js_symbols(content: str) -> list[str]:
    symbols = []
    seen = set()

    def add(name: str, kind: str) -> None:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            symbols.append(f"{name}  ({kind})")

    for pattern, kind in _JS_EXPORT_PATTERNS:
        for match in pattern.finditer(content):
            add(match.group(1), kind)

    # export { a, b as c }
    for match in re.finditer(r"export\s*{\s*([^}]+)\s*}", content):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            name = part.split(" as ")[-1].strip() if " as " in part else part
            add(name, "export { }")

    # export default algumaCoisa;  (referência a algo já definido)
    match = re.search(r"export\s+default\s+(\w+)\s*;", content)
    if match:
        add(match.group(1), "export default (referência)")

    # module.exports = { a, b, c: valor }
    match = re.search(r"module\.exports\s*=\s*{([^}]*)}", content, re.DOTALL)
    if match:
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            key = part.split(":")[0].strip()
            if key:
                add(key, "module.exports = { }")

    # module.exports = NomeUnico;  (uma função/classe/valor só)
    match = re.search(
        r"module\.exports\s*=\s*(\w+)\s*;?\s*$", content, re.MULTILINE
    )
    if match:
        add(match.group(1), "module.exports =")

    return symbols


# --- Go: regex sobre declarações top-level ------------------------------

_GO_PATTERNS = [
    (
        re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(", re.MULTILINE),
        "func",
    ),
    (
        re.compile(
            r"^type\s+(\w+)\s+(?:struct|interface)\b", re.MULTILINE
        ),
        "type",
    ),
    (re.compile(r"^var\s+(\w+)\b", re.MULTILINE), "var"),
    (re.compile(r"^const\s+(\w+)\b", re.MULTILINE), "const"),
]


def _list_go_symbols(content: str) -> list[str]:
    symbols = []

    for pattern, kind in _GO_PATTERNS:
        for match in pattern.finditer(content):
            name = match.group(1)
            visibility = (
                "exportado" if name[:1].isupper() else "não-exportado"
            )
            symbols.append(f"{name}  ({kind}, {visibility})")

    return symbols


# --- Fallback genérico: Java, Ruby, PHP, Rust, C/C++, etc. -------------

_GENERIC_PATTERNS = [
    (
        re.compile(
            r"^\s*(?:public\s+|private\s+|protected\s+)?"
            r"(?:static\s+|final\s+|abstract\s+)*class\s+(\w+)",
            re.MULTILINE,
        ),
        "class",
    ),
    (
        re.compile(
            r"^\s*(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)",
            re.MULTILINE,
        ),
        "interface",
    ),
    (re.compile(r"^\s*def\s+(\w+)", re.MULTILINE), "def"),
    (re.compile(r"^\s*module\s+(\w+)", re.MULTILINE), "module"),
    (re.compile(r"^\s*function\s+(\w+)", re.MULTILINE), "function"),
    (re.compile(r"^\s*pub\s+fn\s+(\w+)", re.MULTILINE), "pub fn"),
    (re.compile(r"^\s*pub\s+struct\s+(\w+)", re.MULTILINE), "pub struct"),
    (re.compile(r"^\s*pub\s+enum\s+(\w+)", re.MULTILINE), "pub enum"),
    (re.compile(r"^\s*pub\s+trait\s+(\w+)", re.MULTILINE), "pub trait"),
]


def _list_generic_symbols(content: str) -> list[str]:
    symbols = []
    seen = set()

    for pattern, kind in _GENERIC_PATTERNS:
        for match in pattern.finditer(content):
            name = match.group(1)
            key = (name, kind)
            if key not in seen:
                seen.add(key)
                symbols.append(f"{name}  ({kind})")

    return symbols


_EXTENSION_HANDLERS = {
    ".py": ("Python — AST, preciso", _list_python_symbols),
    ".js": ("JavaScript — heurística por regex", _list_js_symbols),
    ".jsx": ("JavaScript/JSX — heurística por regex", _list_js_symbols),
    ".ts": ("TypeScript — heurística por regex", _list_js_symbols),
    ".tsx": ("TypeScript/TSX — heurística por regex", _list_js_symbols),
    ".go": ("Go — heurística por regex", _list_go_symbols),
}


def list_symbols(project_name: str, file_path: str) -> str:
    """Lista os símbolos (funções, classes, exports) que um arquivo
    realmente define/exporta.

    Use ANTES de escrever um import/require em outro arquivo, para
    confirmar o nome exato em vez de adivinhar (ex.: descobrir se um
    arquivo exporta 'ProfessionalRepository' ou uma variação como
    'professional.repository') — evita erros de import que só
    apareceriam depois, no check_project ou em tempo de execução.
    """

    projects_dir = get_projects_dir()
    project_path = (projects_dir / project_name).resolve()

    if not project_path.is_relative_to(projects_dir):
        raise PermissionError(
            "Acesso fora do diretório de projetos não permitido."
        )

    if not project_path.exists():
        raise FileNotFoundError(
            f"Projeto não encontrado: {project_name}"
        )

    target = (project_path / file_path).resolve()

    if not target.is_relative_to(project_path):
        raise PermissionError(
            "Acesso fora do diretório do projeto não permitido."
        )

    if not target.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if not target.is_file():
        raise IsADirectoryError(f"Não é um arquivo: {file_path}")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Não foi possível ler '{file_path}' como texto "
            f"(arquivo binário?): {error}"
        )

    suffix = target.suffix.lower()
    label, handler = _EXTENSION_HANDLERS.get(
        suffix, ("heurística genérica por regex", _list_generic_symbols)
    )

    symbols = handler(content)

    if not symbols:
        return (
            f"Nenhum símbolo exportado/definido foi detectado em "
            f"'{file_path}' ({label}). O arquivo pode estar vazio, "
            "não exportar nada publicamente, ou usar um padrão que "
            "esta análise heurística não reconhece — nesse caso, use "
            "read_file para conferir manualmente."
        )

    lines = [f"Símbolos em '{file_path}' ({label}):"]
    lines += [f"- {symbol}" for symbol in symbols]

    return "\n".join(lines)


definition = {
    "type": "function",
    "function": {
        "name": "list_symbols",
        "description": (
            "Lista os símbolos (funções, classes, exports) que um "
            "arquivo REALMENTE define/exporta — via AST em Python "
            "(preciso) ou heurística por regex nas demais linguagens "
            "(Go, JS/TS, e um fallback genérico para Java/Ruby/PHP/"
            "Rust/C/C++). Use esta tool ANTES de escrever um import/"
            "require que referencia outro arquivo do projeto, para "
            "confirmar o nome exato do arquivo e dos símbolos "
            "exportados em vez de adivinhar — evita erros de import "
            "por nome errado (ex.: 'ProfessionalRepository' vs "
            "'professional.repository') que só apareceriam depois, "
            "no check_project ou em tempo de execução."
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
                    "description": (
                        "Caminho do arquivo, relativo à raiz do "
                        "projeto, cujos símbolos exportados devem "
                        "ser listados."
                    ),
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
    name="list_symbols",
    function=list_symbols,
    definition=definition,
    type=ToolType.ANALYSIS,
)