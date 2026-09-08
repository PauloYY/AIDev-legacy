"""Detecção determinística de arquivos potencialmente desnecessários.

Etapa de verificação final (dimensão "arquivos desnecessários"): coleta
EVIDÊNCIAS deterministicamente a partir do workspace real, sem nenhuma
chamada LLM obrigatória:

    workspace
       ↓ list_files
       ↓ referências/imports/configuração/entrypoints/testes
       ↓ candidatos
       ↓ evidências
       ↓ classificação (SAFE / UNCERTAIN / KEEP)

Classificação conservadora:
- KEEP: há evidência de necessidade (arquivo especial, teste,
  entrypoint, referenciado/importado por outro arquivo, listado em
  configuração/manifesto).
- SAFE: claramente desnecessário — sem nenhuma evidência de
  necessidade E com sinal positivo de inutilidade (temporário,
  backup/substituído, duplicata exata de um arquivo necessário).
  Ausência de referência, sozinha, NUNCA gera SAFE.
- UNCERTAIN: todo o resto (ambíguo). Nunca removido automaticamente.

Somente arquivos SAFE são candidatos à remoção automática (via
delete_file, que aplica suas próprias proteções). UNCERTAIN permanece.

Quando houver ambiguidade real, `format_evidence_for_llm()` formata as
evidências já coletadas para interpretação pela LLM — a LLM interpreta
evidências, nunca as inventa (não recebe só uma lista de arquivos).
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


SAFE = "safe"
UNCERTAIN = "uncertain"
KEEP = "keep"


# Arquivos com papel estrutural conhecido: nunca candidatos.
SPECIAL_BASENAMES = frozenset({
    "readme", "readme.md", "readme.txt", "license", "license.md",
    "license.txt", "licence", "notice", "authors", "changelog",
    "changelog.md", "contributing", "contributing.md",
    ".gitignore", ".gitattributes", ".dockerignore",
    "package.json", "package-lock.json",
    "go.mod", "go.sum",
    "cargo.toml", "cargo.lock",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "pom.xml", "build.gradle", "settings.gradle",
    "gemfile", "gemfile.lock",
    "composer.json", "composer.lock",
    "makefile", "dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "pytest.ini", "tox.ini",
    "tsconfig.json", "vite.config.js", "webpack.config.js",
    ".env.example",
})

# Manifestos/configurações cujo conteúdo pode citar outros arquivos.
CONFIG_BASENAMES = frozenset({
    "package.json", "go.mod", "cargo.toml", "pyproject.toml",
    "requirements.txt", "pom.xml", "build.gradle", "gemfile",
    "composer.json", "makefile", "dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "pytest.ini", "tox.ini", "tsconfig.json",
})

# Diretórios de estado interno: nunca candidatos.
SPECIAL_DIR_PARTS = frozenset({".aidev"})

# Arquivos de teste: evidência de necessidade por si só.
TEST_DIR_PARTS = frozenset({"test", "tests", "spec", "specs", "__tests__"})

# Basenames que sugerem ponto de entrada.
ENTRYPOINT_BASENAMES = frozenset({
    "main.py", "main.js", "main.ts", "main.go", "main.c", "main.cpp",
    "main.java", "main.rb", "main.php",
    "app.py", "app.js", "app.ts",
    "index.js", "index.ts",
    "__main__.py", "__init__.py",
})

# Sinais positivos de "temporário gerado durante desenvolvimento".
TEMP_SUFFIXES = frozenset({
    ".tmp", ".temp", ".bak", ".orig", ".swp", ".swo", ".log", ".pyc",
    ".rej",
})
TEMP_STEM_PREFIXES = ("tmp_", "temp_", "scratch_", "debug_")
TEMP_DIR_PARTS = frozenset({"tmp", "temp", "scratch"})

# Sinais positivos de "antigo claramente substituído".
SUPERSEDED_STEM_PREFIXES = (
    "old_", "backup_", "bak_", "deprecated_", "copy_of_",
)
SUPERSEDED_STEM_SUFFIXES = (
    "_old", "_backup", "_bak", "_orig", "_copy", "_deprecated",
)

# Sem esses sinais positivos, ausência de referência => UNCERTAIN.
MAX_SAFE_DELETIONS_PER_SCAN = 10


@dataclass
class FileCandidate:
    path: str
    verdict: str
    evidences: list[str] = field(default_factory=list)


@dataclass
class UnnecessaryFilesReport:
    files_scanned: int = 0
    candidates: list[FileCandidate] = field(default_factory=list)

    @property
    def safe(self) -> list[FileCandidate]:
        return [c for c in self.candidates if c.verdict == SAFE]

    @property
    def uncertain(self) -> list[FileCandidate]:
        return [c for c in self.candidates if c.verdict == UNCERTAIN]

    @property
    def safe_paths(self) -> list[str]:
        return [c.path for c in self.safe]


_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))",
    re.MULTILINE,
)
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]"""
    r"""|require\(\s*['"]([^'"]+)['"]\s*\))"""
)


def _basename(path: str) -> str:
    return PurePosixPath(path).name


def _stem(path: str) -> str:
    name = _basename(path)
    if "." in name:
        return name.rsplit(".", 1)[0]
    return name


def _suffix(path: str) -> str:
    name = _basename(path)
    if "." in name:
        return "." + name.rsplit(".", 1)[1].lower()
    return ""


def _parts(path: str) -> tuple:
    return PurePosixPath(path).parts


def _is_test_file(path: str) -> bool:
    name = _basename(path).lower()
    if any(p in TEST_DIR_PARTS for p in (part.lower() for part in _parts(path))):
        return True
    if name.startswith("test_") or name.startswith("spec_"):
        return True
    for marker in ("_test.", ".spec.", ".test."):
        if marker in name:
            return True
    return False


def _is_special_file(path: str) -> bool:
    name = _basename(path).lower()
    if name in SPECIAL_BASENAMES:
        return True
    return any(p in SPECIAL_DIR_PARTS for p in _parts(path))


def _is_entrypoint(path: str) -> bool:
    return _basename(path).lower() in ENTRYPOINT_BASENAMES


def _temp_signal(path: str) -> str | None:
    name = _basename(path)
    lowered = name.lower()
    if _suffix(path) in TEMP_SUFFIXES:
        return f"suffix {_suffix(path)}"
    if lowered.endswith("~"):
        return "suffix ~"
    stem = _stem(path).lower()
    for prefix in TEMP_STEM_PREFIXES:
        if stem.startswith(prefix):
            return f"prefix {prefix}"
    if any(p.lower() in TEMP_DIR_PARTS for p in _parts(path)[:-1]):
        return "temporary directory"
    return None


def _superseded_signal(path: str) -> str | None:
    stem = _stem(path).lower()
    for prefix in SUPERSEDED_STEM_PREFIXES:
        if stem.startswith(prefix):
            return f"prefix {prefix}"
    for suffix in SUPERSEDED_STEM_SUFFIXES:
        if stem.endswith(suffix):
            return f"suffix {suffix}"
    return None


def _module_to_path_candidates(module: str) -> list[str]:
    """Mapeia um módulo importado p/ caminhos relativos plausíveis."""
    base = module.replace(".", "/")
    return [f"{base}.py", f"{base}/__init__.py"]


def _imported_by_map(
    files: list[str], contents: dict[str, str | None]
) -> dict[str, list[str]]:
    """Para cada arquivo, lista quem o importa (evidência forte)."""
    imported_by: dict[str, list[str]] = {f: [] for f in files}
    file_set = set(files)
    for importer in files:
        content = contents.get(importer)
        if not content:
            continue
        lowered_importer = importer.lower()
        modules: set[str] = set()
        if lowered_importer.endswith(".py"):
            for match in _PY_IMPORT_RE.finditer(content):
                modules.add(match.group(1) or match.group(2) or "")
        elif lowered_importer.endswith((".js", ".jsx", ".ts", ".tsx")):
            for match in _JS_IMPORT_RE.finditer(content):
                modules.add(match.group(1) or match.group(2) or "")
        else:
            continue
        for module in modules:
            module = (module or "").strip()
            if not module:
                continue
            if module.startswith((".", "/")):
                # Import relativo: resolve contra o diretório do importer.
                base = PurePosixPath(importer).parent.joinpath(module)
                text = str(base)
                for cand in (
                    f"{text}.py", f"{text}/__init__.py",
                    f"{text}.js", f"{text}.ts",
                    f"{text}/index.js", f"{text}/index.ts",
                ):
                    norm = str(PurePosixPath(cand))
                    if norm in file_set and norm != importer:
                        if importer not in imported_by[norm]:
                            imported_by[norm].append(importer)
                continue
            for cand in _module_to_path_candidates(module):
                if cand in file_set and cand != importer:
                    if importer not in imported_by[cand]:
                        imported_by[cand].append(importer)
    return imported_by


def _substring_references(
    files: list[str], contents: dict[str, str | None]
) -> dict[str, list[tuple[str, int]]]:
    """Referência textual fraca (mesma semântica de find_references).

    Para cada arquivo, em quais OUTROS arquivos seu stem/basename/
    caminho aparece como substring, e quantas linhas. Conservador:
    qualquer menção conta como evidência de necessidade.
    """
    references: dict[str, list[tuple[str, int]]] = {f: [] for f in files}
    for target in files:
        tokens = {_stem(target), _basename(target), target}
        tokens = {t for t in tokens if t}
        if not tokens:
            continue
        for other in files:
            if other == target:
                continue
            content = contents.get(other)
            if not content:
                continue
            hits = 0
            for line in content.splitlines():
                if any(token in line for token in tokens):
                    hits += 1
            if hits:
                references[target].append((other, hits))
    return references


def analyze_unnecessary_files(
    project_name: str, tools_execute
) -> UnnecessaryFilesReport:
    """Analisa o workspace e classifica candidatos (determinístico).

    Nunca levanta: falha ao listar/ler resulta em relatório vazio
    (sem candidatos), nunca em bloqueio.
    """
    report = UnnecessaryFilesReport()
    try:
        files_result = tools_execute(
            "list_files", {"project_name": project_name}
        )
    except Exception as error:
        logger.warning(
            "Detecção de arquivos desnecessários: "
            "não foi possível listar arquivos: %s", error,
        )
        return report

    try:
        files = sorted(str(f) for f in (files_result or []))
    except Exception:
        return report

    if not files:
        return report

    report.files_scanned = len(files)

    contents: dict[str, str | None] = {}
    for path in files:
        try:
            result = tools_execute(
                "read_file",
                {"project_name": project_name, "file_path": path},
            )
            contents[path] = str(result) if result is not None else None
        except Exception:
            contents[path] = None

    # Duplicatas exatas (hash do conteúdo legível).
    hash_groups: dict[str, list[str]] = {}
    for path in files:
        content = contents.get(path)
        if content is None:
            continue
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hash_groups.setdefault(digest, []).append(path)

    imported_by = _imported_by_map(files, contents)
    substring_refs = _substring_references(files, contents)

    # Conteúdo agregado dos manifestos p/ checagem "listado em config".
    config_paths = [
        f for f in files if _basename(f).lower() in CONFIG_BASENAMES
    ]

    # Primeira passada: vereditos KEEP (evidência de necessidade).
    keep: set[str] = set()
    keep_evidence: dict[str, list[str]] = {f: [] for f in files}

    for path in files:
        if _is_special_file(path):
            keep.add(path)
            keep_evidence[path].append(
                f"special file ({_basename(path)})"
            )
        if _is_test_file(path):
            keep.add(path)
            keep_evidence[path].append("test file")
        if _is_entrypoint(path):
            keep.add(path)
            keep_evidence[path].append(
                f"possible entrypoint ({_basename(path)})"
            )
        for importer in imported_by.get(path, []):
            keep.add(path)
            keep_evidence[path].append(f"imported by {importer}")
        for other, hits in substring_refs.get(path, []):
            keep.add(path)
            keep_evidence[path].append(
                f"referenced in {other} ({hits} line(s))"
            )
        for config in config_paths:
            if config == path:
                continue
            content = contents.get(config)
            if not content:
                continue
            tokens = {_stem(path), _basename(path), path}
            if any(t and t in content for t in tokens):
                keep.add(path)
                keep_evidence[path].append(
                    f"mentioned in config ({config})"
                )

    # Segunda passada: candidatos (não-KEEP) + classificação.
    for path in files:
        if path in keep:
            continue
        content = contents.get(path)
        evidences = [
            "no references found in other files",
            "not an entrypoint",
            "not mentioned in config",
            "not used by tests",
            "not a special file",
        ]
        if content is None:
            report.candidates.append(FileCandidate(
                path=path,
                verdict=UNCERTAIN,
                evidences=[
                    "unreadable content (binary?) — no analysis",
                    *evidences,
                ],
            ))
            continue

        temp = _temp_signal(path)
        if temp:
            evidences.append(
                f"name suggests a temporary file ({temp})"
            )
        superseded = _superseded_signal(path)
        if superseded:
            evidences.append(
                f"name suggests a superseded file ({superseded})"
            )

        duplicate_of_kept: str | None = None
        try:
            digest = hashlib.sha256(
                content.encode("utf-8")).hexdigest()
            group = [p for p in hash_groups.get(digest, [])
                     if p != path]
            if group:
                kept_dupes = [p for p in group if p in keep]
                if kept_dupes:
                    duplicate_of_kept = sorted(kept_dupes)[0]
                    evidences.append(
                        "exact duplicate of "
                        f"{duplicate_of_kept} (which is needed)"
                    )
                else:
                    evidences.append(
                        "exact duplicate of "
                        f"{sorted(group)[0]} (also unreferenced)"
                    )
        except Exception:
            pass

        if temp or superseded or duplicate_of_kept:
            verdict = SAFE
        else:
            verdict = UNCERTAIN
        report.candidates.append(FileCandidate(
            path=path, verdict=verdict, evidences=evidences))

    return report


def format_evidence_for_llm(report: UnnecessaryFilesReport) -> str:
    """Formata evidências coletadas p/ interpretação pela LLM.

    A LLM interpreta evidências, não as inventa: recebe candidatos +
    evidências determinísticas (nunca só uma lista de arquivos).
    Uso opcional, só quando houver ambiguidade real.
    """
    lines = [
        "UNNECESSARY FILE CANDIDATES "
        f"({report.files_scanned} file(s) analyzed):",
    ]
    if not report.candidates:
        lines.append("(no candidates)")
        return "\n".join(lines)
    for candidate in sorted(
        report.candidates, key=lambda c: (c.verdict, c.path)
    ):
        lines.append(f"\n{candidate.path} → {candidate.verdict.upper()}")
        for evidence in candidate.evidences:
            lines.append(f"- {evidence}")
    return "\n".join(lines)
