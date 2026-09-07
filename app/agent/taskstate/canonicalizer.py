"""Fase 4 — canonicalização do prompt (PT → EN canônico).

Transformação LINGUÍSTICA, não planejamento: o canonicalizer nunca
inventa arquitetura, decide implementação, remove ambiguidade ou cria
requisito — isso é papel do TaskInterpreter (e, em runtime, do Planner).

Custo LLM (deliberado, conservador):
- EN (ou PT sem acentos): ZERO chamadas — canônico = original
  normalizado (só whitespace).
- PT com acentos (sinal forte, determinístico): UMA chamada na entrada
  da run, com fallback seguro para o original em qualquer erro.
- Nunca uma chamada por iteração.

Por que só com acentos? Detectar idioma sem LLM é heurístico; traduzir
texto EN "só por garantia" seria a chamada redundante que a Fase 4
manda evitar. PT real quase sempre tem acentos; PT ASCII segue com o
original como canônico + language="pt" (só metadado, sem tradução).
"""

import re
from dataclasses import dataclass

# Sinal forte: caracteres que praticamente só aparecem em PT.
_PT_ACCENTS = frozenset("ãõçáàâéêíóôúüÃÕÇÁÀÂÉÊÍÓÔÚÜ")

# Sinal fraco: stopwords PT com word-boundary (só metadado de idioma,
# nunca disparam tradução sozinhas).
_PT_WORDS = frozenset({
    "crie", "criar", "faca", "faça", "adicione", "implemente",
    "teste", "testes", "arquivo", "arquivos", "projeto", "objetivo",
    "gerenciador", "sistema", "lista", "notas", "tarefas", "para",
    "uma", "este", "esta", "mais", "como", "nao", "não", "voce",
    "você", "esta", "está", "sao", "são", "deve", "devem", "apenas",
    "somente", "antes", "depois", "entre", "sobre", "meu", "minha",
    "seu", "sua", "isso", "isto", "qual", "quando", "onde",
})

# Integração Fase 4 (PT-ASCII): verbos no imperativo/infinitivo PT.
# Formas escolhidas para NÃO colidir com inglês comum ("test" ≠
# "teste", "list" ≠ "liste", "remove" propositalmente ausente: "remove
# the file" é inglês válido).
_PT_VERBS = frozenset({
    "crie", "criar", "faca", "faça", "fazer", "adicione", "adicionar",
    "remova", "remover", "implemente", "implementar", "corrija",
    "corrigir", "teste", "testar", "liste", "listar", "construa",
    "construir", "atualize", "atualizar", "verifique", "verificar",
    "execute", "executar", "rode", "rodar", "analise", "analisar",
    "descreva", "descrever",
})

# Palavras funcionais PT (len ≥ 2) para combinar com verbo. "do"/"no"/
# "as"/"se" excluídos de propósito: colidem com inglês ("do the
# thing", "no exceptions", "as a file").
_PT_FUNC = frozenset({
    "que", "para", "com", "uma", "um", "de", "da", "das", "dos",
    "em", "na", "nos", "nas", "os", "ao", "aos", "pelo", "pela",
    "pelos", "pelas", "este", "esta", "estes", "estas", "esse",
    "essa", "esses", "essas", "isso", "isto", "aquele", "aquela",
    "meu", "minha", "seu", "sua", "nosso", "mais", "como", "mas",
    "entre", "sobre", "muito", "tambem", "também", "ja", "já",
})

# Substantivos PT frequentes em prompts (apoio, len ≥ 2).
_PT_NOUNS = frozenset({
    "arquivo", "arquivos", "projeto", "sistema", "teste", "testes",
    "tarefa", "tarefas", "nota", "notas", "objetivo", "gerenciador",
    "pagina", "página", "login", "cadastro", "cliente", "clientes",
    "api", "tela", "botao", "botão",
})


def _pt_words_lower(text: str) -> set[str]:
    try:
        return set(_WORD_RE.findall(text.lower()))
    except Exception:
        return set()


def pt_ascii_evidence(text: str) -> dict[str, int]:
    """Evidência PT-ASCII: {verbs, funcs, nouns} (distintos, len ≥ 2).

    Conservador por construção: exige combinação (verbo + outro
    token) ou repetição (≥2 verbos / ≥3 funcionais). Um token isolado
    ("meu", "projeto") nunca dispara tradução sozinho.
    """
    words = {w for w in _pt_words_lower(text) if len(w) >= 2}
    return {
        "verbs": len(words & _PT_VERBS),
        "funcs": len(words & _PT_FUNC),
        "nouns": len(words & _PT_NOUNS),
    }


def pt_ascii_should_translate(text: str) -> bool:
    """True com forte evidência de PT mesmo sem acentos."""
    try:
        ev = pt_ascii_evidence(text)
    except Exception:
        return False
    if ev["verbs"] >= 2:
        return True
    if ev["verbs"] >= 1 and (ev["funcs"] + ev["nouns"]) >= 1:
        return True
    return ev["funcs"] >= 3

_EN_WORDS = frozenset({
    "the", "a", "an", "and", "with", "for", "create", "build",
    "implement", "add", "list", "test", "tests", "file", "project",
    "create", "simple", "system", "manage",
})

_WORD_RE = re.compile(r"[a-zà-úâêôãõç]+", re.IGNORECASE)


def normalize_prompt(text: str) -> str:
    """Normalização determinística (preserva linhas e conteúdo)."""
    try:
        if text is None:
            return ""
        raw = text if isinstance(text, str) else str(text)
    except Exception:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).strip()
             for line in raw.splitlines()]
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return collapsed.strip()


def detect_language(text: str) -> str:
    """Idioma provável: "pt" | "en" | "unknown" (só metadado)."""
    try:
        lowered = text.lower()
    except Exception:
        return "unknown"
    if any(char in lowered for char in _PT_ACCENTS):
        return "pt"
    if pt_ascii_should_translate(text):
        return "pt"
    words = set(_WORD_RE.findall(lowered))
    pt_hits = len(words & _PT_WORDS)
    if pt_hits >= 2:
        return "pt"
    # Sem sinal PT: ASCII é tratado como inglês (idioma interno do
    # agente); não-ASCII sem sinal PT fica "unknown" (conservador).
    if all(ord(char) < 128 for char in text):
        return "en"
    return "unknown"


def pt_ascii_enabled() -> bool:
    """Flag AIDEV_PT_ASCII_TRANSLATION (default ligado)."""
    try:
        from app.config import Config

        return bool(getattr(Config, "pt_ascii_translation", True))
    except Exception:
        return True


def needs_translation(text: str, allow_pt_ascii: bool | None = None) -> bool:
    """Se a chamada LLM de tradução se justifica (integração Fase 4).

    Sempre: acentos PT. PT-ASCII (verbos + funcionais, conservador)
    somente com AIDEV_PT_ASCII_TRANSLATION=1 (`allow_pt_ascii`
    explícito sobrepõe a flag — útil em testes). Nunca levanta.
    """
    try:
        if any(char in text for char in _PT_ACCENTS):
            return True
        if allow_pt_ascii is None:
            allow_pt_ascii = pt_ascii_enabled()
        return bool(allow_pt_ascii) and pt_ascii_should_translate(text)
    except Exception:
        return False


@dataclass
class CanonicalizationResult:
    original_prompt: str
    canonical_prompt: str
    language: str = "unknown"
    translation_applied: bool = False
    error: str | None = None
    llm_calls: int = 0


class PromptCanonicalizer:
    """Canonicaliza o prompt de entrada (uma vez por run)."""

    COMPONENT = "PromptCanonicalizer"

    # Guarda contra saída degenerada (também protege contra doubles
    # de teste que retornam conteúdo fixo): tradução legítima PT→EN
    # tem tamanho próximo do original.
    MIN_WORD_RATIO = 0.5
    MAX_WORD_RATIO = 2.0

    def __init__(self, llm=None):
        self.llm = llm

    def canonicalize(self, prompt) -> CanonicalizationResult:
        try:
            if prompt is None:
                original = ""
            else:
                original = (prompt if isinstance(prompt, str)
                            else str(prompt))
        except Exception:
            original = ""
        normalized = normalize_prompt(original)
        if not normalized:
            return CanonicalizationResult(
                original_prompt=original,
                canonical_prompt="",
                language="unknown",
                error="empty_prompt",
            )
        language = detect_language(normalized)
        if self.llm is None or not needs_translation(normalized):
            return CanonicalizationResult(
                original_prompt=original,
                canonical_prompt=normalized,
                language=language,
            )
        return self._translate(original, normalized, language)

    def _translate(
        self, original: str, normalized: str, language: str,
    ) -> CanonicalizationResult:
        fallback = CanonicalizationResult(
            original_prompt=original,
            canonical_prompt=normalized,
            language=language,
        )
        try:
            content = self._generate(normalized)
        except Exception as error:
            fallback.error = f"translation_error: {type(error).__name__}"
            return fallback
        fallback.llm_calls = 1
        candidate = content.strip() if isinstance(content, str) else ""
        if self._looks_degenerate(normalized, candidate):
            fallback.error = "translation_rejected_degenerate"
            return fallback
        fallback.canonical_prompt = candidate
        fallback.translation_applied = True
        return fallback

    def _generate(self, text: str):
        from app.llm.models import Message

        prompt = (
            "Translate the following task prompt to English. "
            "LINGUISTIC translation only — do NOT plan, design, decide, "
            "disambiguate, add requirements or remove information.\n"
            "Preserve verbatim: file names, paths, commands, library/API "
            "names, identifiers, numbers, versions, URLs and code.\n"
            "Return ONLY the translated prompt, no explanations.\n"
            "\nPROMPT:\n" + text
        )
        try:
            response = self.llm.generate(
                messages=[Message(role="user", content=prompt)],
                component=self.COMPONENT,
                iteration=0,
            )
        except TypeError:
            response = self.llm.generate(
                messages=[Message(role="user", content=prompt)],
            )
        return getattr(response, "content", None)

    @classmethod
    def _looks_degenerate(cls, original: str, candidate: str) -> bool:
        if not candidate or not candidate.strip():
            return True
        orig_words = len(original.split())
        cand_words = len(candidate.split())
        if orig_words == 0 or cand_words == 0:
            return True
        ratio = cand_words / orig_words
        if not (cls.MIN_WORD_RATIO <= ratio <= cls.MAX_WORD_RATIO):
            return True
        orig_chars, cand_chars = len(original), len(candidate)
        if orig_chars == 0 or cand_chars == 0:
            return True
        char_ratio = cand_chars / orig_chars
        return not (0.4 <= char_ratio <= 2.5)
