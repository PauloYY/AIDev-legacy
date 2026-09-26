import pytest

from app.llm.utils import estimate_tokens


def test_empty_string():
    assert estimate_tokens("") == 0


def test_none_returns_zero():
    assert estimate_tokens(None) == 0


def test_short_text():
    assert estimate_tokens("abc") == 1


def test_english_text_approximation():
    text = "Hello world this is a test"
    result = estimate_tokens(text)
    assert result >= 6
    assert result <= 8


def test_portuguese_text_approximation():
    text = "Este é um texto em português para teste"
    result = estimate_tokens(text)
    assert result >= 9
    assert result <= 12


def test_large_text():
    text = "x" * 4000
    result = estimate_tokens(text)
    assert result == 1000


def test_text_with_newlines():
    text = "line1\nline2\nline3"
    result = estimate_tokens(text)
    assert result >= 3
    assert result <= 5


def test_estimate_is_gross_approximation():
    """A estimativa é grosseira — o teste só garante que não quebra
    e produz valores na faixa esperada (1 token por ~4 chars)."""
    text = "A" * 100
    result = estimate_tokens(text)
    assert 20 <= result <= 30
