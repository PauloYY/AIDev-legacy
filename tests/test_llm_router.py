import pytest

from app.exceptions import LLMAPIError, LLMConnectionError, LLMRateLimitError
from app.llm.models import LLMResponse, Usage
from app.llm.providers.base import LLMProvider
from app.llm.router import LLMRouter


class FakeProvider(LLMProvider):
    def __init__(self, name, error=None, response=None):
        self.name = name
        self.error = error
        self.response = response
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        if self.error:
            raise self.error

        return self.response


def _ok_response(provider_name="p"):
    return LLMResponse(
        content="ok",
        tool_calls=[],
        usage=Usage(1, 1, 2),
        provider=provider_name,
    )


def test_falls_back_on_rate_limit():
    failing = FakeProvider("a", error=LLMRateLimitError("limite"))
    working = FakeProvider("b", response=_ok_response("b"))

    router = LLMRouter([failing, working])

    response = router.generate(messages=[])

    assert response.provider == "b"
    assert failing.calls == 1
    assert working.calls == 1


def test_falls_back_on_connection_error():
    failing = FakeProvider("a", error=LLMConnectionError("timeout"))
    working = FakeProvider("b", response=_ok_response("b"))

    router = LLMRouter([failing, working])

    response = router.generate(messages=[])

    assert response.provider == "b"


def test_falls_back_on_transient_5xx_error():
    failing = FakeProvider("a", error=LLMAPIError("indisponível", status_code=503))
    working = FakeProvider("b", response=_ok_response("b"))

    router = LLMRouter([failing, working])

    response = router.generate(messages=[])

    assert response.provider == "b"


def test_does_not_fall_back_on_non_transient_4xx_error():
    failing = FakeProvider("a", error=LLMAPIError("request inválida", status_code=400))
    working = FakeProvider("b", response=_ok_response("b"))

    router = LLMRouter([failing, working])

    with pytest.raises(LLMAPIError):
        router.generate(messages=[])

    assert working.calls == 0


def test_raises_when_all_providers_fail():
    a = FakeProvider("a", error=LLMRateLimitError("limite"))
    b = FakeProvider("b", error=LLMRateLimitError("limite"))

    router = LLMRouter([a, b], max_wait_rounds=0)  # sem espera, para o teste ser rápido

    with pytest.raises(LLMRateLimitError):
        router.generate(messages=[])


def test_raises_when_no_providers_configured():
    router = LLMRouter([])

    with pytest.raises(RuntimeError):
        router.generate(messages=[])


def test_waits_and_recovers_after_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.llm.router.time.sleep", lambda seconds: sleeps.append(seconds))

    calls = {"count": 0}

    class RecoveringProvider(LLMProvider):
        name = "a"

        def generate(self, messages, tools=None):
            calls["count"] += 1

            if calls["count"] == 1:
                raise LLMRateLimitError("limite")

            return _ok_response("a")

    router = LLMRouter([RecoveringProvider()], max_wait_rounds=1, base_wait_seconds=5)

    response = router.generate(messages=[])

    assert response.provider == "a"
    assert sleeps == [5]  # esperou uma vez antes de conseguir


def test_uses_retry_after_instead_of_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.llm.router.time.sleep", lambda seconds: sleeps.append(seconds))

    provider = FakeProvider("a", error=LLMRateLimitError("limite", retry_after=3))

    router = LLMRouter([provider], max_wait_rounds=1, base_wait_seconds=99)

    with pytest.raises(LLMRateLimitError):
        router.generate(messages=[])

    assert sleeps == [3]  # usou o retry_after (3s), não o backoff (99s)