import httpx
import pytest

from app.exceptions import LLMAPIError, LLMRateLimitError
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    @property
    def is_error(self):
        return self.status_code >= 400

    def json(self):
        return self._payload


@pytest.fixture
def provider():
    return OpenAICompatibleProvider(
        base_url="https://example.com/v1/chat/completions",
        api_key="key",
        model="some-model",
        name="fake",
    )


def test_raises_generic_api_error_on_plain_500(provider, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: FakeResponse(
            500,
            {"error": {"code": "", "message": "internal error"}},
        ),
    )

    with pytest.raises(LLMAPIError) as exc_info:
        provider.generate(messages=[])

    assert not isinstance(exc_info.value, LLMRateLimitError)
    assert exc_info.value.status_code == 500


def test_reclassifies_rate_limit_check_failed_as_rate_limit(provider, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: FakeResponse(
            500,
            {
                "error": {
                    "code": "",
                    "message": "rate_limit_check_failed (request id: abc123)",
                }
            },
        ),
    )

    with pytest.raises(LLMRateLimitError):
        provider.generate(messages=[])


def test_real_429_still_raises_rate_limit_with_retry_after(provider, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: FakeResponse(
            429,
            {"error": {"code": 429, "message": "too many requests"}},
            headers={"Retry-After": "12"},
        ),
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        provider.generate(messages=[])

    assert exc_info.value.retry_after == 12.0