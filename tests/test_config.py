from app.config import Config


def test_available_providers_when_fully_configured(monkeypatch):
    monkeypatch.setattr(Config, "groq_api_key", "key")
    monkeypatch.setattr(Config, "groq_model", "model")
    monkeypatch.setattr(Config, "openrouter_api_key", None)
    monkeypatch.setattr(Config, "openrouter_model", None)
    monkeypatch.setattr(Config, "cerebras_api_key", None)
    monkeypatch.setattr(Config, "cerebras_model", None)

    assert Config.available_providers() == ["groq"]


def test_provider_missing_model_is_not_available(monkeypatch):
    monkeypatch.setattr(Config, "groq_api_key", "key")
    monkeypatch.setattr(Config, "groq_model", None)
    monkeypatch.setattr(Config, "openrouter_api_key", None)
    monkeypatch.setattr(Config, "openrouter_model", None)
    monkeypatch.setattr(Config, "cerebras_api_key", None)
    monkeypatch.setattr(Config, "cerebras_model", None)

    assert Config.available_providers() == []


def test_missing_providers_reports_missing_fields(monkeypatch):
    monkeypatch.setattr(Config, "groq_api_key", None)
    monkeypatch.setattr(Config, "groq_model", "model")
    monkeypatch.setattr(Config, "openrouter_api_key", None)
    monkeypatch.setattr(Config, "openrouter_model", None)
    monkeypatch.setattr(Config, "cerebras_api_key", None)
    monkeypatch.setattr(Config, "cerebras_model", None)

    missing = Config.missing_providers()

    assert "GROQ_API_KEY" in missing["groq"]
    assert "OPENROUTER_API_KEY" in missing["openrouter"]
    assert "OPENROUTER_MODEL" in missing["openrouter"]
