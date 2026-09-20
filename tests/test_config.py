import pytest

from doceval import config


def test_require_api_key_returns_value(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "sk-test-123")
    assert config.require_api_key() == "sk-test-123"


def test_require_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "  sk-test-123  ")
    assert config.require_api_key() == "sk-test-123"


def test_require_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    with pytest.raises(config.MissingAPIKey) as exc:
        config.require_api_key()
    assert config.API_KEY_ENV in exc.value.message
    assert "console.typesafe.ai/keys" in exc.value.message


def test_require_api_key_raises_when_blank(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "   ")
    with pytest.raises(config.MissingAPIKey):
        config.require_api_key()


def test_resolve_model_defaults(monkeypatch):
    monkeypatch.delenv(config.MODEL_ENV, raising=False)
    assert config.resolve_model() == "jev-latest"


def test_resolve_model_honors_override(monkeypatch):
    monkeypatch.setenv(config.MODEL_ENV, "jev-1.13.0")
    assert config.resolve_model() == "jev-1.13.0"


def test_real_env_wins_over_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{config.API_KEY_ENV}=from-dotenv\n")
    monkeypatch.setenv(config.API_KEY_ENV, "from-environment")
    config.load_env(tmp_path)
    assert config.require_api_key() == "from-environment"


def test_dotenv_fills_unset_value(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{config.API_KEY_ENV}=from-dotenv\n")
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    config.load_env(tmp_path)
    assert config.require_api_key() == "from-dotenv"
