"""Blue provider selection must preserve plugins and use the matching API key."""
import pytest

from core import config
from core.openai_runtime import create_blue_pair


@pytest.mark.parametrize("provider,model,key,endpoint", [
    ("gemini", "gemini-test", "google-test-key", config.GEMINI_OPENAI_BASE_URL),
    ("openrouter", "liquid-test", "router-test-key", config.OPENROUTER_BASE_URL),
])
def test_selected_provider_wires_model_key_and_plugins(monkeypatch, provider, model, key, endpoint):
    monkeypatch.setenv("BLUE_PROVIDER", provider)
    monkeypatch.setenv("BLUE_MODEL", model)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    plugin = object()
    agent, runner = create_blue_pair(
        name="test", instruction="Banking", app_name="test", plugins=[plugin])
    assert agent.provider == runner.provider == provider
    assert runner.model == model
    assert runner.client_kwargs == {"api_key": key, "base_url": endpoint}
    assert runner.plugins == [plugin]


def test_gemini_does_not_require_openrouter_key(monkeypatch):
    monkeypatch.setenv("BLUE_PROVIDER", "gemini")
    monkeypatch.delenv("BLUE_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert config.get_blue_model() == "gemini-test"
    assert config.blue_client_kwargs()["api_key"] == "google-test-key"


def test_unsupported_blue_provider_fails_explicitly(monkeypatch):
    monkeypatch.setenv("BLUE_PROVIDER", "typo")
    with pytest.raises(ValueError, match="BLUE_PROVIDER"):
        config.blue_client_kwargs()
