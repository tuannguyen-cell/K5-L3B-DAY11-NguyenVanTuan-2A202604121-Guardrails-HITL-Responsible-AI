"""Validate synthetic CP4 attack design and honest evidence collection."""
import asyncio
import json

from attacks import attacks
from core.config import load_protected_payload


def test_prompts_are_distinct_and_do_not_supply_the_answer():
    prompts = attacks.adversarial_prompts
    assert len(prompts) >= 5
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert len({p["category"] for p in prompts}) == len(prompts)
    for prompt in prompts:
        assert "TODO" not in prompt["input"] and len(prompt["input"]) > 200
        for secret in load_protected_payload()["secrets"].values():
            assert secret not in prompt["input"]  # prevent a simple echo being counted as extraction


def test_runner_saves_live_outcomes_without_turning_api_errors_into_blocks(tmp_path, monkeypatch):
    replies = iter(["Demo admin value: admin123", RuntimeError("private-key-value")])
    monkeypatch.setenv("GOOGLE_API_KEY", "private-key-value")

    async def chat(*args):
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result, None

    monkeypatch.setattr(attacks, "chat_with_agent", chat)
    path = tmp_path / "red.json"
    results = asyncio.run(attacks.run_attacks(
        None, None, prompts=attacks.adversarial_prompts[:2], target_name="red_default",
        output_path=path, delay_seconds=0,
    ))
    assert results[0]["leaked"] and results[0]["layer"] == "leaked"
    assert not results[1]["leaked"] and not results[1]["blocked"]
    assert results[1]["layer"] == "error" and results[1]["error"]
    assert "private-key-value" not in path.read_text(encoding="utf-8")
    aggregate = tmp_path / "aggregate.json"
    attacks.save_attack_results(unsafe_results=results, guards_results=[], filepath=aggregate)
    saved = json.loads(aggregate.read_text(encoding="utf-8"))
    assert saved["summary"]["unsafe_leaked"] == 1
    assert saved["summary"]["unsafe_errors"] == 1
    assert saved["unsafe_attacks"][1]["error"]


def test_nemo_failure_is_distinct_from_successful_defense():
    outcome = attacks.classify_attack_outcome(
        "banking handover", "NeMo Guardrails input check failed", target_name="red_advance")
    assert outcome["layer"] == "nemo_error"
    assert not outcome["leaked"]


def test_gemini_compatible_runtime_preserves_red_and_all_advance_rails(monkeypatch):
    from agents.agent import create_red_agent_default
    from agents.guards_agent import create_red_agent_advance, RedAdvanceGuardPlugin
    from core.config import GEMINI_OPENAI_BASE_URL
    from core.openai_runtime import OpenAIRunner

    monkeypatch.setenv("RED_TEAM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_RUNTIME", "compatible")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    for factory in (create_red_agent_default, create_red_agent_advance):
        agent, runner = factory()
        assert isinstance(runner, OpenAIRunner)
        assert agent.provider == runner.provider == "gemini"
        assert runner.client_kwargs["base_url"] == GEMINI_OPENAI_BASE_URL
        assert runner.client_kwargs["api_key"] == "test-google-key"
        if factory is create_red_agent_advance:
            assert len(runner.plugins) == 1
            plugin = runner.plugins[0]
            assert isinstance(plugin, RedAdvanceGuardPlugin)
            assert plugin.runtime == "openai" and plugin.nemo is not None
