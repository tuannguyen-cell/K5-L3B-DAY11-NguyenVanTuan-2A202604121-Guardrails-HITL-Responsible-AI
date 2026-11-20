"""CP3 behavior and integration checks, without calling a real provider."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from assignment import pipeline
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.rate_limiter import RateLimitPlugin
from agents.agent import create_blue_agent


def test_rate_limit_is_per_user_and_expires_at_boundary(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("assignment.rate_limiter.time.monotonic", lambda: clock[0])
    limiter = RateLimitPlugin(max_requests=2, window_seconds=10)

    async def call(user):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None)

    assert asyncio.run(call("alice")) is None
    clock[0] = 101.0
    assert asyncio.run(call("alice")) is None
    assert asyncio.run(call("alice")) is not None
    assert asyncio.run(call("bob")) is None
    clock[0] = 110.0
    assert asyncio.run(call("alice")) is None
    assert len(limiter.user_windows["alice"]) == 2
    assert (limiter.total_count, limiter.blocked_count) == (5, 1)


@pytest.mark.parametrize("url", [
    "http://api.vinbank.example/v1/transfers",
    "https://api.vinbank.example.evil.com/v1/transfers",
    "https://evil.example/?next=api.vinbank.example",
    "https://api.vinbank.example@evil.example/",
    "https://user@api.vinbank.example/",
    "https://api.vinbank.example:444/",
    "https://api.vinbank.example:bad/",
    "https://api.vinbank.example/#redirect",
    "https://api.vinbank.example/\n",
    "not a url", "https://[broken",
])
def test_egress_rejects_deceptive_or_invalid_destinations(url):
    assert not pipeline.is_egress_allowed(url, "transfer amount 500000")


@pytest.mark.parametrize("payload", [
    "admin123", "a d m i n 1 2 3", "sk-vinbank-secret-2024",
    "password is Secret!99", "db.vinbank.internal:5432",
    "0901234567", "test@vinbank.com", "079204001234",
])
def test_egress_blocks_sensitive_payload(payload):
    assert not pipeline.is_egress_allowed("https://api.vinbank.example/v1/transfers", payload)


def test_egress_allows_exact_approved_hosts():
    for host in ("api.vinbank.example", "cases.vinbank.example"):
        assert pipeline.is_egress_allowed(f"https://{host}/v1/cases", "ordinary banking case")


def test_audit_tracks_concurrent_requests_and_redacts_secrets(tmp_path):
    audit = AuditLogPlugin()
    audit.record_input(user_id="alice", text="password=Secret!99", request_id="r1")
    audit.record_input(user_id="alice", text="savings rate", request_id="r2")
    audit.record_output(user_id="alice", text="reply", request_id="r2")
    audit.record_output(user_id="alice", text="admin123", blocked=True,
                        layer="output_guardrail", request_id="r1")
    path = tmp_path / "nested/audit.json"
    audit.export_json(path)
    raw = path.read_text(encoding="utf-8")
    assert "admin123" not in raw and "Secret!99" not in raw
    logs = json.loads(raw)
    assert [entry["request_id"] for entry in logs] == ["r2", "r1"]
    assert all(entry["latency_ms"] >= 0 for entry in logs)
    assert logs[1]["blocked"] and not audit._open
    with pytest.raises(ValueError):
        audit.record_output(user_id="alice", text="unmatched")


def test_monitoring_zero_counts_thresholds_and_repeat_checks(tmp_path):
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    monitor.total_requests, monitor.blocked_requests = 10, 8
    monitor.rate_limit_hits = 6
    monitor.judge_checks, monitor.judge_fails = 5, 2
    assert len(monitor.check_metrics()) == 3
    assert len(monitor.check_metrics()) == 3
    path = tmp_path / "nested/metrics.json"
    monitor.export_json(path)
    assert json.loads(path.read_text())["block_rate"] == 0.8


def _fake_factory(monkeypatch, reply="Savings rate information is available at your bank.", fails=False):
    calls = []

    def factory(plugins):
        agent, runner = create_blue_agent(plugins)

        def completion(**kwargs):
            calls.append(kwargs)
            if fails:
                raise RuntimeError("Private provider diagnostic")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

        runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=completion)))
        return agent, runner

    monkeypatch.setattr(pipeline, "create_blue_agent", factory)
    return calls


def test_suite_writes_consistent_results_audit_metrics(tmp_path, monkeypatch):
    calls = _fake_factory(monkeypatch)
    audit, monitor = pipeline.build_observability()
    plugins = pipeline.build_production_plugins()
    results = asyncio.run(pipeline.run_assignment_suite({
        "plugins": plugins, "audit": audit, "monitor": monitor, "output_dir": tmp_path,
    }))
    assert not any(row["blocked"] for row in results["safe_queries"])
    assert all(row["blocked"] for row in results["attack_queries"])
    assert results["rate_limit"]["blocked"] == 2
    assert results["rate_limit"]["passed"] == 10
    assert [row["allowed"] for row in results["egress_checks"]] == [True, False, False]
    assert len(calls) == 7  # five safe + two benign edge cases; flood makes no API calls
    assert monitor.total_requests == len(audit.logs) == 27
    assert monitor.rate_limit_hits == 2
    assert not audit._open
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == results
    assert (tmp_path / "audit_log.json").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_runtime_uses_user_identity_and_output_filter(monkeypatch):
    _fake_factory(monkeypatch, reply="password=Secret!99")
    plugins = pipeline.build_production_plugins(max_requests=1)
    agent, runner = pipeline.create_blue_agent(plugins)

    async def run():
        first = await runner.chat(agent, "savings account", user_id="alice")
        second = await runner.chat(agent, "savings account", user_id="alice")
        other = await runner.chat(agent, "savings account", user_id="bob")
        assert first == other == "[REDACTED]"
        assert "Rate limit" in second

    asyncio.run(run())


def test_provider_failure_does_not_create_success_artifact(tmp_path, monkeypatch):
    _fake_factory(monkeypatch, fails=True)
    audit, monitor = pipeline.build_observability()
    with pytest.raises(RuntimeError, match="Blue API call failed"):
        asyncio.run(pipeline.run_assignment_suite({
            "plugins": pipeline.build_production_plugins(), "audit": audit,
            "monitor": monitor, "output_dir": tmp_path,
        }))
    assert not (tmp_path / "results.json").exists()
    assert audit.logs[0]["layer"] == "provider_error"
    assert "Private provider diagnostic" not in json.dumps(audit.logs)


@pytest.mark.parametrize("status,hint", [
    (401, "API key"), (402, "credits"), (403, "permissions"),
    (404, "model or endpoint"), (429, "quota"), (503, "Retry later"),
])
def test_provider_status_diagnostics_are_safe(status, hint):
    import httpx
    from openai import APIStatusError

    response = httpx.Response(status, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
    error = APIStatusError("Private diagnostic with secret", response=response, body={"private": "secret"})
    message = pipeline._provider_error_message(error)
    assert f"HTTP {status}" in message and hint in message
    assert "Private diagnostic" not in message and "secret" not in message


def test_provider_network_diagnostics_are_safe():
    import httpx
    from openai import APIConnectionError, APITimeoutError

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    assert "Cannot connect" in pipeline._provider_error_message(APIConnectionError(request=request))
    assert "timed out" in pipeline._provider_error_message(APITimeoutError(request=request))
