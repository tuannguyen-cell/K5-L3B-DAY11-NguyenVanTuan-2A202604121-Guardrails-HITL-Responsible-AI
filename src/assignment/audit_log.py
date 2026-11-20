"""
Assignment 11 — Redacted audit records and request latency.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Begin a request; redact sensitive values before retaining any text."""
        key = request_id or user_id
        if key in self._open:
            raise ValueError("Request already open; supply a distinct request_id")
        self._open[key] = {
            "request_id": key, "user_id": user_id,
            "input": content_filter(text)["redacted"],
            "started_at": utc_now_iso(), "start": time.monotonic(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an existing request and retain a redacted audit record."""
        key = request_id or user_id
        if key not in self._open or self._open[key]["user_id"] != user_id:
            raise ValueError("No matching audit input for this request")
        entry = self._open.pop(key)
        start = entry.pop("start")
        entry.update(
            output=content_filter(text)["redacted"], blocked=blocked, layer=layer,
            finished_at=utc_now_iso(), latency_ms=max(0, (time.monotonic() - start) * 1000),
        )
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
