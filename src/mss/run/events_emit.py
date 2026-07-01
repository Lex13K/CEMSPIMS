"""Emit structured events on stdout for the app job runner (@@EVENT protocol)."""

from __future__ import annotations

import json
import sys
from typing import Any

_PREFIX = "@@EVENT "


def emit_event(kind: str, **fields: Any) -> None:
    payload = {"kind": kind, **fields}
    print(f"{_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)


def parse_event_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(_PREFIX):
        return None
    try:
        data = json.loads(stripped[len(_PREFIX) :])
        if isinstance(data, dict) and "kind" in data:
            return data
    except json.JSONDecodeError:
        return None
    return None


def event_to_app_event(data: dict[str, Any]):
    """Convert @@EVENT dict to AppEvent."""
    from mss.app import events as ev

    kind = data.get("kind")
    fields = {k: v for k, v in data.items() if k != "kind"}
    if kind == "step_started":
        return ev.step_started(
            run_id=fields["run_id"],
            pipeline=fields["pipeline"],
            step_id=fields["step_id"],
        )
    if kind == "step_finished":
        return ev.step_finished(
            run_id=fields["run_id"],
            pipeline=fields["pipeline"],
            step_id=fields["step_id"],
            status=fields.get("status", "completed"),
            error=fields.get("error"),
        )
    return None
