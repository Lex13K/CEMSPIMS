"""SSE broadcast helpers for the experiment app."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from mss.app.projection import project_graph
from mss.app.state_store import StateStore


async def stream_events(store: StateStore) -> AsyncIterator[str]:
    """Yield SSE-formatted messages: snapshot on connect, then deltas."""
    q = store.subscribe()
    try:
        snap = project_graph(store.get_state())
        yield _sse("snapshot", snap)
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield _sse("ping", {"t": "keepalive"})
                continue
            if msg.get("type") == "snapshot":
                snap = project_graph(store.get_state())
                yield _sse("snapshot", snap)
            else:
                yield _sse(msg.get("type", "delta"), msg)
    finally:
        store.unsubscribe(q)


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
