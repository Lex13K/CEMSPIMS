"""Parse CLI log lines for live headline display (best-effort)."""

from __future__ import annotations

import re
from typing import Any

_EPOCH_RE = re.compile(
    r"Epoch:\s+(\d+)%\|.*?(\d+)/(\d+).*?best=(\d+).*?pat=([\d]+)/([\d]+).*?train=([\d.]+).*?val=([\d.]+)"
)
_TQDM_RE = re.compile(r"^(?:Train|Evaluate[^:]*|ingest|universe|edges|node_features):\s*(\d+)%\|.*?(\d+)/(\d+)")
_GENERIC_PCT = re.compile(r"(\d+)%\|")
_EDGES_PROGRESS = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\]\s*edges")
_NODE_FEAT = re.compile(r"\[node-features\]\s*dates\s*(\d+)\s*/\s*(\d+)")
_STEP_RE = re.compile(r"---\s+([\w.]+):\s+step\s+([\w_]+)\s+---")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_headline(line: str) -> str | None:
    """Return a short human headline from one log line, or None."""
    clean = strip_ansi(line).strip()
    if not clean:
        return None
    m = _EPOCH_RE.search(clean)
    if m:
        return (
            f"Epoch {m.group(2)}/{m.group(3)} · train={m.group(7)} · val={m.group(8)}"
        )
    m = _TQDM_RE.search(clean)
    if m:
        return f"{m.group(1)}% ({m.group(2)}/{m.group(3)})"
    m = _EDGES_PROGRESS.search(clean)
    if m:
        return f"edges {m.group(1)}/{m.group(2)}"
    m = _NODE_FEAT.search(clean)
    if m:
        return f"node features {m.group(1)}/{m.group(2)}"
    m = _GENERIC_PCT.search(clean)
    if m and len(clean) < 120:
        return clean[:100]
    if clean.startswith("[") and "/" in clean and len(clean) < 100:
        return clean
    return None


def parse_step_banner(line: str) -> dict[str, str] | None:
    clean = strip_ansi(line)
    m = _STEP_RE.search(clean)
    if m:
        return {"pipeline": m.group(1), "step_id": m.group(2)}
    return None
