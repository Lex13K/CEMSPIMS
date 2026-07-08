#!/usr/bin/env python3
"""Rebuild app state snapshot from disk journals/manifests.

After running the pipeline outside the app (CLI) or editing journals on disk,
run this so the graph reflects reality:

    python scripts/sync_app_state.py
    python scripts/sync_app_state.py --reload   # also POST rescan to running app
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mss.app.state_bootstrap import build_state_from_disk  # noqa: E402
from mss.io.paths import project_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync experiment app state from disk")
    parser.add_argument("--reload", action="store_true", help="POST /api/state/rescan to running app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    state = build_state_from_disk()
    snap_path = project_root() / "data" / ".app_state" / "state.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snap_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(snap_path)
    print(f"Wrote {snap_path} (revision base {state.revision}, {len(state.runs)} runs)")

    if args.reload:
        url = f"http://{args.host}:{args.port}/api/state/rescan"
        req = urllib.request.Request(url, method="POST", data=b"")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
            print(f"Reloaded running app: {body}")
        except urllib.error.URLError as e:
            print(f"Reload failed (is app running?): {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
