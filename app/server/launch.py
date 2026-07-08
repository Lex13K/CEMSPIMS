"""Launch the experiment graph app."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import uvicorn

from app.server.main import app, mount_static


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CEMSPIMS experiment graph app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser")
    parser.add_argument(
        "--web-dist",
        type=str,
        default=None,
        help="Path to built frontend (default: app/web/dist)",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    web_dist = Path(args.web_dist) if args.web_dist else root / "app" / "web" / "dist"
    if web_dist.is_dir():
        mount_static(web_dist)
    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
