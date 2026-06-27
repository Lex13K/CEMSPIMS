from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root: .../CEMSPIMS/src/mss/io/paths.py -> parents[3]."""
    return Path(__file__).resolve().parents[3]


def resolve_under_root(root: Path, relative: str) -> Path:
    p = Path(relative)
    return (root / p).resolve() if not p.is_absolute() else p.resolve()
