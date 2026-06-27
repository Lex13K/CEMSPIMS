from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from mss.io.paths import project_root, resolve_under_root

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


def sanitize_run_id(run_id: str) -> str:
    s = run_id.strip()
    if not s:
        raise ValueError("run id must be non-empty")
    if s in (".", ".."):
        raise ValueError(f"invalid run id: {s!r}")
    if "/" in s or "\\" in s:
        raise ValueError("run id must not contain path separators")
    if not _RUN_ID_PATTERN.fullmatch(s):
        raise ValueError(
            "run id may only contain letters, digits, '.', '_', and '-'"
        )
    return s


def expand_path_template(template: str, run_id: str) -> str:
    return template.replace("{run_id}", run_id)


@dataclass(frozen=True)
class ResolvedConfig:
    """Runtime paths after resolving config strings against project root."""

    run_id: str
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    project_root: Path
    source_config_path: Path


def load_resolved_config(config_path: Path, run_id: str) -> ResolvedConfig:
    rid = sanitize_run_id(run_id)
    root = project_root()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = data.get("paths") or {}
    raw_rel = paths.get("raw", "data/raw")
    interim_tmpl = paths.get("interim", "data/{run_id}/interim")
    processed_tmpl = paths.get("processed", "data/{run_id}/processed")
    interim_rel = expand_path_template(str(interim_tmpl), rid)
    processed_rel = expand_path_template(str(processed_tmpl), rid)
    return ResolvedConfig(
        run_id=rid,
        project_root=root,
        raw_dir=resolve_under_root(root, str(raw_rel)),
        interim_dir=resolve_under_root(root, interim_rel),
        processed_dir=resolve_under_root(root, processed_rel),
        source_config_path=config_path.resolve(),
    )


def default_config_path() -> Path:
    return project_root() / "configs" / "default.toml"
