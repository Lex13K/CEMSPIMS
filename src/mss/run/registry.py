"""Discover runs and render experiment trees."""

from __future__ import annotations

import difflib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.run.manifest import RunManifest, load_run_manifest

_RESERVED_DATA_DIRS = frozenset({"shared", ".app_jobs"})


def _data_root() -> Path:
    return project_root() / "data"


def run_data_dir(run_id: str) -> Path:
    return _data_root() / sanitize_run_id(run_id)


def is_materialized_run(run_id: str, configs_dir: Path | None = None) -> bool:
    """True when data/<run_id>/ exists and its config file is reachable."""
    rid = sanitize_run_id(run_id)
    if rid in _RESERVED_DATA_DIRS:
        return False
    data_dir = run_data_dir(rid)
    if not data_dir.is_dir():
        return False
    cdir = _configs_dir(configs_dir)
    if (cdir / f"{rid}.toml").is_file():
        return True
    man_path = data_dir / "run_manifest.json"
    if not man_path.is_file():
        return False
    try:
        manifest = load_run_manifest(man_path)
        return (project_root() / manifest.config_path).is_file()
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_path: Path
    manifest: RunManifest | None
    parent_run_id: str | None
    branch_pipeline: str | None
    branch_step_id: str | None
    config_sha256: str
    created_at: str


def _configs_dir(configs_dir: Path | None) -> Path:
    return (configs_dir or (project_root() / "configs")).resolve()


def discover_run_ids(configs_dir: Path | None = None) -> list[str]:
    """Run ids from data/<run_id>/ directories that have a usable config."""
    cdir = _configs_dir(configs_dir)
    data_root = _data_root()
    if not data_root.is_dir():
        return []
    return [
        p.name
        for p in sorted(data_root.iterdir())
        if p.is_dir() and is_materialized_run(p.name, cdir)
    ]


def _load_record(run_id: str, configs_dir: Path) -> RunRecord | None:
    cfg_path = configs_dir / f"{run_id}.toml"
    if not cfg_path.is_file():
        return None
    try:
        cfg = load_resolved_config(cfg_path, run_id)
    except (ValueError, OSError):
        return None
    manifest: RunManifest | None = None
    if cfg.manifest_path.is_file():
        try:
            manifest = load_run_manifest(cfg.manifest_path)
        except (OSError, ValueError):
            manifest = None
    return RunRecord(
        run_id=run_id,
        config_path=cfg_path,
        manifest=manifest,
        parent_run_id=manifest.parent_run_id if manifest else None,
        branch_pipeline=manifest.branch_pipeline if manifest else None,
        branch_step_id=manifest.branch_step_id if manifest else None,
        config_sha256=manifest.config_sha256 if manifest else "",
        created_at=manifest.created_at if manifest else "",
    )


def list_runs(configs_dir: Path | None = None) -> list[RunRecord]:
    cdir = _configs_dir(configs_dir)
    out: list[RunRecord] = []
    for rid in discover_run_ids(cdir):
        rec = _load_record(rid, cdir)
        if rec is not None:
            out.append(rec)
    return out


def format_run_tree(configs_dir: Path | None = None) -> str:
    records = {r.run_id: r for r in list_runs(configs_dir)}
    if not records:
        return "(no runs found)"

    children: dict[str | None, list[str]] = {}
    for rid, rec in records.items():
        parent = rec.parent_run_id
        children.setdefault(parent, []).append(rid)

    lines: list[str] = []

    def walk(node: str, prefix: str = "") -> None:
        rec = records[node]
        branch = f" @ {rec.branch_pipeline}" if rec.branch_pipeline else ""
        lines.append(f"{prefix}{node}{branch}")
        kids = sorted(children.get(node, []))
        for i, kid in enumerate(kids):
            is_last = i == len(kids) - 1
            ext = "└── " if is_last else "├── "
            walk(kid, prefix + ("    " if is_last else "│   "))

    roots = sorted(children.get(None, []))
    if not roots:
        for rid in sorted(records):
            lines.append(rid)
    else:
        for i, root in enumerate(roots):
            if i > 0:
                lines.append("")
            walk(root)

    orphans = sorted(set(records) - set(roots) - {c for kids in children.values() for c in kids})
    for rid in orphans:
        if rid not in roots:
            lines.append(f"{rid} (orphan: parent missing)")

    return "\n".join(lines)


def diff_configs(
    run_a: str,
    run_b: str,
    *,
    configs_dir: Path | None = None,
) -> str:
    cdir = _configs_dir(configs_dir)
    a_id = sanitize_run_id(run_a)
    b_id = sanitize_run_id(run_b)
    path_a = cdir / f"{a_id}.toml"
    path_b = cdir / f"{b_id}.toml"
    if not path_a.is_file():
        raise FileNotFoundError(f"Config not found: {path_a}")
    if not path_b.is_file():
        raise FileNotFoundError(f"Config not found: {path_b}")

    text_a = path_a.read_text(encoding="utf-8").splitlines(keepends=True)
    text_b = path_b.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = difflib.unified_diff(
        text_a,
        text_b,
        fromfile=str(path_a.name),
        tofile=str(path_b.name),
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        return f"No differences between {a_id} and {b_id}."
    return "".join(line + "\n" for line in lines)


def summarize_config_diff(run_a: str, run_b: str, *, configs_dir: Path | None = None) -> list[str]:
    """Key-level TOML diff for quick inspection."""
    cdir = _configs_dir(configs_dir)
    a_id = sanitize_run_id(run_a)
    b_id = sanitize_run_id(run_b)
    da = tomllib.loads((cdir / f"{a_id}.toml").read_text(encoding="utf-8"))
    db = tomllib.loads((cdir / f"{b_id}.toml").read_text(encoding="utf-8"))

    def flatten(d: dict, prefix: str = "") -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    fa, fb = flatten(da), flatten(db)
    keys = sorted(set(fa) | set(fb))
    changes: list[str] = []
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            changes.append(f"  {k}: {va!r} → {vb!r}")
    return changes
