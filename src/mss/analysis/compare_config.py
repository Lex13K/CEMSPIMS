"""`[analysis.compare_runs]` TOML configuration and comparison presets."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from mss.io.config import ResolvedConfig, load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.run.manifest import load_run_manifest
from mss.run.registry import summarize_config_diff


@dataclass(frozen=True)
class RunLabelOverride:
    run_id: str
    label: str
    column_key: str


@dataclass(frozen=True)
class CompareRunsConfig:
    comparison_id: str
    peer_runs: tuple[str, ...]
    run_labels: tuple[RunLabelOverride, ...]


@dataclass(frozen=True)
class CompareRunSpec:
    run_id: str
    config_path: Path
    label: str
    column_key: str
    primary_variation: str
    cfg: ResolvedConfig


def _sanitize_column_key(run_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", run_id.strip())
    return s.strip("_") or "run"


def config_path_for_run(configs_dir: Path, run_id: str) -> Path:
    return (configs_dir / f"{run_id}.toml").resolve()


def load_compare_runs_config(config_path: Path) -> CompareRunsConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    cr = dict(data.get("analysis", {}).get("compare_runs", {}) or {})
    raw_peers = cr.get("peer_runs")
    if raw_peers is None:
        peers: tuple[str, ...] = ()
    else:
        peers = tuple(str(x).strip() for x in list(raw_peers) if str(x).strip())
    comparison_id = str(cr.get("comparison_id", "")).strip()
    overrides: list[RunLabelOverride] = []
    for item in list(cr.get("run_labels") or []):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("run_id", "")).strip()
        if not rid:
            continue
        label = str(item.get("label", rid)).strip() or rid
        col = str(item.get("column_key", _sanitize_column_key(rid))).strip()
        overrides.append(RunLabelOverride(run_id=rid, label=label, column_key=col))
    return CompareRunsConfig(
        comparison_id=comparison_id,
        peer_runs=peers,
        run_labels=tuple(overrides),
    )


def _label_override(
    overrides: tuple[RunLabelOverride, ...], run_id: str
) -> RunLabelOverride | None:
    for o in overrides:
        if o.run_id == run_id:
            return o
    return None


def _primary_variation_vs_anchor(
    anchor_id: str,
    run_id: str,
    *,
    configs_dir: Path,
) -> str:
    if run_id == anchor_id:
        return "anchor (reference)"
    try:
        manifest_path = project_root() / "data" / run_id / "run_manifest.json"
        if manifest_path.is_file():
            man = load_run_manifest(manifest_path)
            if man.parent_run_id == anchor_id and man.branch_pipeline:
                return f"branched from {anchor_id} at {man.branch_pipeline}"
    except (OSError, ValueError):
        pass
    try:
        changes = summarize_config_diff(anchor_id, run_id, configs_dir=configs_dir)
        if changes:
            first = changes[0].strip()
            if first.startswith("  "):
                first = first[2:]
            return first
    except (FileNotFoundError, ValueError):
        pass
    return "config differs from anchor"


def build_compare_run_specs(
    anchor_run_id: str,
    peer_run_ids: tuple[str, ...],
    *,
    configs_dir: Path | None = None,
    label_overrides: tuple[RunLabelOverride, ...] = (),
) -> tuple[CompareRunSpec, ...]:
    """Build specs for anchor + peers (anchor first)."""
    cdir = (configs_dir or (project_root() / "configs")).resolve()
    anchor_id = sanitize_run_id(anchor_run_id)
    seen: set[str] = {anchor_id}
    ordered: list[str] = [anchor_id]
    for raw in peer_run_ids:
        rid = sanitize_run_id(raw)
        if rid in seen:
            continue
        seen.add(rid)
        ordered.append(rid)

    specs: list[CompareRunSpec] = []
    for rid in ordered:
        cfg_path = config_path_for_run(cdir, rid)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found for run {rid!r}: {cfg_path}")
        cfg = load_resolved_config(cfg_path, rid)
        ov = _label_override(label_overrides, rid)
        label = ov.label if ov else rid
        column_key = ov.column_key if ov else _sanitize_column_key(rid)
        variation = _primary_variation_vs_anchor(anchor_id, rid, configs_dir=cdir)
        specs.append(
            CompareRunSpec(
                run_id=rid,
                config_path=cfg_path,
                label=label,
                column_key=column_key,
                primary_variation=variation,
                cfg=cfg,
            )
        )
    return tuple(specs)


COMPARE_PRESET_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "v2_ablations": {
        "default": ("v2 baseline (512x4)", "default"),
        "default_train_scale": ("512x6 high lr", "default_train_scale"),
        "scale_gpu": ("1000 nodes", "scale_gpu"),
    },
}


def preset_label_overrides(preset_id: str) -> tuple[RunLabelOverride, ...]:
    labels = COMPARE_PRESET_LABELS.get(preset_id, {})
    return tuple(
        RunLabelOverride(run_id=rid, label=label, column_key=col)
        for rid, (label, col) in labels.items()
    )


COMPARE_PRESETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "v2_ablations": (
        "default",
        ("default_train_scale", "scale_gpu"),
    ),
}


def resolve_preset(preset_id: str) -> tuple[str, tuple[str, ...]]:
    key = str(preset_id).strip()
    if key not in COMPARE_PRESETS:
        allowed = ", ".join(sorted(COMPARE_PRESETS))
        raise ValueError(f"Unknown compare preset {preset_id!r}; expected one of: {allowed}")
    return COMPARE_PRESETS[key]
