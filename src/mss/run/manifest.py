"""Per-run manifest: lineage and shared-layer pointers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mss.io.config import ResolvedConfig
from mss.run.hashes import sha256_config, sha256_file
from mss.run.paths import prepare_artifact_paths


@dataclass
class RunManifest:
    schema_version: int = 1
    run_id: str = ""
    parent_run_id: str | None = None
    branch_pipeline: str | None = None
    branch_step_id: str | None = None
    created_at: str = ""
    config_path: str = ""
    config_sha256: str = ""
    shared_raw_dir: str = ""
    shared_interim_dir: str = ""
    prepare_artifact_hashes: dict[str, str] = field(default_factory=dict)
    inherited_artifacts: list[str] = field(default_factory=list)
    pipeline_scope: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        scope = data.get("pipeline_scope")
        if scope is not None and not isinstance(scope, list):
            scope = None
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data.get("run_id", "")),
            parent_run_id=data.get("parent_run_id"),
            branch_pipeline=data.get("branch_pipeline"),
            branch_step_id=data.get("branch_step_id"),
            created_at=str(data.get("created_at", "")),
            config_path=str(data.get("config_path", "")),
            config_sha256=str(data.get("config_sha256", "")),
            shared_raw_dir=str(data.get("shared_raw_dir", "")),
            shared_interim_dir=str(data.get("shared_interim_dir", "")),
            prepare_artifact_hashes=dict(data.get("prepare_artifact_hashes") or {}),
            inherited_artifacts=list(data.get("inherited_artifacts") or []),
            pipeline_scope=list(scope) if scope else None,
        )


def load_run_manifest(path: Path) -> RunManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest.from_dict(data)


def save_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def set_pipeline_scope(cfg: ResolvedConfig, pipelines: list[str]) -> RunManifest:
    """Update manifest pipeline_scope for progressive graph rendering."""
    path = cfg.manifest_path
    if not path.is_file():
        manifest = ensure_run_manifest(cfg)
    else:
        manifest = load_run_manifest(path)
    manifest.pipeline_scope = list(pipelines)
    save_run_manifest(path, manifest)
    return manifest


def extend_pipeline_scope(cfg: ResolvedConfig, add_pipelines: list[str]) -> RunManifest:
    path = cfg.manifest_path
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    manifest = load_run_manifest(path)
    from mss.pipeline.order import CANONICAL_PIPELINE_ORDER

    skip = {"data.prepare", "thesis.export", "analysis.compare_runs"}
    full = [p for p in CANONICAL_PIPELINE_ORDER if p not in skip]
    current = list(manifest.pipeline_scope or full)
    merged = current + [p for p in add_pipelines if p not in current]
    merged.sort(key=lambda p: CANONICAL_PIPELINE_ORDER.index(p))
    manifest.pipeline_scope = merged
    save_run_manifest(path, manifest)
    return manifest


def _rel_to_root(cfg: ResolvedConfig, path: Path) -> str:
    try:
        return path.resolve().relative_to(cfg.project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _collect_prepare_hashes(cfg: ResolvedConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, p in prepare_artifact_paths(cfg).items():
        if p.is_file():
            out[name] = sha256_file(p)
    return out


def build_root_manifest(cfg: ResolvedConfig) -> RunManifest:
    """Manifest for a run with no parent (created on first pipeline execution)."""
    rel_cfg = _rel_to_root(cfg, cfg.source_config_path)
    return RunManifest(
        run_id=cfg.run_id,
        parent_run_id=None,
        branch_pipeline=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_path=rel_cfg,
        config_sha256=sha256_config(cfg.source_config_path),
        shared_raw_dir=_rel_to_root(cfg, cfg.raw_dir),
        shared_interim_dir=_rel_to_root(cfg, cfg.shared_interim_dir),
        prepare_artifact_hashes=_collect_prepare_hashes(cfg),
        inherited_artifacts=[],
    )


def ensure_run_manifest(cfg: ResolvedConfig) -> RunManifest:
    """Load existing manifest or create a root manifest if missing."""
    path = cfg.manifest_path
    if path.is_file():
        manifest = load_run_manifest(path)
        hashes = _collect_prepare_hashes(cfg)
        if hashes and hashes != manifest.prepare_artifact_hashes:
            manifest.prepare_artifact_hashes = hashes
            save_run_manifest(path, manifest)
        return manifest
    manifest = build_root_manifest(cfg)
    save_run_manifest(path, manifest)
    return manifest


def refresh_prepare_hashes(cfg: ResolvedConfig) -> None:
    """Update prepare artifact hashes on an existing manifest after data.prepare."""
    path = cfg.manifest_path
    if not path.is_file():
        return
    manifest = load_run_manifest(path)
    manifest.prepare_artifact_hashes = _collect_prepare_hashes(cfg)
    save_run_manifest(path, manifest)
