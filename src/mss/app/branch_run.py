"""Branch, configure, and launch pipeline jobs from the experiment app."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.server.job_runner import get_job_runner
from mss.app import events as app_events
from mss.app.config_patch import apply_structured_patch
from mss.app.state_store import get_state_store
from mss.config.validate import collect_config_warnings
from mss.io.config import load_resolved_config, sanitize_run_id
from mss.run.branch import branch_run
from mss.run.manifest import extend_pipeline_scope, load_run_manifest, set_pipeline_scope


def register_run_created(child_id: str, configs_dir: Path) -> None:
    cfg_path = configs_dir / f"{child_id}.toml"
    cfg = load_resolved_config(cfg_path, child_id)
    manifest = load_run_manifest(cfg.manifest_path) if cfg.manifest_path.is_file() else None
    get_state_store().apply(
        app_events.run_created(
            run_id=child_id,
            parent_run_id=manifest.parent_run_id if manifest else None,
            branch_pipeline=manifest.branch_pipeline if manifest else None,
            branch_step_id=manifest.branch_step_id if manifest else None,
            pipeline_scope=list(manifest.pipeline_scope) if manifest and manifest.pipeline_scope else None,
            created_at=manifest.created_at if manifest else datetime.now(tz=timezone.utc).isoformat(),
        )
    )


def branch_and_run(
    *,
    parent: str,
    child: str,
    at_pipeline: str,
    at_step_id: str | None = None,
    config_patch: dict[str, Any],
    pipelines: list[str],
    overwrite: list[str] | None,
    configs_dir: Path,
) -> dict[str, Any]:
    if not pipelines:
        raise ValueError("pipelines must be non-empty")
    if at_pipeline not in pipelines:
        raise ValueError(f"pipelines must include branch boundary {at_pipeline!r}")
    path = branch_run(
        parent,
        child,
        at_pipeline=at_pipeline,
        at_step_id=at_step_id,
        configs_dir=configs_dir,
    )
    if config_patch:
        apply_structured_patch(path, config_patch)
    warnings = collect_config_warnings(path, configs_dir=configs_dir)
    child_id = sanitize_run_id(child)
    cfg = load_resolved_config(path, child_id)
    set_pipeline_scope(cfg, pipelines)
    register_run_created(child_id, configs_dir)
    get_state_store().apply(
        app_events.scope_extended(run_id=child_id, pipeline_scope=pipelines)
    )
    runner = get_job_runner()
    active = [j for j in runner.list_jobs(limit=5) if j.run_id == child_id and j.status == "running"]
    if active:
        raise ValueError(f"job already running for {child_id}")
    job = runner.start(child_id, pipelines=pipelines, overwrite=overwrite)
    return {
        "config_path": str(path),
        "child": child_id,
        "warnings": warnings,
        "job": job.to_dict(),
    }


def ensure_pipelines_in_scope(
    run_id: str,
    pipelines: list[str],
    *,
    configs_dir: Path,
) -> list[str]:
    """Merge pipelines into manifest scope and sync the in-memory graph store."""
    if not pipelines:
        return []
    rid = sanitize_run_id(run_id)
    cfg_path = configs_dir / f"{rid}.toml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = load_resolved_config(cfg_path, rid)
    manifest = load_run_manifest(cfg.manifest_path) if cfg.manifest_path.is_file() else None
    current = set(manifest.pipeline_scope if manifest and manifest.pipeline_scope else [])
    add = [p for p in pipelines if p not in current]
    if not add:
        return list(manifest.pipeline_scope) if manifest and manifest.pipeline_scope else []
    manifest = extend_pipeline_scope(cfg, add)
    scope = list(manifest.pipeline_scope) if manifest.pipeline_scope else []
    get_state_store().apply(app_events.scope_extended(run_id=rid, pipeline_scope=scope))
    return scope


def extend_run_and_launch(
    run_id: str,
    *,
    add_pipelines: list[str],
    pipelines_to_run: list[str] | None,
    overwrite: list[str] | None,
    configs_dir: Path,
) -> dict[str, Any]:
    rid = sanitize_run_id(run_id)
    cfg_path = configs_dir / f"{rid}.toml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = load_resolved_config(cfg_path, rid)
    manifest = extend_pipeline_scope(cfg, add_pipelines)
    scope = list(manifest.pipeline_scope) if manifest.pipeline_scope else []
    get_state_store().apply(app_events.scope_extended(run_id=rid, pipeline_scope=scope))
    out: dict[str, Any] = {"run_id": rid, "pipeline_scope": scope}
    if pipelines_to_run:
        runner = get_job_runner()
        active = [j for j in runner.list_jobs(limit=5) if j.run_id == rid and j.status == "running"]
        if active:
            raise ValueError(f"job already running for {rid}")
        job = runner.start(rid, pipelines=pipelines_to_run, overwrite=overwrite)
        out["job"] = job.to_dict()
    return out
