"""Resolve node detail payloads for the experiment graph API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mss.app.step_status import resolve_step_status
from mss.app.step_artifacts import artifacts_for_step, terminal_artifacts
from mss.app.step_config_map import fields_for_step, read_field_values
from mss.io.config import load_resolved_config
from mss.io.paths import project_root
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.step_catalog import get_step_info, iter_catalog_steps
from mss.run.manifest import load_run_manifest
from mss.run.ownership import owner_run_for_step
from mss.run.registry import RunRecord, list_runs, summarize_config_diff
from mss.run.step_journal import get_journal_entry, load_step_journal


def _parse_node_id(node_id: str) -> tuple[str, str | None, str, str]:
    """
    Returns (kind, run_id, pipeline, step_id).
    kind: raw | global_step | run_step | terminal
    """
    parts = node_id.split(":", 3)
    if node_id == "global:raw":
        return "raw", None, "data.prepare", "raw"
    if parts[0] == "global" and len(parts) == 3:
        return "global_step", None, parts[1], parts[2]
    if parts[0] == "run" and len(parts) == 4:
        return "run_step", parts[1], parts[2], parts[3]
    if parts[0] == "terminal" and len(parts) == 2:
        return "terminal", parts[1], "results", "terminal"
    raise ValueError(f"Unknown node id: {node_id}")


def _cfg_for_node(run_id: str | None, configs_dir):
    if not run_id:
        records = list_runs(configs_dir)
        if not records:
            raise FileNotFoundError("no runs")
        run_id = records[0].run_id
    rec = next((r for r in list_runs(configs_dir) if r.run_id == run_id), None)
    if rec is None:
        raise FileNotFoundError(f"run not found: {run_id}")
    return load_resolved_config(rec.config_path, run_id), rec


def _status_for_step(
    cfg,
    pipeline: str,
    step_id: str,
    *,
    rec: RunRecord | None = None,
    records: dict[str, RunRecord] | None = None,
    inherited: bool = False,
) -> tuple[str, str | None]:
    del records
    return resolve_step_status(cfg, pipeline, step_id, rec=rec, inherited=inherited)


def node_detail(node_id: str, *, configs_dir=None, include_artifacts: bool = False) -> dict[str, Any]:
    kind, run_id, pipeline, step_id = _parse_node_id(node_id)
    cdir = configs_dir or (project_root() / "configs")
    info = get_step_info(pipeline, step_id)

    if kind == "terminal":
        cfg, rec = _cfg_for_node(run_id, cdir)
        status, _ = _status_for_step(cfg, "analysis.summarize", "loss_figure")
        if status != "complete":
            status = "missing"
        payload: dict[str, Any] = {
            "node_id": node_id,
            "kind": kind,
            "title": info.title,
            "description": info.description,
            "run_id": run_id,
            "pipeline": pipeline,
            "step_id": step_id,
            "status": status,
            "journal": None,
            "inherited_from": None,
            "config_fields": [],
            "branch_pipeline": rec.branch_pipeline,
            "pipeline_scope": _scope_for_cfg(cfg),
            "is_shared": False,
            "can_branch": False,
            "can_run": False,
            "can_delete_run": rec.parent_run_id is not None,
        }
        if include_artifacts:
            payload["artifacts"] = terminal_artifacts(cfg)
        return payload

    cfg, rec = _cfg_for_node(run_id, cdir)
    records = {r.run_id: r for r in list_runs(cdir)}
    journal_entry = get_journal_entry(cfg, pipeline, step_id)
    journal_dict = journal_entry.to_dict() if journal_entry else None

    is_shared = kind in ("raw", "global_step")

    owner_id = run_id
    if run_id and kind == "run_step":
        owner_id = owner_run_for_step(run_id, pipeline, step_id, records)

    status_cfg = cfg
    if owner_id != run_id:
        owner_rec = records.get(owner_id)
        if owner_rec is not None:
            status_cfg = load_resolved_config(owner_rec.config_path, owner_id)

    if kind == "raw":
        from mss.run.experiment_graph import _raw_status

        status = _raw_status(cfg)
        stale = None
    else:
        status, stale = _status_for_step(
            status_cfg,
            pipeline,
            step_id,
            rec=records.get(owner_id) if owner_id else rec,
            inherited=owner_id != run_id if run_id else False,
        )

    inherited = owner_id if owner_id != run_id else None
    if inherited is None and rec.parent_run_id and rec.branch_pipeline:
        bidx = CANONICAL_PIPELINE_ORDER.index(rec.branch_pipeline)
        pidx = CANONICAL_PIPELINE_ORDER.index(pipeline) if pipeline in CANONICAL_PIPELINE_ORDER else 99
        if pidx < bidx:
            inherited = rec.parent_run_id

    fields = fields_for_step(pipeline, step_id)
    config_fields = read_field_values(cfg.source_config_path, fields) if fields else []

    owns_step = kind == "run_step" and run_id == rec.run_id and not inherited
    can_branch = owns_step and pipeline != "data.prepare"

    payload = {
        "node_id": node_id,
        "kind": kind,
        "title": info.title,
        "description": info.description,
        "pipeline": pipeline,
        "step_id": step_id,
        "run_id": run_id,
        "status": status,
        "stale_reason": stale,
        "journal": journal_dict,
        "inherited_from": inherited,
        "config_fields": config_fields,
        "branch_pipeline": rec.branch_pipeline,
        "parent_run_id": rec.parent_run_id,
        "pipeline_scope": _scope_for_cfg(cfg),
        "is_shared": is_shared,
        "can_branch": can_branch,
        "can_run": owns_step and not is_shared,
        "can_delete_run": rec.parent_run_id is not None,
    }
    if include_artifacts:
        if kind == "terminal":
            payload["artifacts"] = terminal_artifacts(cfg)
        else:
            payload["artifacts"] = artifacts_for_step(cfg, pipeline, step_id)
    return payload


def node_artifacts(node_id: str, *, configs_dir=None) -> list[dict[str, Any]]:
    kind, run_id, pipeline, step_id = _parse_node_id(node_id)
    cdir = configs_dir or (project_root() / "configs")
    cfg, _ = _cfg_for_node(run_id, cdir)
    if kind == "terminal":
        return [terminal_artifacts(cfg)]
    return artifacts_for_step(cfg, pipeline, step_id)


def _scope_for_cfg(cfg) -> list[str] | None:
    if not cfg.manifest_path.is_file():
        return None
    try:
        return load_run_manifest(cfg.manifest_path).pipeline_scope
    except (OSError, ValueError):
        return None


def pipelines_from_step(pipeline: str) -> list[str]:
    """Pipelines from branch boundary through end (app DAG pipelines only)."""
    skip = {"data.prepare", "thesis.export", "analysis.compare_runs"}
    out: list[str] = []
    started = False
    for name in CANONICAL_PIPELINE_ORDER:
        if name in skip:
            continue
        if name == pipeline:
            started = True
        if started:
            out.append(name)
    return out


def _catalog_step_index(pipeline: str, step_id: str) -> int:
    for i, (p, s) in enumerate(iter_catalog_steps()):
        if p == pipeline and s == step_id:
            return i
    raise ValueError(f"Unknown catalog step: {pipeline}/{step_id}")


def _at_or_after_branch_step(
    pipeline: str,
    step_id: str,
    *,
    at_pipeline: str,
    at_step_id: str,
) -> bool:
    return _catalog_step_index(pipeline, step_id) >= _catalog_step_index(at_pipeline, at_step_id)


def branch_fields_for_node(node_id: str, *, configs_dir=None) -> dict[str, Any]:
    """Config fields from the branch node step onward (not earlier steps)."""
    kind, run_id, pipeline, step_id = _parse_node_id(node_id)
    if kind not in ("run_step",) or not run_id:
        raise ValueError("branch fields only available for run steps")
    cdir = configs_dir or (project_root() / "configs")
    cfg, _rec = _cfg_for_node(run_id, cdir)
    pipelines = pipelines_from_step(pipeline)
    fields_by_step: list[dict[str, Any]] = []
    for p, sid in iter_catalog_steps():
        if p not in pipelines:
            continue
        if not _at_or_after_branch_step(p, sid, at_pipeline=pipeline, at_step_id=step_id):
            continue
        step_fields = fields_for_step(p, sid)
        if not step_fields:
            continue
        info = get_step_info(p, sid)
        values = read_field_values(cfg.source_config_path, step_fields)
        fields_by_step.append(
            {
                "pipeline": p,
                "step_id": sid,
                "title": info.title,
                "fields": values,
            }
        )
    parent_path = cdir / f"{run_id}.toml"
    return {
        "node_id": node_id,
        "parent_run_id": run_id,
        "at_pipeline": pipeline,
        "at_step_id": step_id,
        "fields_by_step": fields_by_step,
        "pipelines_from_here": pipelines,
        "parent_config_path": str(parent_path),
    }


def preview_config_patch(
    parent_run: str,
    child_config_path: Path,
    patch: dict[str, Any],
    *,
    configs_dir=None,
) -> list[str]:
    """Diff lines after applying patch to a copy conceptually — uses parent vs patched child."""
    from mss.app.config_patch import apply_structured_patch
    import shutil
    import tempfile

    cdir = configs_dir or (project_root() / "configs")
    parent_path = cdir / f"{parent_run}.toml"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "child.toml"
        shutil.copy2(child_config_path, tmp)
        apply_structured_patch(tmp, patch)
        return summarize_config_diff(parent_run, sanitize_run_id_from_path(tmp), configs_dir=cdir)


def sanitize_run_id_from_path(path: Path) -> str:
    return path.stem


def edge_detail(edge_id: str, graph: dict[str, Any]) -> dict[str, Any]:
    edge = next((e for e in graph.get("edges", []) if e["id"] == edge_id), None)
    if edge is None:
        raise ValueError(f"edge not found: {edge_id}")
    if edge.get("kind") != "branch":
        return {"edge_id": edge_id, "kind": edge.get("kind"), "diff": []}
    parent = edge.get("parent_run_id")
    child = edge.get("child_run_id")
    if not parent or not child:
        return {"edge_id": edge_id, "kind": "branch", "diff": []}
    diffs = summarize_config_diff(parent, child)
    return {
        "edge_id": edge_id,
        "kind": "branch",
        "parent_run_id": parent,
        "child_run_id": child,
        "label": edge.get("label"),
        "diff": diffs,
    }
