"""FastAPI application for the CEMSPIMS experiment graph."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.server.job_runner import get_job_runner
from app.server.sse import stream_events
from mss.analysis.compare_config import (
    build_compare_run_specs,
    load_compare_runs_config,
    preset_label_overrides,
    resolve_preset,
)
from mss.analysis.compare_runs import run_compare_runs
from mss.app import events as app_events
from mss.app.branch_run import branch_and_run, extend_run_and_launch, register_run_created
from mss.app.config_patch import apply_structured_patch
from mss.app.node_detail import (
    branch_fields_for_node,
    edge_detail,
    node_artifacts,
    pipelines_from_step,
)
from mss.app.projection import project_graph
from mss.app.state_bootstrap import build_state_from_disk, bootstrap_state_store
from mss.app.state_store import get_state_store, init_state_store
from mss.app.step_config_map import fields_for_step, read_field_values
from mss.config.validate import collect_config_warnings
from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.run.branch import branch_run
from mss.run.registry import list_runs
from mss.run.step_journal import load_step_journal


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = init_state_store()
    bootstrap_state_store(store, configs_dir=project_root() / "configs")
    store.set_event_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="CEMSPIMS Experiment App", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _configs_dir() -> Path:
    return project_root() / "configs"


class BranchRequest(BaseModel):
    parent: str
    child: str
    at_pipeline: str
    at_step_id: str | None = None
    config_patch: dict[str, Any] = Field(default_factory=dict)


class BranchAndRunRequest(BaseModel):
    parent: str
    child: str
    at_pipeline: str
    at_step_id: str | None = None
    config_patch: dict[str, Any] = Field(default_factory=dict)
    pipelines: list[str]
    overwrite: list[str] | None = None


class ExtendScopeRequest(BaseModel):
    add_pipelines: list[str]
    pipelines_to_run: list[str] | None = None
    overwrite: list[str] | None = None


class JobRequest(BaseModel):
    run_id: str
    pipelines: list[str] | None = None
    overwrite: list[str] | None = None


class CompareRequest(BaseModel):
    anchor: str
    peer_runs: list[str]
    comparison_id: str
    preset: str | None = None
    overwrite: bool = False


class ConfigPatchRequest(BaseModel):
    content: str


def _graph_payload() -> dict[str, Any]:
    return project_graph(get_state_store().get_state())


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    return StreamingResponse(
        stream_events(get_state_store()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/state/rescan")
def rescan_state() -> dict[str, Any]:
    store = get_state_store()
    state = build_state_from_disk(_configs_dir())
    store.replace_state(state)
    return {"revision": store.get_revision(), "runs": len(state.runs)}


@app.get("/api/graph")
def get_graph_endpoint() -> dict[str, Any]:
    return _graph_payload()


@app.get("/api/runs")
def get_runs() -> list[dict[str, Any]]:
    store = get_state_store().get_state()
    out: list[dict[str, Any]] = []
    for rid, run in store.runs.items():
        out.append(
            {
                "run_id": rid,
                "parent_run_id": run.parent_run_id,
                "branch_pipeline": run.branch_pipeline,
                "created_at": run.created_at,
                "summarize_complete": run.terminal_status == "complete",
                "pipeline_scope": run.pipeline_scope,
            }
        )
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    rid = sanitize_run_id(run_id)
    rec = next((r for r in list_runs(_configs_dir()) if r.run_id == rid), None)
    if rec is None:
        raise HTTPException(404, "run not found")
    cfg = load_resolved_config(rec.config_path, rid)
    journal = load_step_journal(cfg).to_dict()
    warnings = collect_config_warnings(rec.config_path, configs_dir=_configs_dir())
    return {
        "run_id": rid,
        "manifest": rec.manifest.to_dict() if rec.manifest else None,
        "journal": journal,
        "warnings": warnings,
        "config_path": str(rec.config_path),
    }


@app.delete("/api/runs/{run_id}")
def delete_run_endpoint(run_id: str) -> dict[str, Any]:
    from mss.run.delete_run import delete_run

    rid = sanitize_run_id(run_id)
    if rid == "default":
        children = [
            r.run_id
            for r in list_runs(_configs_dir())
            if r.parent_run_id == rid
        ]
        if children:
            raise HTTPException(
                400,
                f"cannot delete default while child runs exist: {', '.join(children)}",
            )
    runner = get_job_runner()
    for job in runner.list_jobs(limit=500):
        if job.run_id != rid:
            continue
        if job.status == "running":
            try:
                runner.cancel(job.job_id)
            except ValueError:
                pass
        else:
            runner._purge_job_files(job.job_id)
    try:
        delete_run(rid, configs_dir=_configs_dir())
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    get_state_store().apply(app_events.run_removed(run_id=rid))
    return {"deleted": rid, "revision": get_state_store().get_revision()}


@app.get("/api/nodes/{node_id:path}/config-fields")
def get_node_config_fields(node_id: str) -> dict[str, Any]:
    from mss.app.node_detail import _parse_node_id

    try:
        kind, run_id, pipeline, step_id = _parse_node_id(node_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    fields = fields_for_step(pipeline, step_id)
    parent_fields: list[dict[str, Any]] = []
    if run_id:
        rec = next((r for r in list_runs(_configs_dir()) if r.run_id == run_id), None)
        if rec and rec.parent_run_id:
            parent_path = _configs_dir() / f"{rec.parent_run_id}.toml"
            if parent_path.is_file() and fields:
                parent_fields = read_field_values(parent_path, fields)
    if run_id:
        cfg_path = _configs_dir() / f"{run_id}.toml"
        values = read_field_values(cfg_path, fields) if fields else []
    else:
        recs = list_runs(_configs_dir())
        values = read_field_values(recs[0].config_path, fields) if recs and fields else []
    return {
        "node_id": node_id,
        "pipeline": pipeline,
        "step_id": step_id,
        "fields": values,
        "parent_fields": parent_fields,
        "pipelines_from_here": pipelines_from_step(pipeline) if pipeline in CANONICAL_PIPELINE_ORDER else [],
    }


@app.get("/api/nodes/{node_id:path}/branch-fields")
def get_node_branch_fields(node_id: str) -> dict[str, Any]:
    try:
        return branch_fields_for_node(node_id, configs_dir=_configs_dir())
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/nodes/{node_id:path}/artifacts")
def get_node_artifacts(node_id: str) -> dict[str, Any]:
    try:
        return {"artifacts": node_artifacts(node_id, configs_dir=_configs_dir())}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/nodes/{node_id:path}")
def get_node(node_id: str) -> dict[str, Any]:
    """Config/journal detail on demand — graph status comes from SSE store."""
    from mss.app.node_detail import node_detail

    cdir = _configs_dir()
    try:
        return node_detail(node_id, configs_dir=cdir, include_artifacts=False)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/edges/{edge_id}")
def get_edge(edge_id: str) -> dict[str, Any]:
    graph = _graph_payload()
    try:
        return edge_detail(edge_id, graph)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/runs/branch")
def post_branch(req: BranchRequest) -> dict[str, Any]:
    try:
        path = branch_run(
            req.parent,
            req.child,
            at_pipeline=req.at_pipeline,
            at_step_id=req.at_step_id,
            configs_dir=_configs_dir(),
        )
        if req.config_patch:
            apply_structured_patch(path, req.config_patch)
        register_run_created(sanitize_run_id(req.child), _configs_dir())
        return {
            "config_path": str(path),
            "child": req.child,
            "revision": get_state_store().get_revision(),
        }
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/runs/branch-and-run")
def post_branch_and_run(req: BranchAndRunRequest) -> dict[str, Any]:
    try:
        out = branch_and_run(
            parent=req.parent,
            child=req.child,
            at_pipeline=req.at_pipeline,
            at_step_id=req.at_step_id,
            config_patch=req.config_patch,
            pipelines=req.pipelines,
            overwrite=req.overwrite,
            configs_dir=_configs_dir(),
        )
        out["revision"] = get_state_store().get_revision()
        return out
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/runs/{run_id}/extend-scope")
def post_extend_scope(run_id: str, req: ExtendScopeRequest) -> dict[str, Any]:
    try:
        out = extend_run_and_launch(
            run_id,
            add_pipelines=req.add_pipelines,
            pipelines_to_run=req.pipelines_to_run,
            overwrite=req.overwrite,
            configs_dir=_configs_dir(),
        )
        out["revision"] = get_state_store().get_revision()
        return out
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@app.patch("/api/runs/{run_id}/config")
def patch_config(run_id: str, req: ConfigPatchRequest) -> dict[str, Any]:
    rid = sanitize_run_id(run_id)
    path = _configs_dir() / f"{rid}.toml"
    if not path.is_file():
        raise HTTPException(404, "config not found")
    path.write_text(req.content, encoding="utf-8")
    warnings = collect_config_warnings(path, configs_dir=_configs_dir())
    return {"warnings": warnings, "revision": get_state_store().get_revision()}


@app.get("/api/runs/{run_id}/config")
def get_config(run_id: str) -> dict[str, str]:
    rid = sanitize_run_id(run_id)
    path = _configs_dir() / f"{rid}.toml"
    if not path.is_file():
        raise HTTPException(404, "config not found")
    return {"content": path.read_text(encoding="utf-8")}


@app.post("/api/runs/{run_id}/config/preview-patch")
def preview_patch(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Preview diff vs parent after applying patch (branch wizard step 2)."""
    parent = body.get("parent")
    patch = body.get("config_patch") or {}
    child = body.get("child") or run_id
    if not parent:
        return {"diff": [], "warnings": []}
    import shutil
    import tempfile
    import tomllib

    parent_path = _configs_dir() / f"{sanitize_run_id(parent)}.toml"
    if not parent_path.is_file():
        raise HTTPException(404, "parent config not found")

    def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
        return out

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"{child}.toml"
        child_path = _configs_dir() / f"{sanitize_run_id(child)}.toml"
        shutil.copy2(child_path if child_path.is_file() else parent_path, tmp)
        apply_structured_patch(tmp, patch)
        pflat = _flatten(tomllib.loads(parent_path.read_text(encoding="utf-8")))
        cflat = _flatten(tomllib.loads(tmp.read_text(encoding="utf-8")))
        diff_lines = []
        for k in sorted(set(pflat) | set(cflat)):
            if pflat.get(k) != cflat.get(k):
                diff_lines.append(f"  {k}: {pflat.get(k)!r} -> {cflat.get(k)!r}")
        warnings = collect_config_warnings(tmp, configs_dir=_configs_dir())
        return {"diff": diff_lines, "warnings": warnings}


@app.post("/api/jobs")
def post_job(req: JobRequest) -> dict[str, Any]:
    runner = get_job_runner()
    active = [j for j in runner.list_jobs(limit=5) if j.run_id == req.run_id and j.status == "running"]
    if active:
        raise HTTPException(409, f"job already running for {req.run_id}")
    if req.pipelines:
        from mss.app.branch_run import ensure_pipelines_in_scope

        ensure_pipelines_in_scope(req.run_id, req.pipelines, configs_dir=_configs_dir())
    rec = runner.start(req.run_id, pipelines=req.pipelines, overwrite=req.overwrite)
    return {**rec.to_dict(), "revision": get_state_store().get_revision()}


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    runner = get_job_runner()
    configs = _configs_dir()
    runner.prune_stale_jobs(configs_dir=configs)
    out: list[dict[str, Any]] = []
    for j in runner.list_jobs(limit=200):
        rec = runner.reconcile_job(j, configs_dir=configs)
        if rec is not None:
            out.append(rec.to_dict())
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    rec = get_job_runner().load(job_id)
    if rec is None:
        raise HTTPException(404, "job not found")
    d = rec.to_dict()
    d["log"] = get_job_runner().read_log(job_id)
    return d


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    try:
        out = get_job_runner().cancel(job_id).to_dict()
        return out
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@app.websocket("/api/jobs/{job_id}/stream")
async def ws_job_stream(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    runner = get_job_runner()
    last_len = 0
    try:
        while True:
            log = runner.read_log(job_id)
            if len(log) > last_len:
                await websocket.send_text(log[last_len:])
                last_len = len(log)
            rec = runner.load(job_id)
            if rec and rec.status in ("completed", "failed", "cancelled"):
                await websocket.send_text(f"\n[job {rec.status}]\n")
                break
            import asyncio

            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        return


@app.get("/api/compare/eligible")
def compare_eligible() -> list[str]:
    return [r["run_id"] for r in get_runs() if r.get("summarize_complete")]


@app.post("/api/compare")
def post_compare(req: CompareRequest) -> dict[str, Any]:
    cdir = _configs_dir()
    anchor_id = sanitize_run_id(req.anchor)
    if req.preset:
        anchor_id, peers = resolve_preset(req.preset)
        overrides = preset_label_overrides(req.preset)
        comparison_id = req.comparison_id or req.preset
    else:
        peers = tuple(sanitize_run_id(p) for p in req.peer_runs)
        overrides = ()
        comparison_id = req.comparison_id
    cfg_path = cdir / f"{anchor_id}.toml"
    if not cfg_path.is_file():
        raise HTTPException(404, "anchor config not found")
    anchor_cfg = load_resolved_config(cfg_path, anchor_id)
    cr = load_compare_runs_config(cfg_path)
    from mss.analysis.compare_config import RunLabelOverride

    by_id: dict[str, RunLabelOverride] = {o.run_id: o for o in overrides}
    for o in cr.run_labels:
        by_id[o.run_id] = o
    merged = tuple(by_id.values())
    specs = build_compare_run_specs(anchor_id, peers, configs_dir=cdir, label_overrides=merged)
    try:
        out = run_compare_runs(anchor_cfg, specs, comparison_id, overwrite=req.overwrite)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
    return {"output": str(out), "comparison_id": comparison_id}


@app.get("/api/pipelines")
def get_pipelines() -> dict[str, Any]:
    skip = {"data.prepare", "thesis.export", "analysis.compare_runs"}
    order = [p for p in CANONICAL_PIPELINE_ORDER if p not in skip]
    return {"order": order, "full_order": list(CANONICAL_PIPELINE_ORDER)}


_spa_mounted = False


def mount_static(web_dist: Path) -> None:
    """Serve the built SPA; unknown paths fall back to index.html for client routing."""
    global _spa_mounted
    if _spa_mounted or not web_dist.is_dir():
        return
    index_html = web_dist / "index.html"
    if not index_html.is_file():
        return

    assets_dir = web_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def spa_index() -> FileResponse:
        return FileResponse(index_html)

    @app.get("/{spa_path:path}")
    def spa_fallback(spa_path: str) -> FileResponse:
        if spa_path.startswith("api/"):
            raise HTTPException(404)
        candidate = web_dist / spa_path
        if spa_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)

    _spa_mounted = True
