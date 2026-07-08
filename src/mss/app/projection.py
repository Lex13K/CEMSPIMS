"""Pure in-memory graph projection from AppState (no disk reads)."""

from __future__ import annotations

from typing import Any

from mss.app.state_store import AppState, RunState, StepState


def _step_key(pipeline: str, step_id: str) -> str:
    return f"{pipeline}/{step_id}"


from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.step_catalog import get_step_info, iter_catalog_steps
from mss.run.experiment_graph import (
    PIPELINE_COLORS,
    GraphEdge,
    GraphNode,
    _build_spine_y,
    _catalog_step_index,
    _fork_node_for_branch,
    _node_id_global,
    _node_id_raw,
    _node_id_run,
    _node_id_terminal,
    _pipeline_index,
    _terminal_y_for_scope,
)
from mss.run.ownership import (
    branch_step_id as _branch_step_id_from_rec,
    include_owned_step_in_graph,
    owner_run_for_step,
)
from mss.run.registry import RunRecord


def _pipeline_in_scope(scope: list[str] | None, pipeline: str) -> bool:
    if scope is None:
        return True
    return pipeline in scope


def _run_record(run: RunState) -> RunRecord:
    from mss.run.manifest import RunManifest

    manifest = None
    if run.branch_pipeline or run.parent_run_id:
        manifest = RunManifest(
            run_id=run.run_id,
            parent_run_id=run.parent_run_id,
            branch_pipeline=run.branch_pipeline or "",
            branch_step_id=run.branch_step_id,
            created_at=run.created_at,
            config_path=f"configs/{run.run_id}.toml",
            pipeline_scope=run.pipeline_scope,
        )
    return RunRecord(
        run_id=run.run_id,
        config_path=__import__("pathlib").Path(f"configs/{run.run_id}.toml"),
        manifest=manifest,
        parent_run_id=run.parent_run_id,
        branch_pipeline=run.branch_pipeline,
        branch_step_id=run.branch_step_id,
        config_sha256="",
        created_at=run.created_at,
    )


def _step_status_from_state(
    state: AppState,
    run_id: str,
    pipeline: str,
    step_id: str,
    *,
    owner: str,
    inherited: bool,
) -> str:
    key = _step_key(pipeline, step_id)
    if pipeline == "data.prepare" and step_id != "raw":
        gs = state.global_steps.get(key)
        return gs.status if gs else ("complete" if inherited else "missing")
    owner_run = state.runs.get(owner)
    if owner_run is None:
        return "missing"
    st = owner_run.steps.get(key)
    if st:
        return st.status
    return "complete" if inherited else "missing"


def _include_owned_step(state: AppState, run: RunState, pipeline: str, step_id: str, owner: str) -> bool:
    key = _step_key(pipeline, step_id)
    touched = key in run.steps
    return include_owned_step_in_graph(
        parent_run_id=run.parent_run_id,
        branch_pipeline=run.branch_pipeline,
        branch_step_id=run.branch_step_id,
        pipeline=pipeline,
        step_id=step_id,
        owner=owner,
        run_id=run.run_id,
        pipeline_scope=run.pipeline_scope,
        touched=touched,
    )


def _branch_index(run: RunState) -> int:
    if run.branch_pipeline:
        return _pipeline_index(run.branch_pipeline)
    return _pipeline_index("graph.prepare")


def project_graph(state: AppState) -> dict[str, Any]:
    if not state.runs:
        return {
            "nodes": [],
            "edges": [],
            "meta": {"run_ids": [], "pipeline_colors": PIPELINE_COLORS, "revision": state.revision},
        }

    records = {rid: _run_record(r) for rid, r in state.runs.items()}
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_seq = 0

    def add_edge(
        source: str,
        target: str,
        kind: str,
        *,
        label: str | None = None,
        parent_run_id: str | None = None,
        child_run_id: str | None = None,
    ) -> None:
        nonlocal edge_seq
        eid = f"e{edge_seq}"
        edge_seq += 1
        edges.append(
            GraphEdge(
                id=eid,
                source=source,
                target=target,
                kind=kind,  # type: ignore[arg-type]
                label=label,
                parent_run_id=parent_run_id,
                child_run_id=child_run_id,
            )
        )

    y_step = 88
    lane_w = 200
    spine_y = _build_spine_y(y_step)
    run_ids_sorted = sorted(state.runs.keys())
    children_map: dict[str | None, list[str]] = {}
    for rid, run in state.runs.items():
        children_map.setdefault(run.parent_run_id, []).append(rid)
    topo: list[str] = []

    def walk(rid: str) -> None:
        topo.append(rid)
        for kid in sorted(children_map.get(rid, [])):
            walk(kid)

    for root in sorted(children_map.get(None, [])):
        walk(root)
    for rid in run_ids_sorted:
        if rid not in topo:
            topo.append(rid)
    run_ids_sorted = topo
    run_lane = {rid: i for i, rid in enumerate(run_ids_sorted)}

    raw_info = get_step_info("data.prepare", "raw")
    nodes[_node_id_raw()] = GraphNode(
        id=_node_id_raw(),
        kind="raw",
        label=raw_info.title,
        pipeline="data.prepare",
        step_id="raw",
        run_id=None,
        color=PIPELINE_COLORS["data.prepare"],
        status=state.raw_status,
        position={"x": 0, "y": 0},
    )

    prev_global = _node_id_raw()
    for pipeline, step_id in iter_catalog_steps():
        if pipeline != "data.prepare":
            break
        info = get_step_info(pipeline, step_id)
        nid = _node_id_global(pipeline, step_id)
        key = _step_key(pipeline, step_id)
        gs = state.global_steps.get(key)
        st = gs.status if gs else "missing"
        nodes[nid] = GraphNode(
            id=nid,
            kind="global_step",
            label=info.title,
            pipeline=pipeline,
            step_id=step_id,
            run_id=None,
            color=PIPELINE_COLORS[pipeline],
            status=st,
            position={"x": 0, "y": spine_y[(pipeline, step_id)]},
        )
        add_edge(prev_global, nid, "sequence")
        prev_global = nid

    last_global = prev_global
    per_run_steps = [(p, s) for p, s in iter_catalog_steps() if p != "data.prepare"]

    for rid in run_ids_sorted:
        run = state.runs[rid]
        rec = records[rid]
        bidx = _branch_index(run)
        lane_x = 0.0 if run.parent_run_id is None else float(run_lane[rid] * lane_w)
        prev_nid: str | None = None
        first_nid: str | None = None
        first_owned_nid: str | None = None
        child_prev_nid: str | None = None
        scope = run.pipeline_scope
        branch_step = run.branch_step_id or _branch_step_id_from_rec(rec)

        for pipeline, step_id in per_run_steps:
            if not _pipeline_in_scope(scope, pipeline):
                continue
            pidx = _pipeline_index(pipeline)
            owner = owner_run_for_step(rid, pipeline, step_id, records)
            if pidx < bidx and owner != rid:
                continue
            if branch_step and pipeline == run.branch_pipeline:
                if _catalog_step_index(pipeline, step_id) < _catalog_step_index(
                    run.branch_pipeline, branch_step
                ):
                    if owner != rid:
                        continue
            if pipeline == "data.prepare":
                continue
            if not _include_owned_step(state, run, pipeline, step_id, owner):
                continue

            info = get_step_info(pipeline, step_id)
            if owner == rid:
                nid = _node_id_run(rid, pipeline, step_id)
                kind = "run_step"
                st = _step_status_from_state(state, rid, pipeline, step_id, owner=owner, inherited=False)
                inherited = False
            else:
                nid = _node_id_run(owner, pipeline, step_id)
                kind = "run_step"
                st = _step_status_from_state(
                    state, owner, pipeline, step_id, owner=owner, inherited=True
                )
                inherited = True
                if nid in nodes:
                    if first_nid is None:
                        first_nid = nid
                    if prev_nid and prev_nid != nid:
                        add_edge(prev_nid, nid, "sequence")
                    prev_nid = nid
                    continue

            label = info.title
            if inherited:
                label = f"{info.title} ({owner})"

            y_pos = spine_y[(pipeline, step_id)]
            nodes[nid] = GraphNode(
                id=nid,
                kind=kind,  # type: ignore[arg-type]
                label=label,
                pipeline=pipeline,
                step_id=step_id,
                run_id=rid if not inherited else owner,
                color=PIPELINE_COLORS.get(pipeline, "#94a3b8"),
                status=st,
                position={"x": lane_x, "y": y_pos},
            )
            if first_nid is None:
                first_nid = nid
            if owner == rid:
                if first_owned_nid is None:
                    first_owned_nid = nid
                if run.parent_run_id:
                    if child_prev_nid:
                        add_edge(child_prev_nid, nid, "sequence")
                    child_prev_nid = nid
                elif prev_nid:
                    add_edge(prev_nid, nid, "sequence")
                else:
                    add_edge(last_global, nid, "sequence")
                prev_nid = nid
            elif prev_nid:
                add_edge(prev_nid, nid, "sequence")
                prev_nid = nid
            elif not run.parent_run_id:
                add_edge(last_global, nid, "sequence")
                prev_nid = nid

        if run.parent_run_id and run.branch_pipeline and first_owned_nid:
            parent = run.parent_run_id
            fork_nid = _fork_node_for_branch(
                parent, run.branch_pipeline, branch_step, records
            )
            if fork_nid and fork_nid != first_owned_nid:
                add_edge(
                    fork_nid,
                    first_owned_nid,
                    "branch",
                    parent_run_id=parent,
                    child_run_id=rid,
                )

        if _pipeline_in_scope(scope, "analysis.summarize") and run.show_terminal:
            tid = _node_id_terminal(rid)
            ty = _terminal_y_for_scope(scope, per_run_steps, spine_y, y_step)
            nodes[tid] = GraphNode(
                id=tid,
                kind="terminal",
                label=rid,
                pipeline="results",
                step_id="terminal",
                run_id=rid,
                color=PIPELINE_COLORS["results"],
                status=run.terminal_status,
                position={"x": lane_x, "y": ty},
            )
            if prev_nid:
                add_edge(prev_nid, tid, "sequence")
            elif not run.parent_run_id and last_global:
                add_edge(last_global, tid, "sequence")

    jobs = [j.to_dict() for j in state.jobs.values()]
    live_by_run = {k: v.to_dict() for k, v in state.live_by_run.items()}

    return {
        "nodes": [n.to_dict() for n in nodes.values()],
        "edges": [e.to_dict() for e in edges],
        "meta": {
            "run_ids": run_ids_sorted,
            "pipeline_colors": PIPELINE_COLORS,
            "revision": state.revision,
        },
        "jobs": jobs,
        "live_by_run": live_by_run,
    }
