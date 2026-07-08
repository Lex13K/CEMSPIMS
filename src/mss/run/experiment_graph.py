"""Build experiment DAG JSON for the graph app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mss.io.config import load_resolved_config
from mss.io.paths import project_root
from mss.pipeline.artifacts import step_is_complete, step_stale_reason
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.step_catalog import get_step_info, iter_catalog_steps
from mss.run.manifest import load_run_manifest
from mss.run.ownership import branch_step_id as _branch_step_id, include_owned_step_in_graph, owner_run_for_step as _owner_run_for_step
from mss.run.registry import RunRecord, list_runs
from mss.run.step_journal import get_journal_entry

NodeKind = Literal["raw", "global_step", "run_step", "terminal"]
EdgeKind = Literal["sequence", "branch"]

SKIP_PIPELINES = frozenset({"thesis.export", "analysis.compare_runs"})


def _pipeline_scope_for_cfg(cfg) -> list[str] | None:
    path = cfg.manifest_path
    if not path.is_file():
        return None
    try:
        manifest = load_run_manifest(path)
        return manifest.pipeline_scope
    except (OSError, ValueError):
        return None


def _pipeline_in_scope(scope: list[str] | None, pipeline: str) -> bool:
    if scope is None:
        return True
    return pipeline in scope

PIPELINE_COLORS: dict[str, str] = {
    "data.prepare": "#3b82f6",
    "graph.prepare": "#22c55e",
    "dataset.package": "#14b8a6",
    "model.cache_graphs": "#a855f7",
    "model.train": "#f97316",
    "model.evaluate": "#ef4444",
    "analysis.summarize": "#f59e0b",
    "results": "#64748b",
}


def _pipeline_index(name: str) -> int:
    return CANONICAL_PIPELINE_ORDER.index(name)


def _node_id_global(pipeline: str, step_id: str) -> str:
    return f"global:{pipeline}:{step_id}"


def _node_id_run(run_id: str, pipeline: str, step_id: str) -> str:
    return f"run:{run_id}:{pipeline}:{step_id}"


def _node_id_terminal(run_id: str) -> str:
    return f"terminal:{run_id}"


def _node_id_raw() -> str:
    return "global:raw"


def _catalog_step_index(pipeline: str, step_id: str) -> int:
    for i, (p, s) in enumerate(iter_catalog_steps()):
        if p == pipeline and s == step_id:
            return i
    raise ValueError(f"Unknown catalog step: {pipeline}/{step_id}")


def _branch_index(record: RunRecord) -> int:
    if record.branch_pipeline:
        return _pipeline_index(record.branch_pipeline)
    return _pipeline_index("graph.prepare")


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: NodeKind
    label: str
    pipeline: str
    step_id: str
    run_id: str | None
    color: str
    status: str
    position: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "pipeline": self.pipeline,
            "step_id": self.step_id,
            "run_id": self.run_id,
            "color": self.color,
            "status": self.status,
            "position": self.position,
        }


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str | None = None
    parent_run_id: str | None = None
    child_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
        }
        if self.label:
            d["label"] = self.label
        if self.parent_run_id:
            d["parent_run_id"] = self.parent_run_id
        if self.child_run_id:
            d["child_run_id"] = self.child_run_id
        return d


def _step_status(
    cfg,
    pipeline: str,
    step_id: str,
    *,
    rec: RunRecord | None = None,
    inherited: bool = False,
) -> str:
    from mss.app.step_status import resolve_step_status

    status, _ = resolve_step_status(cfg, pipeline, step_id, rec=rec, inherited=inherited)
    return status


def _include_owned_step(cfg, rec: RunRecord, pipeline: str, step_id: str, owner: str, rid: str) -> bool:
    scope = None
    if rec.manifest and rec.manifest.pipeline_scope:
        scope = list(rec.manifest.pipeline_scope)
    touched = get_journal_entry(cfg, pipeline, step_id) is not None
    return include_owned_step_in_graph(
        parent_run_id=rec.parent_run_id,
        branch_pipeline=rec.branch_pipeline,
        branch_step_id=_branch_step_id(rec),
        pipeline=pipeline,
        step_id=step_id,
        owner=owner,
        run_id=rid,
        pipeline_scope=scope,
        touched=touched,
    )


def _raw_status(cfg) -> str:
    raw = cfg.raw_dir
    if not raw.is_dir():
        return "missing"
    needed = ("WRDS.csv", "VIX.csv", "snp500_volatility.csv")
    if all((raw / n).is_file() for n in needed):
        return "complete"
    return "missing"


def _build_spine_y(y_step: float) -> dict[tuple[str, str], float]:
    """Canonical Y for each pipeline step so all runs align on the same rows."""
    spine: dict[tuple[str, str], float] = {}
    row = 1
    for pipeline, step_id in iter_catalog_steps():
        spine[(pipeline, step_id)] = row * y_step
        row += 1
    return spine


def _terminal_y_for_scope(
    scope: list[str] | None,
    per_run_steps: list[tuple[str, str]],
    spine_y: dict[tuple[str, str], float],
    y_step: float,
) -> float:
    rows = [
        spine_y[(p, s)]
        for p, s in per_run_steps
        if _pipeline_in_scope(scope, p)
    ]
    if not rows:
        return y_step
    return max(rows) + y_step


def _fork_node_for_branch(
    parent_run_id: str,
    branch_pipeline: str,
    branch_step_id: str | None,
    records: dict[str, RunRecord],
) -> str | None:
    """Last parent node strictly before the branch step (or pipeline boundary)."""
    cutoff_idx: int | None = None
    bidx: int | None = None
    if branch_step_id:
        cutoff_idx = _catalog_step_index(branch_pipeline, branch_step_id)
    else:
        bidx = _pipeline_index(branch_pipeline)

    last: str | None = None
    for pipeline, step_id in iter_catalog_steps():
        if pipeline == "data.prepare":
            last = _node_id_global(pipeline, step_id)
            continue
        if cutoff_idx is not None:
            if _catalog_step_index(pipeline, step_id) >= cutoff_idx:
                break
        elif bidx is not None and _pipeline_index(pipeline) >= bidx:
            break
        owner = _owner_run_for_step(parent_run_id, pipeline, step_id, records)
        last = _node_id_run(owner, pipeline, step_id)
    return last


def build_experiment_graph(configs_dir=None) -> dict[str, Any]:
    records_list = list_runs(configs_dir)
    records = {r.run_id: r for r in records_list}
    if not records:
        return {"nodes": [], "edges": [], "meta": {"run_ids": []}}

    configs = configs_dir or (project_root() / "configs")
    cfgs: dict[str, Any] = {}
    for rid, rec in records.items():
        try:
            cfgs[rid] = load_resolved_config(rec.config_path, rid)
        except (OSError, ValueError):
            continue

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_seq = 0

    def add_edge(
        source: str,
        target: str,
        kind: EdgeKind,
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
                kind=kind,
                label=label,
                parent_run_id=parent_run_id,
                child_run_id=child_run_id,
            )
        )

    # Layout constants
    y_step = 88
    lane_w = 200
    spine_y = _build_spine_y(y_step)
    run_ids_sorted = sorted(records.keys())
    children_map: dict[str | None, list[str]] = {}
    for rid, rec in records.items():
        children_map.setdefault(rec.parent_run_id, []).append(rid)
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

    # Raw node
    anchor_cfg = cfgs.get(run_ids_sorted[0])
    raw_status = _raw_status(anchor_cfg) if anchor_cfg else "missing"
    raw_info = get_step_info("data.prepare", "raw")
    nodes[_node_id_raw()] = GraphNode(
        id=_node_id_raw(),
        kind="raw",
        label=raw_info.title,
        pipeline="data.prepare",
        step_id="raw",
        run_id=None,
        color=PIPELINE_COLORS["data.prepare"],
        status=raw_status,
        position={"x": 0, "y": 0},
    )

    y = 1
    prev_global = _node_id_raw()
    for pipeline, step_id in iter_catalog_steps():
        if pipeline != "data.prepare":
            break
        info = get_step_info(pipeline, step_id)
        nid = _node_id_global(pipeline, step_id)
        st = "missing"
        if anchor_cfg:
            st = _step_status(anchor_cfg, pipeline, step_id)
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
        y += 1

    last_global = prev_global
    run_first_step: dict[str, str] = {}
    run_last_step: dict[str, str] = {}

    per_run_steps = [(p, s) for p, s in iter_catalog_steps() if p != "data.prepare"]

    for rid in run_ids_sorted:
        rec = records[rid]
        cfg = cfgs.get(rid)
        if cfg is None:
            continue
        bidx = _branch_index(rec)
        lane_x = 0.0 if rec.parent_run_id is None else float(run_lane[rid] * lane_w)
        prev_nid: str | None = None
        first_nid: str | None = None
        first_owned_nid: str | None = None
        child_prev_nid: str | None = None
        scope = _pipeline_scope_for_cfg(cfg)
        branch_step = _branch_step_id(rec)

        for pipeline, step_id in per_run_steps:
            if not _pipeline_in_scope(scope, pipeline):
                continue
            pidx = _pipeline_index(pipeline)
            owner = _owner_run_for_step(rid, pipeline, step_id, records)
            owner_cfg = cfgs.get(owner)
            if owner_cfg is None:
                continue

            if pidx < bidx and owner != rid:
                continue

            if branch_step and pipeline == rec.branch_pipeline:
                if _catalog_step_index(pipeline, step_id) < _catalog_step_index(
                    rec.branch_pipeline, branch_step
                ):
                    if owner != rid:
                        continue

            if pipeline == "data.prepare":
                continue

            if not _include_owned_step(cfg, rec, pipeline, step_id, owner, rid):
                continue

            info = get_step_info(pipeline, step_id)
            if owner == rid:
                nid = _node_id_run(rid, pipeline, step_id)
                kind: NodeKind = "run_step"
                st = _step_status(cfg, pipeline, step_id, rec=rec)
                inherited = False
            else:
                nid = _node_id_run(owner, pipeline, step_id)
                kind = "run_step"
                st = _step_status(owner_cfg, pipeline, step_id, rec=records.get(owner), inherited=True)
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
                kind=kind,
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
                if rec.parent_run_id:
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
            elif not rec.parent_run_id:
                add_edge(last_global, nid, "sequence")
                prev_nid = nid

        if first_nid:
            run_first_step[rid] = first_nid
        if prev_nid:
            run_last_step[rid] = prev_nid

        if rec.parent_run_id and rec.branch_pipeline and first_owned_nid:
            parent = rec.parent_run_id
            fork_nid = _fork_node_for_branch(
                parent, rec.branch_pipeline, branch_step, records
            )
            if fork_nid and fork_nid != first_owned_nid:
                add_edge(
                    fork_nid,
                    first_owned_nid,
                    "branch",
                    parent_run_id=parent,
                    child_run_id=rid,
                )

        if _pipeline_in_scope(scope, "analysis.summarize"):
            show_terminal = not rec.parent_run_id or get_journal_entry(
                cfg, "analysis.summarize", "loss_figure"
            ) is not None
            if show_terminal:
                tid = _node_id_terminal(rid)
                from mss.pipeline import completeness as comp

                term_status = (
                    "complete"
                    if comp.analysis_summarize_loss_figure_semantically_complete(cfg)
                    else "missing"
                )
                ty = _terminal_y_for_scope(scope, per_run_steps, spine_y, y_step)
                nodes[tid] = GraphNode(
                    id=tid,
                    kind="terminal",
                    label=rid,
                    pipeline="results",
                    step_id="terminal",
                    run_id=rid,
                    color=PIPELINE_COLORS["results"],
                    status=term_status,
                    position={"x": lane_x, "y": ty},
                )
                if prev_nid:
                    add_edge(prev_nid, tid, "sequence")
                elif not rec.parent_run_id and last_global:
                    add_edge(last_global, tid, "sequence")

    return {
        "nodes": [n.to_dict() for n in nodes.values()],
        "edges": [e.to_dict() for e in edges],
        "meta": {
            "run_ids": run_ids_sorted,
            "pipeline_colors": PIPELINE_COLORS,
        },
    }
