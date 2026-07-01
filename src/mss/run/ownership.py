"""Which run owns execution of a pipeline step (parent chain / branch boundaries)."""

from __future__ import annotations

from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.step_catalog import iter_catalog_steps
from mss.run.registry import RunRecord


def _pipeline_index(name: str) -> int:
    return CANONICAL_PIPELINE_ORDER.index(name)


def _catalog_step_index(pipeline: str, step_id: str) -> int:
    for i, (p, s) in enumerate(iter_catalog_steps()):
        if p == pipeline and s == step_id:
            return i
    raise ValueError(f"Unknown catalog step: {pipeline}/{step_id}")


def branch_step_id(record: RunRecord) -> str | None:
    if record.manifest and record.manifest.branch_step_id:
        return record.manifest.branch_step_id
    return None


def pipeline_in_scope(scope: list[str] | None, pipeline: str) -> bool:
    if scope is None:
        return True
    return pipeline in scope


def include_owned_step_in_graph(
    *,
    parent_run_id: str | None,
    branch_pipeline: str | None,
    branch_step_id: str | None,
    pipeline: str,
    step_id: str,
    owner: str,
    run_id: str,
    pipeline_scope: list[str] | None,
    touched: bool = False,
) -> bool:
    """Whether a child-owned step should appear on the experiment graph."""
    if owner != run_id:
        return True
    if not parent_run_id:
        return True
    if not pipeline_in_scope(pipeline_scope, pipeline):
        return False
    if not branch_pipeline:
        return touched
    pidx = _pipeline_index(pipeline)
    bidx = _pipeline_index(branch_pipeline)
    if pidx > bidx:
        return True
    if pidx < bidx:
        return False
    if branch_step_id:
        return _catalog_step_index(pipeline, step_id) >= _catalog_step_index(
            branch_pipeline, branch_step_id
        )
    return True


def owner_run_for_step(
    run_id: str,
    pipeline: str,
    step_id: str,
    records: dict[str, RunRecord],
) -> str:
    """Walk parent chain to find run that owns execution of this pipeline step."""
    rec = records[run_id]
    if not rec.parent_run_id or not rec.branch_pipeline:
        return run_id

    branch_step = branch_step_id(rec)
    pidx = _pipeline_index(pipeline)
    bidx = _pipeline_index(rec.branch_pipeline)

    if pidx < bidx:
        parent = records.get(rec.parent_run_id)
        if parent is None:
            return run_id
        return owner_run_for_step(rec.parent_run_id, pipeline, step_id, records)

    if pidx > bidx:
        return run_id

    if branch_step and _catalog_step_index(pipeline, step_id) < _catalog_step_index(
        rec.branch_pipeline, branch_step
    ):
        return owner_run_for_step(rec.parent_run_id, pipeline, step_id, records)
    return run_id
