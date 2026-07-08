"""Shared step status resolution for graph nodes and node detail API."""

from __future__ import annotations

from mss.io.config import ResolvedConfig
from mss.pipeline.artifacts import step_is_complete
from mss.run.registry import RunRecord
from mss.run.step_journal import get_journal_entry


def resolve_step_status(
    cfg: ResolvedConfig,
    pipeline: str,
    step_id: str,
    *,
    rec: RunRecord | None = None,
    inherited: bool = False,
) -> tuple[str, str | None]:
    """
    Return (status, stale_reason) for display.

    UI statuses: running | complete | missing
    (Stale is not shown — completed steps with artifacts count as complete.)
    """
    del rec
    entry = get_journal_entry(cfg, pipeline, step_id)

    if entry is not None:
        if entry.status == "running":
            return "running", None
        if entry.status == "completed":
            if step_is_complete(pipeline, step_id, cfg):
                return "complete", None
            return "missing", None
        if entry.status == "failed":
            return "missing", entry.error
        if entry.status == "skipped":
            return "missing", None

    if inherited or step_is_complete(pipeline, step_id, cfg):
        return "complete", None
    return "missing", None
