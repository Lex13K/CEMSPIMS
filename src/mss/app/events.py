"""Typed events for the experiment app state store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "run_created",
    "run_removed",
    "scope_extended",
    "job_started",
    "job_updated",
    "job_finished",
    "step_started",
    "step_finished",
    "step_log_line",
    "raw_status",
    "snapshot",
]


@dataclass
class AppEvent:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload}


def run_created(
    *,
    run_id: str,
    parent_run_id: str | None = None,
    branch_pipeline: str | None = None,
    branch_step_id: str | None = None,
    pipeline_scope: list[str] | None = None,
    created_at: str = "",
) -> AppEvent:
    return AppEvent(
        "run_created",
        {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "branch_pipeline": branch_pipeline,
            "branch_step_id": branch_step_id,
            "pipeline_scope": pipeline_scope,
            "created_at": created_at,
        },
    )


def run_removed(*, run_id: str) -> AppEvent:
    return AppEvent("run_removed", {"run_id": run_id})


def scope_extended(*, run_id: str, pipeline_scope: list[str]) -> AppEvent:
    return AppEvent("scope_extended", {"run_id": run_id, "pipeline_scope": pipeline_scope})


def job_started(
    *,
    job_id: str,
    run_id: str,
    status: str,
    created_at: str,
    pipelines: list[str] | None = None,
    overwrite: list[str] | None = None,
) -> AppEvent:
    return AppEvent(
        "job_started",
        {
            "job_id": job_id,
            "run_id": run_id,
            "status": status,
            "created_at": created_at,
            "pipelines": pipelines,
            "overwrite": overwrite,
        },
    )


def job_updated(*, job_id: str, status: str, error: str | None = None) -> AppEvent:
    return AppEvent("job_updated", {"job_id": job_id, "status": status, "error": error})


def job_finished(
    *,
    job_id: str,
    run_id: str,
    status: str,
    return_code: int | None = None,
    error: str | None = None,
) -> AppEvent:
    return AppEvent(
        "job_finished",
        {
            "job_id": job_id,
            "run_id": run_id,
            "status": status,
            "return_code": return_code,
            "error": error,
        },
    )


def step_started(*, run_id: str, pipeline: str, step_id: str) -> AppEvent:
    return AppEvent(
        "step_started",
        {"run_id": run_id, "pipeline": pipeline, "step_id": step_id},
    )


def step_finished(
    *,
    run_id: str,
    pipeline: str,
    step_id: str,
    status: str,
    error: str | None = None,
) -> AppEvent:
    return AppEvent(
        "step_finished",
        {
            "run_id": run_id,
            "pipeline": pipeline,
            "step_id": step_id,
            "status": status,
            "error": error,
        },
    )


def step_log_line(
    *,
    job_id: str,
    run_id: str,
    line: str,
    ts: str | None = None,
) -> AppEvent:
    return AppEvent(
        "step_log_line",
        {"job_id": job_id, "run_id": run_id, "line": line, "ts": ts},
    )


def raw_status(*, status: str) -> AppEvent:
    return AppEvent("raw_status", {"status": status})


def snapshot(*, data: dict[str, Any]) -> AppEvent:
    return AppEvent("snapshot", data)
