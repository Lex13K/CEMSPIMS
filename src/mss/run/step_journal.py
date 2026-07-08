"""Per-run step execution journal for the experiment graph app."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mss.io.config import ResolvedConfig

StepStatus = Literal["completed", "skipped", "failed", "running"]


def step_key(pipeline: str, step_id: str) -> str:
    return f"{pipeline}/{step_id}"


@dataclass
class StepJournalEntry:
    pipeline: str
    step_id: str
    status: StepStatus
    started_at: str
    completed_at: str | None = None
    overwritten: bool = False
    stale_reason: str | None = None
    fingerprint_match: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepJournal:
    schema_version: int = 1
    entries: dict[str, StepJournalEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepJournal:
        entries: dict[str, StepJournalEntry] = {}
        for key, raw in (data.get("entries") or {}).items():
            if not isinstance(raw, dict):
                continue
            entries[key] = StepJournalEntry(
                pipeline=str(raw.get("pipeline", "")),
                step_id=str(raw.get("step_id", "")),
                status=raw.get("status", "completed"),  # type: ignore[arg-type]
                started_at=str(raw.get("started_at", "")),
                completed_at=raw.get("completed_at"),
                overwritten=bool(raw.get("overwritten", False)),
                stale_reason=raw.get("stale_reason"),
                fingerprint_match=raw.get("fingerprint_match"),
                error=raw.get("error"),
            )
        return cls(schema_version=int(data.get("schema_version", 1)), entries=entries)


def journal_path(cfg: ResolvedConfig) -> Path:
    return cfg.run_data_dir / "step_journal.json"


def load_step_journal(cfg: ResolvedConfig) -> StepJournal:
    path = journal_path(cfg)
    if not path.is_file():
        return StepJournal()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StepJournal.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return StepJournal()


def save_step_journal(cfg: ResolvedConfig, journal: StepJournal) -> None:
    path = journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(journal.to_dict(), indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_step_start(
    cfg: ResolvedConfig,
    pipeline: str,
    step_id: str,
    *,
    overwritten: bool,
    stale_reason: str | None,
) -> None:
    journal = load_step_journal(cfg)
    key = step_key(pipeline, step_id)
    journal.entries[key] = StepJournalEntry(
        pipeline=pipeline,
        step_id=step_id,
        status="running",
        started_at=_utc_now(),
        overwritten=overwritten,
        stale_reason=stale_reason,
    )
    save_step_journal(cfg, journal)


def record_step_finish(
    cfg: ResolvedConfig,
    pipeline: str,
    step_id: str,
    *,
    status: StepStatus,
    fingerprint_match: bool | None = None,
    error: str | None = None,
) -> None:
    journal = load_step_journal(cfg)
    key = step_key(pipeline, step_id)
    entry = journal.entries.get(key)
    if entry is None:
        entry = StepJournalEntry(
            pipeline=pipeline,
            step_id=step_id,
            status=status,
            started_at=_utc_now(),
        )
    entry.status = status
    entry.completed_at = _utc_now()
    if fingerprint_match is not None:
        entry.fingerprint_match = fingerprint_match
    if error is not None:
        entry.error = error
    if status == "completed":
        if fingerprint_match is True:
            entry.stale_reason = None
        elif fingerprint_match is False:
            from mss.pipeline.artifacts import step_stale_reason

            entry.stale_reason = step_stale_reason(pipeline, step_id, cfg)
    journal.entries[key] = entry
    save_step_journal(cfg, journal)


def get_journal_entry(
    cfg: ResolvedConfig, pipeline: str, step_id: str
) -> StepJournalEntry | None:
    return load_step_journal(cfg).entries.get(step_key(pipeline, step_id))
