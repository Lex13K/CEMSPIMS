"""Tests for mss.run.step_journal."""

from __future__ import annotations

from mss.run.step_journal import (
    load_step_journal,
    record_step_finish,
    record_step_start,
    step_key,
)
from tests.conftest import make_resolved_config


def test_step_journal_roundtrip(tmp_path) -> None:
    cfg = make_resolved_config(tmp_path, run_id="j")
    record_step_start(cfg, "model.train", "train", overwritten=True, stale_reason=None)
    record_step_finish(cfg, "model.train", "train", status="completed", fingerprint_match=True)
    journal = load_step_journal(cfg)
    entry = journal.entries[step_key("model.train", "train")]
    assert entry.status == "completed"
    assert entry.overwritten is True
    assert entry.completed_at is not None


def test_record_step_finish_clears_stale_reason_on_success(tmp_path) -> None:
    cfg = make_resolved_config(tmp_path, run_id="j2")
    record_step_start(
        cfg,
        "model.evaluate",
        "write_forecast_panel",
        overwritten=False,
        stale_reason="config fingerprint changed for [model.evaluate]",
    )
    record_step_finish(
        cfg,
        "model.evaluate",
        "write_forecast_panel",
        status="completed",
        fingerprint_match=True,
    )
    entry = load_step_journal(cfg).entries[step_key("model.evaluate", "write_forecast_panel")]
    assert entry.stale_reason is None
    assert entry.fingerprint_match is True
