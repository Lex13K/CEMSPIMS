"""Tests for journal-first node detail status."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mss.app.node_detail import _status_for_step
from mss.app.step_status import resolve_step_status
from mss.io.config import load_resolved_config
from mss.run.step_journal import StepJournal, StepJournalEntry, save_step_journal


def test_status_for_step_uses_journal_without_disk_scan(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    run_dir = tmp_path / "data" / "default"
    (run_dir / "interim").mkdir(parents=True)
    cfg_path = configs / "default.toml"
    cfg_path.write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg = load_resolved_config(cfg_path, "default")
    journal = StepJournal(
        entries={
            "graph.prepare/edges": StepJournalEntry(
                pipeline="graph.prepare",
                step_id="edges",
                status="completed",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:05:00Z",
            )
        }
    )
    save_step_journal(cfg, journal)

    with patch("mss.app.step_status.step_is_complete", return_value=True) as mock_done:
        status, stale = _status_for_step(cfg, "graph.prepare", "edges")
        mock_done.assert_called_once()
    assert status == "complete"
    assert stale is None
