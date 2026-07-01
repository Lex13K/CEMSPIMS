"""Tests for shared step status and stale fingerprint semantics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mss.app.step_status import resolve_step_status
from mss.io.config import load_resolved_config
from mss.run.manifest import RunManifest, save_run_manifest
from mss.run.registry import RunRecord
from mss.run.step_journal import StepJournal, StepJournalEntry, save_step_journal


def _cfg(tmp_path: Path, run_id: str = "child") -> tuple:
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "data" / run_id).mkdir(parents=True)
    cfg_path = configs / f"{run_id}.toml"
    cfg_path.write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n"
        "[model.evaluate]\nmetric = \"mse_log\"\n",
        encoding="utf-8",
    )
    return load_resolved_config(cfg_path, run_id), configs


def test_resolve_step_status_completed_with_false_fingerprint_is_complete_when_artifacts_ok(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg, _ = _cfg(tmp_path)
    save_step_journal(
        cfg,
        StepJournal(
            entries={
                "model.evaluate/write_forecast_panel": StepJournalEntry(
                    pipeline="model.evaluate",
                    step_id="write_forecast_panel",
                    status="completed",
                    started_at="2026-01-01T00:00:00Z",
                    completed_at="2026-01-01T01:00:00Z",
                    stale_reason="config fingerprint changed for [model.evaluate]",
                    fingerprint_match=False,
                )
            }
        ),
    )
    with patch("mss.app.step_status.step_is_complete", return_value=True):
        status, stale = resolve_step_status(cfg, "model.evaluate", "write_forecast_panel")
    assert status == "complete"
    assert stale is None


def test_resolve_step_status_inherited_parent_step_is_complete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg, configs = _cfg(tmp_path, "default")
    with patch("mss.app.step_status.step_is_complete", return_value=True):
        status, stale = resolve_step_status(
            cfg, "graph.prepare", "edges", inherited=True
        )
    assert status == "complete"
    assert stale is None


def test_resolve_step_status_completed_journal_uses_step_is_complete(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg, _ = _cfg(tmp_path, "default")
    save_step_journal(
        cfg,
        StepJournal(
            entries={
                "data.prepare/ingest": StepJournalEntry(
                    pipeline="data.prepare",
                    step_id="ingest",
                    status="completed",
                    started_at="2026-01-01T00:00:00Z",
                    completed_at="2026-01-01T01:00:00Z",
                )
            }
        ),
    )
    with patch("mss.app.step_status.step_is_complete", return_value=True) as mock_done:
        status, stale = resolve_step_status(cfg, "data.prepare", "ingest")
        mock_done.assert_called_once_with("data.prepare", "ingest", cfg)
    assert status == "complete"
    assert stale is None


def test_resolve_step_status_no_journal_uses_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg, _ = _cfg(tmp_path, "default")
    with patch("mss.app.step_status.step_is_complete", return_value=True):
        status, stale = resolve_step_status(cfg, "data.prepare", "returns_panel")
    assert status == "complete"
    assert stale is None


def test_resolve_step_status_branch_without_artifacts_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    cfg, configs = _cfg(tmp_path)
    rec = RunRecord(
        run_id="child",
        config_path=configs / "child.toml",
        manifest=None,
        parent_run_id="default",
        branch_pipeline="model.train",
        branch_step_id="train",
        config_sha256="",
        created_at="",
    )
    with patch("mss.app.step_status.step_is_complete", return_value=False):
        status, stale = resolve_step_status(
            cfg, "model.evaluate", "write_forecast_panel", rec=rec, inherited=False
        )
    assert status == "missing"
    assert stale is None
