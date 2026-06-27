"""Unit tests for model.train completeness helpers (no torch import)."""

from __future__ import annotations

import json
from pathlib import Path

from mss.io.config import ResolvedConfig
from mss.model_train.checks import (
    can_resume_training,
    check_final_metrics_complete,
    model_train_step_semantically_complete,
)


def _cfg(tmp: Path) -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    return ResolvedConfig(
        run_id="t",
        project_root=tmp,
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        source_config_path=cfg_toml.resolve(),
    )


def test_check_final_metrics_missing(tmp_path: Path) -> None:
    out = check_final_metrics_complete(tmp_path / "final_metrics.json")
    assert not out["passed"]
    assert any("missing" in issue for issue in out["issues"])


def test_check_final_metrics_completed(tmp_path: Path) -> None:
    p = tmp_path / "final_metrics.json"
    p.write_text(
        json.dumps({"status": "completed", "best_epoch": 1, "best_val_loss": 0.1}),
        encoding="utf-8",
    )
    out = check_final_metrics_complete(p)
    assert out["passed"]
    assert not out["issues"]


def test_check_final_metrics_wrong_status(tmp_path: Path) -> None:
    p = tmp_path / "final_metrics.json"
    p.write_text(
        json.dumps({"status": "running", "best_epoch": 1, "best_val_loss": 0.1}),
        encoding="utf-8",
    )
    out = check_final_metrics_complete(p)
    assert not out["passed"]


def test_can_resume_with_checkpoint_and_state_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    d = tmp_path / "interim" / "model_train"
    d.mkdir(parents=True)
    (d / "checkpoint.pt").write_bytes(b"x")
    (d / "training_state.json").write_text("{}", encoding="utf-8")
    assert can_resume_training(cfg)


def test_can_resume_false_when_final_metrics_completed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    d = tmp_path / "interim" / "model_train"
    d.mkdir(parents=True)
    (d / "checkpoint.pt").write_bytes(b"x")
    (d / "training_state.json").write_text("{}", encoding="utf-8")
    (d / "final_metrics.json").write_text(
        json.dumps({"status": "completed", "best_epoch": 1, "best_val_loss": 0.1}),
        encoding="utf-8",
    )
    assert not can_resume_training(cfg)


def test_model_train_step_semantically_complete(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert not model_train_step_semantically_complete(cfg)
    d = tmp_path / "interim" / "model_train"
    d.mkdir(parents=True)
    (d / "final_metrics.json").write_text(
        json.dumps({"status": "completed", "best_epoch": 0, "best_val_loss": 1.0}),
        encoding="utf-8",
    )
    assert model_train_step_semantically_complete(cfg)
