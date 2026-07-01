"""Unit tests for model.train completeness helpers (no torch import)."""

from __future__ import annotations

import json
from pathlib import Path

from mss.config.fingerprints import (
    current_model_train_fingerprint,
    write_dataset_fingerprint_sidecar,
    write_graph_fingerprint_sidecar,
)
from mss.io.config import ResolvedConfig
from mss.model_train.checks import (
    can_resume_training,
    check_final_metrics_complete,
    model_train_step_semantically_complete,
)
from tests.conftest import make_resolved_config


def _cfg(tmp: Path) -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text(
        "[graph.universe]\nn_nodes = 500\n\n"
        "[dataset]\ntrain_end = \"2014-12-31\"\nval_end = \"2018-12-31\"\n"
        "test_end = \"2024-12-31\"\n\n"
        "[model.train]\nseed = 42\n",
        encoding="utf-8",
    )
    return make_resolved_config(tmp, source_config_path=cfg_toml.resolve())


def _stamp_train_fingerprints(cfg: ResolvedConfig) -> None:
    write_graph_fingerprint_sidecar(cfg)
    write_dataset_fingerprint_sidecar(cfg)
    fp = current_model_train_fingerprint(cfg.source_config_path)
    d = cfg.run_interim_dir / "model_train"
    d.mkdir(parents=True, exist_ok=True)
    fm = d / "final_metrics.json"
    data = json.loads(fm.read_text(encoding="utf-8")) if fm.is_file() else {}
    data.update(
        {
            "status": "completed",
            "best_epoch": data.get("best_epoch", 0),
            "best_val_loss": data.get("best_val_loss", 1.0),
            "config_fingerprint": fp,
        }
    )
    fm.write_text(json.dumps(data), encoding="utf-8")


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
        json.dumps(
            {
                "status": "completed",
                "best_epoch": 0,
                "best_val_loss": 1.0,
                "config_fingerprint": current_model_train_fingerprint(cfg.source_config_path),
            }
        ),
        encoding="utf-8",
    )
    assert not model_train_step_semantically_complete(cfg)
    _stamp_train_fingerprints(cfg)
    assert model_train_step_semantically_complete(cfg)


def test_model_train_incomplete_when_train_fingerprint_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _stamp_train_fingerprints(cfg)
    assert model_train_step_semantically_complete(cfg)
    cfg.source_config_path.write_text(
        "[graph.universe]\nn_nodes = 500\n\n"
        "[dataset]\ntrain_end = \"2014-12-31\"\nval_end = \"2018-12-31\"\n"
        "test_end = \"2024-12-31\"\n\n"
        "[model.train]\nseed = 99\n",
        encoding="utf-8",
    )
    assert not model_train_step_semantically_complete(cfg)
