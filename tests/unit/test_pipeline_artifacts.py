"""Tests for mss.pipeline.artifacts step completion predicates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mss.io.config import ResolvedConfig
from mss.pipeline.artifacts import (
    ensure_step_inputs_ready,
    ingest_step_complete,
    returns_panel_step_complete,
    step_is_complete,
    targets_step_complete,
)


def _cfg(tmp: Path, *, run: str = "t") -> ResolvedConfig:
    (tmp / "cfg.toml").write_text(
        "[analysis.summarize]\ndraw_hypothesis_table = false\n", encoding="utf-8"
    )
    root = tmp
    return ResolvedConfig(
        run_id=run,
        project_root=root,
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        source_config_path=tmp / "cfg.toml",
    )


def test_ingest_empty_not_complete(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert not ingest_step_complete(cfg)


def test_ingest_complete_when_manifest_and_parquets(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    interim = tmp_path / "interim"
    crsp = interim / "crsp_parquet" / "year=2020"
    crsp.mkdir(parents=True)
    (crsp / "data.parquet").write_bytes(b"x")
    (interim / "ingest_manifest.json").write_text("{}", encoding="utf-8")
    (interim / "sp500_returns.parquet").write_bytes(b"x")
    (interim / "vix.parquet").write_bytes(b"x")
    assert ingest_step_complete(cfg)


def _minimal_returns_panel_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "permno": [10001, 10001],
            "permco": [1, 1],
            "shrcd": [10, 10],
            "exchcd": [3, 3],
            "ret_used": [0.01, -0.02],
            "prc": [100.0, 98.0],
            "vol": [1000, 1100],
            "shrout": [1000000, 1000000],
            "dlstcd": [None, None],
            "dlret": [None, None],
            "dlretx": [None, None],
            "dlprc": [None, None],
            "dlpdt": [None, None],
            "ticker": ["X", "X"],
            "comnam": ["X", "X"],
            "ncusip": [None, None],
            "cusip": [None, None],
            "secstat": [None, None],
        }
    )
    df.to_parquet(path, index=False)


def _minimal_targets_parquet_manifest(tg_path: Path, man_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "rv_fwd_30": [0.04, 0.05],
            "log_rv_fwd_30": [-3.2, -3.0],
            "vix": [18.0, 19.0],
        }
    )
    df.to_parquet(tg_path, index=False)
    meta = {
        "stats": {"n_rows": 2},
        "outputs": {"targets_path": str(tg_path)},
    }
    man_path.write_text(json.dumps(meta), encoding="utf-8")


def test_returns_panel_and_targets_complete(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert not returns_panel_step_complete(cfg)
    assert not targets_step_complete(cfg)
    (tmp_path / "interim" / "returns_panel.parquet").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "interim" / "returns_panel.parquet").write_bytes(b"x")
    assert not returns_panel_step_complete(cfg)
    _minimal_returns_panel_parquet(tmp_path / "interim" / "returns_panel.parquet")
    assert returns_panel_step_complete(cfg)
    interim = tmp_path / "interim"
    _minimal_targets_parquet_manifest(interim / "targets.parquet", interim / "targets_manifest.json")
    assert targets_step_complete(cfg)


def test_step_is_complete_unknown_pipeline(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert not step_is_complete("graph.prepare", "ingest", cfg)


def test_ensure_returns_panel_needs_crsp(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="crsp_parquet"):
        ensure_step_inputs_ready("data.prepare", "returns_panel", cfg)


def test_ensure_targets_needs_sp500_vix(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="sp500_returns"):
        ensure_step_inputs_ready("data.prepare", "targets", cfg)


def test_step_is_complete_model_train_requires_completed_final_metrics(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert not step_is_complete("model.train", "train", cfg)
    mt = tmp_path / "interim" / "model_train"
    mt.mkdir(parents=True)
    (mt / "checkpoint.pt").write_bytes(b"x")
    assert not step_is_complete("model.train", "train", cfg)
    (mt / "final_metrics.json").write_text(
        '{"status": "completed", "best_epoch": 1, "best_val_loss": 0.0}',
        encoding="utf-8",
    )
    assert step_is_complete("model.train", "train", cfg)


def test_ensure_model_train_needs_manifest(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="manifest.json"):
        ensure_step_inputs_ready("model.train", "train", cfg)


def test_ensure_model_evaluate_score_splits_needs_manifest(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="manifest.json"):
        ensure_step_inputs_ready("model.evaluate", "score_splits", cfg)


def test_ensure_analysis_summarize_loss_figure_needs_training_state(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="training_state.json"):
        ensure_step_inputs_ready("analysis.summarize", "loss_figure", cfg)


def test_ensure_analysis_summarize_loss_figure_needs_test_loss(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    mt = tmp_path / "interim" / "model_train"
    mt.mkdir(parents=True, exist_ok=True)
    (mt / "training_state.json").write_text(
        '{"metrics_history":[{"epoch":1,"train_loss":0.1,"val_loss":0.2}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test_loss.json"):
        ensure_step_inputs_ready("analysis.summarize", "loss_figure", cfg)


def test_ensure_analysis_summarize_loss_figure_needs_forecasts(tmp_path) -> None:
    import json

    import pandas as pd

    cfg = _cfg(tmp_path)
    mt = tmp_path / "interim" / "model_train"
    mt.mkdir(parents=True, exist_ok=True)
    (mt / "training_state.json").write_text(
        '{"metrics_history":[{"epoch":1,"train_loss":0.1,"val_loss":0.2}]}',
        encoding="utf-8",
    )
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "test_loss.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "metric": "mse_log",
                "split": "test",
                "n_rows": 1,
                "value": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forecasts.parquet"):
        ensure_step_inputs_ready("analysis.summarize", "loss_figure", cfg)

    fc = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
            "sample": ["test_full"],
            "y_true": [0.1],
            "y_pred_model": [0.2],
        }
    )
    fc.to_parquet(proc / "forecasts.parquet", index=False)
    with pytest.raises(ValueError, match="summary_table"):
        ensure_step_inputs_ready("analysis.summarize", "loss_figure", cfg)

    summ = proc / "summaries"
    summ.mkdir(parents=True, exist_ok=True)
    (summ / "summary_table.csv").write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n"
        "gnn,full_test,0.01,0.02,1.0,1.1,1.2,10\n"
        "vix,full_test,0.02,0.03,1.1,1.2,1.3,10\n",
        encoding="utf-8",
    )

    # analysis.summarize loss_figure also requires universe.parquet from graph.prepare
    gdir = tmp_path / "interim" / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    univ = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "permno": [1],
            "rank": [1],
            "mcap": [100.0],
        }
    )
    univ.to_parquet(gdir / "universe.parquet", index=False)

    ensure_step_inputs_ready("analysis.summarize", "loss_figure", cfg)
