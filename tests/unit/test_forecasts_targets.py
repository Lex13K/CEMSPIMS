"""Unit tests for VIX/HAR benchmark merge in forecasts_targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mss.evaluation.forecasts_targets import merge_forecasts_with_vix
from tests.conftest import make_resolved_config


def _write_merge_fixture(tmp: Path) -> None:
    proc = tmp / "processed" / "scoring"
    proc.mkdir(parents=True)
    interim = tmp / "interim"
    ds_dir = interim / "dataset"
    ds_dir.mkdir(parents=True)
    shared = tmp / "shared" / "interim"
    shared.mkdir(parents=True)

    dates = pd.bdate_range("2020-01-02", periods=80)
    fc = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 60 + ["test"] * 20,
            "sample": ["train_full"] * 60 + ["test_full"] * 20,
            "y_true": np.linspace(2.5, 3.0, 80),
            "y_pred_model": np.linspace(2.4, 2.9, 80),
            "y_pred_placebo": np.linspace(2.45, 2.95, 80),
        }
    )
    fc.to_parquet(proc / "forecasts.parquet", index=False)

    tg = pd.DataFrame(
        {
            "date": dates,
            "vix": np.full(80, 20.0),
            "log_rv_lag_5": np.linspace(2.3, 2.8, 80),
            "log_rv_lag_30": np.linspace(2.35, 2.85, 80),
            "log_rv_lag_252": np.linspace(2.4, 2.9, 80),
        }
    )
    targets_path = shared / "targets.parquet"
    tg.to_parquet(targets_path, index=False)

    manifest = {"targets_path": str(targets_path.resolve())}
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_merge_har_from_targets_no_vix_backfill(tmp_path: Path) -> None:
    _write_merge_fixture(tmp_path)
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("[model.evaluate]\nfit_vix_log_calibration = false\n", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())

    merged = merge_forecasts_with_vix(cfg)
    assert len(merged) == 80
    har = pd.to_numeric(merged["y_pred_har"], errors="coerce")
    assert har.notna().all()
    assert not np.allclose(har.to_numpy(), merged["y_pred_vix"].to_numpy())


def test_merge_emits_v2_benchmark_columns(tmp_path: Path) -> None:
    _write_merge_fixture(tmp_path)
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("[model.evaluate]\nfit_vix_log_calibration = false\n", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())

    merged = merge_forecasts_with_vix(cfg)
    # raw_vix stays raw log spot VIX; calibrated_vix and vix_har are separate benchmark columns.
    assert "y_pred_calibrated_vix" in merged.columns
    assert "y_pred_vix_har" in merged.columns
    raw = merged["y_pred_vix"].to_numpy()
    assert np.allclose(raw, np.log(20.0))  # raw stays raw when calibration off
    cal = pd.to_numeric(merged["y_pred_calibrated_vix"], errors="coerce")
    assert cal.notna().all()
    vh = pd.to_numeric(merged["y_pred_vix_har"], errors="coerce")
    assert vh.notna().all()


def test_calibration_flag_folds_into_raw_vix_only(tmp_path: Path) -> None:
    _write_merge_fixture(tmp_path)
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("[model.evaluate]\nfit_vix_log_calibration = true\n", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())

    merged = merge_forecasts_with_vix(cfg)
    raw = merged["y_pred_vix"].to_numpy()
    # With calibration on (legacy), raw_vix is the affine map, no longer exactly log(20).
    assert not np.allclose(raw, np.log(20.0))
    # calibrated_vix equals the folded raw_vix in this legacy mode.
    assert np.allclose(raw, merged["y_pred_calibrated_vix"].to_numpy())


def test_merge_har_nan_when_lags_missing(tmp_path: Path) -> None:
    proc = tmp_path / "processed" / "scoring"
    proc.mkdir(parents=True)
    interim = tmp_path / "interim"
    ds_dir = interim / "dataset"
    ds_dir.mkdir(parents=True)
    shared = tmp_path / "shared" / "interim"
    shared.mkdir(parents=True)

    dates = pd.bdate_range("2020-01-02", periods=30)
    fc = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 25 + ["test"] * 5,
            "sample": ["train_full"] * 25 + ["test_full"] * 5,
            "y_true": np.linspace(2.5, 3.0, 30),
            "y_pred_model": np.linspace(2.4, 2.9, 30),
            "y_pred_placebo": np.linspace(2.45, 2.95, 30),
        }
    )
    fc.to_parquet(proc / "forecasts.parquet", index=False)

    tg = pd.DataFrame({"date": dates, "vix": np.full(30, 20.0)})
    targets_path = shared / "targets.parquet"
    tg.to_parquet(targets_path, index=False)
    (ds_dir / "manifest.json").write_text(
        json.dumps({"targets_path": str(targets_path.resolve())}),
        encoding="utf-8",
    )

    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())
    with pytest.raises(ValueError, match="missing columns"):
        merge_forecasts_with_vix(cfg)
