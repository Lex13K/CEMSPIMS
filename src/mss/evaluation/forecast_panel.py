"""Build processed/forecast_panel.parquet: VIX-merged date-level panel for formal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mss.evaluation.checks import forecasts_path
from mss.evaluation.forecasts_targets import merge_forecasts_with_vix
from mss.evaluation.metrics import qlike_per_date
from mss.io.config import ResolvedConfig
from mss.model_train.config import load_model_train_config
from mss.model_train.qlike_mapping import level_variance_from_pred_log_numpy


def build_forecast_panel_dataframe(cfg: ResolvedConfig) -> pd.DataFrame:
    """Merge forecasts with VIX and add explicit log/level columns and per-date losses."""
    if not forecasts_path(cfg).is_file():
        raise FileNotFoundError("forecast_panel requires processed/forecasts.parquet (score_splits).")
    merged = merge_forecasts_with_vix(cfg)
    mtrain = load_model_train_config(cfg.source_config_path)
    eps = float(mtrain.loss_eps)
    lvl_floor = float(eps if mtrain.qlike_level_floor is None else mtrain.qlike_level_floor)

    y_true_log = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred_model_log = pd.to_numeric(merged["y_pred_model"], errors="coerce").to_numpy(dtype=float)
    y_pred_placebo_log = pd.to_numeric(merged["y_pred_placebo"], errors="coerce").to_numpy(dtype=float)
    y_pred_har_log = pd.to_numeric(merged["y_pred_har"], errors="coerce").to_numpy(dtype=float)
    y_pred_vix_log = pd.to_numeric(merged["y_pred_vix"], errors="coerce").to_numpy(dtype=float)
    y_pred_vix_level = pd.to_numeric(merged["vix"], errors="coerce").to_numpy(dtype=float)

    y_true_level = np.exp(y_true_log)
    y_pred_model_level = level_variance_from_pred_log_numpy(
        y_pred_model_log,
        transform=mtrain.qlike_pred_transform,
        level_floor=lvl_floor,
    )
    y_pred_placebo_level = level_variance_from_pred_log_numpy(
        y_pred_placebo_log,
        transform=mtrain.qlike_pred_transform,
        level_floor=lvl_floor,
    )
    y_pred_har_level = np.exp(y_pred_har_log)

    qlike_gnn_t = qlike_per_date(y_true_level, y_pred_model_level, eps=eps)
    qlike_placebo_t = qlike_per_date(y_true_level, y_pred_placebo_level, eps=eps)
    qlike_har_t = qlike_per_date(y_true_level, y_pred_har_level, eps=eps)
    qlike_vix_t = qlike_per_date(y_true_level, y_pred_vix_level, eps=eps)
    d_qlike_t = qlike_vix_t - qlike_gnn_t

    mse_log_gnn_t = np.square(y_true_log - y_pred_model_log)
    mse_log_placebo_t = np.square(y_true_log - y_pred_placebo_log)
    mse_log_har_t = np.square(y_true_log - y_pred_har_log)
    mse_log_vix_t = np.square(y_true_log - y_pred_vix_log)
    d_mse_log_t = mse_log_vix_t - mse_log_gnn_t

    out = merged.drop(
        columns=["y_true", "y_pred_model", "y_pred_placebo", "y_pred_har", "y_pred_vix"],
        errors="ignore",
    ).copy()
    out["y_true_log"] = y_true_log
    out["y_true_level"] = y_true_level
    out["y_pred_model_log"] = y_pred_model_log
    out["y_pred_model_level"] = y_pred_model_level
    out["y_pred_placebo_log"] = y_pred_placebo_log
    out["y_pred_placebo_level"] = y_pred_placebo_level
    out["y_pred_har_log"] = y_pred_har_log
    out["y_pred_har_level"] = y_pred_har_level
    out["y_pred_vix_log"] = y_pred_vix_log
    out["y_pred_vix_level"] = y_pred_vix_level
    out["qlike_gnn_t"] = qlike_gnn_t
    out["qlike_placebo_t"] = qlike_placebo_t
    out["qlike_har_t"] = qlike_har_t
    out["qlike_vix_t"] = qlike_vix_t
    out["d_qlike_t"] = d_qlike_t
    out["mse_log_gnn_t"] = mse_log_gnn_t
    out["mse_log_placebo_t"] = mse_log_placebo_t
    out["mse_log_har_t"] = mse_log_har_t
    out["mse_log_vix_t"] = mse_log_vix_t
    out["d_mse_log_t"] = d_mse_log_t
    return out


def run_write_forecast_panel(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    from mss.evaluation.checks import check_forecast_panel, forecast_panel_path

    out = forecast_panel_path(cfg)
    if not overwrite and check_forecast_panel(out).get("passed"):
        return
    df = build_forecast_panel_dataframe(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    chk = check_forecast_panel(out)
    if not chk.get("passed"):
        raise RuntimeError(f"forecast_panel contract failed: {chk.get('issues')}")
