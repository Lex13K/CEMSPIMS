"""Contracts and semantic completeness for minimal model.evaluate outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mss.io.config import ResolvedConfig
from mss.model_train.loss_ids import ALLOWED_TRAINING_LOSS_IDS


FORECASTS_FILENAME = "forecasts.parquet"
FORECAST_PANEL_FILENAME = "forecast_panel.parquet"
TEST_LOSS_FILENAME = "test_loss.json"
SUMMARIES_DIR = "summaries"
SUMMARY_TABLE_FILENAME = "summary_table.csv"
HYPOTHESIS_TESTS_FILENAME = "hypothesis_tests.csv"
REGRESSION_MZ_GNN_FILENAME = "regression_mz_gnn.csv"
REGRESSION_INCREMENTAL_FILENAME = "regression_incremental.csv"
DIAGNOSTICS_SMOOTHING_FILENAME = "diagnostics_smoothing.csv"


def forecasts_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / FORECASTS_FILENAME


def test_loss_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / TEST_LOSS_FILENAME


def summary_table_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / SUMMARIES_DIR / SUMMARY_TABLE_FILENAME


def forecast_panel_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / FORECAST_PANEL_FILENAME


def hypothesis_tests_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / SUMMARIES_DIR / HYPOTHESIS_TESTS_FILENAME


def regression_mz_gnn_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / SUMMARIES_DIR / REGRESSION_MZ_GNN_FILENAME


def regression_incremental_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / SUMMARIES_DIR / REGRESSION_INCREMENTAL_FILENAME


def diagnostics_smoothing_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / SUMMARIES_DIR / DIAGNOSTICS_SMOOTHING_FILENAME


_FORECAST_PANEL_REQUIRED = (
    "date",
    "split",
    "sample",
    "y_true_log",
    "y_true_level",
    "y_pred_model_log",
    "y_pred_model_level",
    "y_pred_vix_log",
    "y_pred_vix_level",
    "vix",
    "qlike_gnn_t",
    "qlike_vix_t",
    "d_qlike_t",
    "mse_log_gnn_t",
    "mse_log_vix_t",
    "d_mse_log_t",
)


def check_forecast_panel(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        out["issues"].append(f"read failed: {e}")
        return out
    for c in _FORECAST_PANEL_REQUIRED:
        if c not in df.columns:
            out["issues"].append(f"missing column: {c}")
    if out["issues"]:
        return out
    if len(df) == 0:
        out["issues"].append("no rows")
        return out
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        out["issues"].append("invalid date values")
    if df[["date", "split"]].duplicated().any():
        out["issues"].append("duplicate (date, split)")
    for c in (
        "y_true_log",
        "y_true_level",
        "y_pred_model_log",
        "y_pred_model_level",
        "y_pred_vix_log",
        "y_pred_vix_level",
        "d_qlike_t",
        "d_mse_log_t",
    ):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")
    for c in (
        "y_pred_placebo_log",
        "y_pred_placebo_level",
        "y_pred_har_log",
        "y_pred_har_level",
        "qlike_placebo_t",
        "qlike_har_t",
        "mse_log_placebo_t",
        "mse_log_har_t",
    ):
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")
    out["passed"] = len(out["issues"]) == 0
    return out


_HYPOTHESIS_TESTS_REQUIRED = (
    "hypothesis_id",
    "inference_procedure",
    "statistic_type",
    "null_hypothesis",
    "alternative",
    "test_name",
    "sample",
    "coefficient_tested",
    "test_scope",
    "joint_hypothesis",
    "tail",
    "alternative_direction",
    "better_model",
    "loss_name",
    "loss_definition",
    "statistic",
    "p_value_primary",
    "p_value_two_sided",
    "p_value_one_sided_upper",
    "hac_max_lags",
    "n_obs",
    "rows_removed_vs_full_test",
    "effective_sample_start",
    "effective_sample_end",
    "stress_rows_removed",
    "year_rows_removed",
    "subsample_operational",
    "notes",
)


def check_hypothesis_tests(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        df = pd.read_csv(path)
    except Exception as e:
        out["issues"].append(f"read failed: {e}")
        return out
    for c in _HYPOTHESIS_TESTS_REQUIRED:
        if c not in df.columns:
            out["issues"].append(f"missing column: {c}")
    if out["issues"]:
        return out
    if len(df) == 0:
        out["issues"].append("no rows")
        return out
    out["passed"] = True
    return out


def check_forecasts(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        out["issues"].append(f"read failed: {e}")
        return out

    required = ("date", "split", "sample", "y_true", "y_pred_model")
    for c in required:
        if c not in df.columns:
            out["issues"].append(f"missing column: {c}")
    if out["issues"]:
        return out
    if len(df) == 0:
        out["issues"].append("no rows")
        return out

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        out["issues"].append("invalid date values")
    if df[["date", "split"]].duplicated().any():
        out["issues"].append("duplicate (date, split)")

    numeric_cols = ["y_true", "y_pred_model"]
    if "y_pred_placebo" in df.columns:
        numeric_cols.append("y_pred_placebo")
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")

    out["passed"] = len(out["issues"]) == 0
    return out


def check_test_loss(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        out["issues"].append(f"invalid JSON: {e}")
        return out

    if payload.get("status") != "completed":
        out["issues"].append("status is not 'completed'")
    mid = payload.get("metric")
    mid_norm = mid.strip().lower() if isinstance(mid, str) else ""
    if mid_norm not in ALLOWED_TRAINING_LOSS_IDS:
        out["issues"].append(f"metric must be one of {sorted(ALLOWED_TRAINING_LOSS_IDS)}")
    if payload.get("split") != "test":
        out["issues"].append("split is not 'test'")
    n_rows = payload.get("n_rows")
    value = payload.get("value")
    try:
        if int(n_rows) <= 0:
            out["issues"].append("n_rows must be > 0")
    except Exception:
        out["issues"].append("n_rows missing or invalid")
    try:
        v = float(value)
        if not np.isfinite(v):
            out["issues"].append("value must be finite")
    except Exception:
        out["issues"].append("value missing or invalid")
    out["passed"] = len(out["issues"]) == 0
    return out


def check_summary_table(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        df = pd.read_csv(path)
    except Exception as e:
        out["issues"].append(f"read failed: {e}")
        return out
    required = ("model", "sample", "mse_log", "mae_log", "mse", "mae", "qlike", "n_samples")
    for c in required:
        if c not in df.columns:
            out["issues"].append(f"missing column: {c}")
    if out["issues"]:
        return out
    if len(df) == 0:
        out["issues"].append("no rows")
        return out
    for c in ("mse_log", "mae_log", "mse", "mae", "qlike"):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")
    n_rows = pd.to_numeric(df["n_samples"], errors="coerce")
    if n_rows.isna().any() or (n_rows <= 0).any():
        out["issues"].append("n_samples must be positive integers")
    out["passed"] = len(out["issues"]) == 0
    return out


def check_diagnostics_smoothing(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing file: {path}")
        return out
    try:
        df = pd.read_csv(path)
    except Exception as e:
        out["issues"].append(f"read failed: {e}")
        return out
    required = (
        "sample",
        "diagnostic",
        "model",
        "value",
    )
    for c in required:
        if c not in df.columns:
            out["issues"].append(f"missing column: {c}")
    if len(df) == 0:
        out["issues"].append("no rows")
    vals = pd.to_numeric(df.get("value"), errors="coerce")
    if vals.isna().any() or (~np.isfinite(vals)).any():
        out["issues"].append("non-finite values in value")
    out["passed"] = len(out["issues"]) == 0
    return out


def model_evaluate_score_splits_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Base forecasts present."""
    return bool(check_forecasts(forecasts_path(cfg)).get("passed"))


def model_evaluate_write_forecast_panel_semantically_complete(cfg: ResolvedConfig) -> bool:
    if not model_evaluate_score_splits_semantically_complete(cfg):
        return False
    return bool(check_forecast_panel(forecast_panel_path(cfg)).get("passed"))


def model_evaluate_aggregate_test_loss_semantically_complete(cfg: ResolvedConfig) -> bool:
    if not model_evaluate_score_splits_semantically_complete(cfg):
        return False
    return bool(check_test_loss(test_loss_path(cfg)).get("passed"))


def model_evaluate_write_summary_table_semantically_complete(cfg: ResolvedConfig) -> bool:
    if not model_evaluate_write_forecast_panel_semantically_complete(cfg):
        return False
    if not model_evaluate_aggregate_test_loss_semantically_complete(cfg):
        return False
    if not bool(check_summary_table(summary_table_path(cfg)).get("passed")):
        return False
    dp = diagnostics_smoothing_path(cfg)
    if not dp.exists():
        return True
    return bool(check_diagnostics_smoothing(dp).get("passed"))


def model_evaluate_run_hypothesis_tests_semantically_complete(cfg: ResolvedConfig) -> bool:
    if not model_evaluate_write_forecast_panel_semantically_complete(cfg):
        return False
    return bool(check_hypothesis_tests(hypothesis_tests_path(cfg)).get("passed"))


def model_evaluate_pipeline_semantically_complete(cfg: ResolvedConfig) -> bool:
    """All model.evaluate steps complete."""
    if not model_evaluate_write_summary_table_semantically_complete(cfg):
        return False
    return model_evaluate_run_hypothesis_tests_semantically_complete(cfg)


def model_evaluate_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Alias for `model_evaluate_pipeline_semantically_complete` (same predicate)."""
    return model_evaluate_pipeline_semantically_complete(cfg)

