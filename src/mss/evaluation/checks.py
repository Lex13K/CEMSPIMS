"""Contracts and semantic completeness for minimal model.evaluate outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mss.io.config import ResolvedConfig
from mss.model_train.loss_ids import ALLOWED_TRAINING_LOSS_IDS
from mss.processed.paths import (
    DIAGNOSTICS_SMOOTHING_FILENAME,
    FORECAST_PANEL_FILENAME,
    FORECASTS_FILENAME,
    HYPOTHESIS_TESTS_FILENAME,
    REGRESSION_INCREMENTAL_FILENAME,
    REGRESSION_MZ_GNN_FILENAME,
    SUMMARY_TABLE_FILENAME,
    TEST_LOSS_FILENAME,
    diagnostics_smoothing_path,
    forecast_panel_path,
    forecasts_path,
    hypothesis_tests_path,
    regression_incremental_path,
    regression_mz_gnn_path,
    summary_table_path,
    test_loss_path,
)

__all__ = [
    "FORECASTS_FILENAME",
    "FORECAST_PANEL_FILENAME",
    "TEST_LOSS_FILENAME",
    "SUMMARY_TABLE_FILENAME",
    "HYPOTHESIS_TESTS_FILENAME",
    "REGRESSION_MZ_GNN_FILENAME",
    "REGRESSION_INCREMENTAL_FILENAME",
    "DIAGNOSTICS_SMOOTHING_FILENAME",
    "forecasts_path",
    "test_loss_path",
    "summary_table_path",
    "forecast_panel_path",
    "hypothesis_tests_path",
    "regression_mz_gnn_path",
    "regression_incremental_path",
    "diagnostics_smoothing_path",
    "check_forecast_panel",
    "check_hypothesis_tests",
    "check_forecasts",
    "check_test_loss",
    "check_summary_table",
    "check_diagnostics_smoothing",
    "model_evaluate_score_splits_semantically_complete",
    "model_evaluate_write_forecast_panel_semantically_complete",
    "model_evaluate_aggregate_test_loss_semantically_complete",
    "model_evaluate_write_summary_table_semantically_complete",
    "model_evaluate_run_hypothesis_tests_semantically_complete",
    "model_evaluate_pipeline_semantically_complete",
    "model_evaluate_step_semantically_complete",
]

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
    "d_qlike_placebo_t",
    "mse_log_gnn_t",
    "mse_log_vix_t",
    "d_mse_log_t",
    "d_mse_log_placebo_t",
)


_FORECAST_PANEL_HAR_OPTIONAL = (
    "y_pred_har_log",
    "y_pred_har_level",
    "qlike_har_t",
    "mse_log_har_t",
)

_FORECAST_PANEL_BENCHMARK_DIFF_OPTIONAL = (
    "d_qlike_calibrated_vix_t",
    "d_mse_log_calibrated_vix_t",
    "d_qlike_vix_har_t",
    "d_mse_log_vix_har_t",
    "d_qlike_har_t",
    "d_mse_log_har_t",
)

# v2 benchmarks: like HAR, fit/regression-based predictions may be NaN on early train rows lacking
# lag history, but must be finite on val/test where formal inference runs.
_FORECAST_PANEL_BENCHMARK_OPTIONAL = (
    "y_pred_calibrated_vix_log",
    "y_pred_calibrated_vix_level",
    "qlike_calibrated_vix_t",
    "mse_log_calibrated_vix_t",
    "y_pred_vix_har_log",
    "y_pred_vix_har_level",
    "qlike_vix_har_t",
    "mse_log_vix_har_t",
)


def _check_eval_finite_columns(
    df: pd.DataFrame, candidate_cols: tuple[str, ...], *, sentinel: str
) -> list[str]:
    """Benchmark preds may be NaN on early train rows but must be finite on val/test."""
    issues: list[str] = []
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        return issues
    if sentinel in df.columns:
        all_vals = pd.to_numeric(df[sentinel], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(all_vals).any():
            issues.append(f"no finite {sentinel} values")
    eval_mask = df["split"].astype(str).isin(("val", "test")).to_numpy()
    if not eval_mask.any():
        return issues
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        ev = vals[eval_mask]
        if np.isnan(ev).any() or not np.isfinite(ev).all():
            issues.append(f"non-finite values in {c} on val/test splits")
    return issues


def _check_har_benchmark_columns(df: pd.DataFrame) -> list[str]:
    """HAR preds need 252-day lag history; early train rows may legitimately be NaN."""
    return _check_eval_finite_columns(
        df, _FORECAST_PANEL_HAR_OPTIONAL, sentinel="y_pred_har_log"
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
        "d_qlike_placebo_t",
        "d_mse_log_placebo_t",
    ):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")
    for c in (
        "y_pred_placebo_log",
        "y_pred_placebo_level",
        "qlike_placebo_t",
        "mse_log_placebo_t",
    ):
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.isfinite(vals).all():
            out["issues"].append(f"non-finite values in {c}")
    out["issues"].extend(_check_har_benchmark_columns(df))
    out["issues"].extend(
        _check_eval_finite_columns(
            df, _FORECAST_PANEL_BENCHMARK_OPTIONAL, sentinel="y_pred_vix_har_log"
        )
    )
    out["issues"].extend(
        _check_eval_finite_columns(
            df, _FORECAST_PANEL_BENCHMARK_DIFF_OPTIONAL, sentinel="d_qlike_vix_har_t"
        )
    )
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
    "benchmark",
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
