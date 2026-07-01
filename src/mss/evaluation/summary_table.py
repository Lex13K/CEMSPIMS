"""Build processed/summaries/summary_table.csv (GNN vs VIX, multiple sample slices)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mss.evaluation.config import ModelEvaluateConfig
from mss.evaluation.metrics import mae, mse, qlike_level


SUMMARY_SAMPLE_ORDER = ("train", "val", "excl_crisis", "full_test", "excl_2020")


def _row_for_model_slice(
    sub: pd.DataFrame,
    *,
    model: str,
    sample: str,
) -> dict[str, Any] | None:
    if len(sub) == 0:
        return None
    yt = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(dtype=float)
    # Benchmark columns whose level is exp(log prediction). raw_vix uses the spot VIX level.
    _LOG_MODEL_COLS = {
        "gnn": "y_pred_model",
        "placebo": "y_pred_placebo",
        "har": "y_pred_har",
        "calibrated_vix": "y_pred_calibrated_vix",
        "vix_har": "y_pred_vix_har",
    }
    if model in _LOG_MODEL_COLS:
        col = _LOG_MODEL_COLS[model]
        if col not in sub.columns:
            return None
        y_log_pred = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        pred_level = np.exp(y_log_pred)
    else:
        y_log_pred = pd.to_numeric(sub["y_pred_vix"], errors="coerce").to_numpy(dtype=float)
        pred_level = pd.to_numeric(sub["vix"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(yt) & np.isfinite(y_log_pred) & np.isfinite(pred_level)
    if int(ok.sum()) == 0:
        return None
    yt = yt[ok]
    y_log_pred = y_log_pred[ok]
    pred_level = pred_level[ok]
    rv = np.exp(yt)
    n = int(len(yt))
    return {
        "model": model,
        "sample": sample,
        "mse_log": mse(yt, y_log_pred),
        "mae_log": mae(yt, y_log_pred),
        "mse": mse(rv, pred_level),
        "mae": mae(rv, pred_level),
        "qlike": qlike_level(rv, pred_level),
        "n_samples": n,
    }


def build_summary_table_rows(df: pd.DataFrame, ecfg: ModelEvaluateConfig) -> list[dict[str, Any]]:
    """One row per (gnn|vix, sample slice); skips empty slices."""
    out: list[dict[str, Any]] = []
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    sp = d["split"].astype(str)
    years = d["date"].dt.year

    test_excl = set(ecfg.summary_test_exclusion_years)
    crisis_start = pd.Timestamp(ecfg.summary_train_crisis_excl_start).normalize()
    crisis_end = pd.Timestamp(ecfg.summary_train_crisis_excl_end).normalize()
    dates_norm = d["date"].dt.normalize()
    in_crisis_window = (dates_norm >= crisis_start) & (dates_norm <= crisis_end)

    slice_defs: list[tuple[str, pd.Series]] = [
        ("train", sp == "train"),
        ("val", sp == "val"),
        (
            "excl_crisis",
            (sp == "train") & ~in_crisis_window,
        ),
        ("full_test", sp == "test"),
        (
            "excl_2020",
            (sp == "test") & ~years.isin(test_excl),
        ),
    ]

    models = ("gnn", "placebo", "har", "vix")
    if "y_pred_calibrated_vix" in d.columns:
        models = models + ("calibrated_vix",)
    if "y_pred_vix_har" in d.columns:
        models = models + ("vix_har",)

    for sample_name, mask in slice_defs:
        sub = d.loc[mask]
        for model in models:
            row = _row_for_model_slice(sub, model=model, sample=sample_name)
            if row is not None:
                out.append(row)

    return out
