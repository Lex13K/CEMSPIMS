"""Simple econometric table for H1-style checks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run_h1_regression_table(forecasts_path: Path, out_csv_path: Path) -> None:
    df = pd.read_parquet(forecasts_path)
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) < 20:
        pd.DataFrame(
            [{"variable": "note", "coef": np.nan, "se": np.nan, "t": np.nan, "pvalue": np.nan, "message": "insufficient_rows"}]
        ).to_csv(out_csv_path, index=False)
        return

    # First-pass H1 proxy: explain target with model and baseline forecasts.
    if "y_pred_vix" in df.columns and df["y_pred_vix"].notna().any():
        x_cols = ["y_pred_model", "y_pred_vix"]
    else:
        x_cols = ["y_pred_model"]
    reg = df[["y_true"] + x_cols].dropna().copy()
    y = reg["y_true"].to_numpy(dtype=float)
    X = reg[x_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    names = ["const"] + x_cols
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    sig2 = float(np.sum(resid**2) / max(n - k, 1))
    cov = sig2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2.0 * (1.0 - np.array([_norm_cdf(abs(float(z))) for z in t]))
    rows = []
    for i, nm in enumerate(names):
        rows.append({"variable": nm, "coef": float(beta[i]), "se": float(se[i]), "t": float(t[i]), "pvalue": float(p[i])})
    pd.DataFrame(rows).to_csv(out_csv_path, index=False)

