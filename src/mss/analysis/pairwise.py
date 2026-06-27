"""Pairwise model-vs-baseline comparison outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"mean_diff": float("nan"), "dm_stat": float("nan"), "p_value_two_sided": float("nan")}
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else float("nan")
    stat = md / se if se and np.isfinite(se) and se > 0 else float("nan")
    if np.isfinite(stat):
        p = 2.0 * (1.0 - _norm_cdf(abs(stat)))
    else:
        p = float("nan")
    return {"mean_diff": md, "dm_stat": stat, "p_value_two_sided": p, "se": se}


def bootstrap_ci_loss_diff(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    reps: int,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, float]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = len(d)
    if n < 2:
        return {"ci_lower": float("nan"), "ci_upper": float("nan"), "reps": reps}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(d[idx]))
    alpha = (1.0 - ci_level) / 2.0
    return {
        "ci_lower": float(np.quantile(vals, alpha)),
        "ci_upper": float(np.quantile(vals, 1.0 - alpha)),
        "reps": reps,
    }


def run_pairwise_report(
    forecasts_path: Path,
    out_json_path: Path,
    out_formal_csv_path: Path,
    *,
    reps: int,
) -> None:
    df = pd.read_parquet(forecasts_path)
    needed = [
        ("se_log_model", "se_log_vix", "mse_log"),
        ("ae_log_model", "ae_log_vix", "mae_log"),
        ("qlike_model_t", "qlike_vix_t", "qlike"),
        ("se_level_model", "se_level_vix", "mse_level"),
        ("ae_level_model", "ae_level_vix", "mae_level"),
    ]
    rows: list[dict[str, Any]] = []
    samples: dict[str, dict[str, Any]] = {}
    for sample_name, sub in df.groupby("sample"):
        sample_out: dict[str, Any] = {"n": int(len(sub)), "results": {}}
        for col_m, col_v, label in needed:
            if col_m not in sub.columns or col_v not in sub.columns:
                continue
            dm = diebold_mariano(sub[col_m].to_numpy(), sub[col_v].to_numpy())
            ci = bootstrap_ci_loss_diff(sub[col_m].to_numpy(), sub[col_v].to_numpy(), reps=reps)
            result = {
                "mean_diff_gnn_minus_vix": dm["mean_diff"],
                "dm_stat": dm["dm_stat"],
                "dm_pvalue_two_sided": dm["p_value_two_sided"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
            }
            sample_out["results"][label] = result
            rows.append({"sample": sample_name, "metric": label, **result})
        samples[sample_name] = sample_out
    out = {
        "description": "Pairwise comparison (loss_model - loss_vix). Positive means model worse.",
        "samples": samples,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out_formal_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_formal_csv_path, index=False)

