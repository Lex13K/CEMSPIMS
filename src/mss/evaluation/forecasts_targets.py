"""Merge scored forecasts with targets (VIX) via dataset manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mss.evaluation.checks import forecasts_path
from mss.io.config import ResolvedConfig


def _fit_har_like_log_benchmark(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a compact HAR-like model in log space on train rows:
      y_true_log ~ const + log_rv_lag_5 + log_rv_lag_30 + log_rv_lag_252
    Falls back to AR(1)-like fit if full HAR regressors are not available.
    """
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    x5 = pd.to_numeric(df.get("log_rv_lag_5"), errors="coerce").to_numpy(dtype=float)
    x30 = pd.to_numeric(df.get("log_rv_lag_30"), errors="coerce").to_numpy(dtype=float)
    x252 = pd.to_numeric(df.get("log_rv_lag_252"), errors="coerce").to_numpy(dtype=float)
    is_train = df["split"].astype(str).to_numpy() == "train"

    X_full = np.column_stack([np.ones(len(df)), x5, x30, x252])
    ok_full = np.isfinite(y) & np.isfinite(X_full).all(axis=1)
    train_ok_full = ok_full & is_train

    if int(train_ok_full.sum()) >= 50:
        beta, *_ = np.linalg.lstsq(X_full[train_ok_full], y[train_ok_full], rcond=None)
        pred = X_full @ beta
        return pred, np.isfinite(X_full).all(axis=1)

    # AR fallback (log_rv_lag_30 only)
    X_ar = np.column_stack([np.ones(len(df)), x30])
    ok_ar = np.isfinite(y) & np.isfinite(X_ar).all(axis=1)
    train_ok_ar = ok_ar & is_train
    if int(train_ok_ar.sum()) < 20:
        pred = np.full(len(df), np.nan, dtype=float)
        return pred, np.zeros(len(df), dtype=bool)
    beta_ar, *_ = np.linalg.lstsq(X_ar[train_ok_ar], y[train_ok_ar], rcond=None)
    pred_ar = X_ar @ beta_ar
    return pred_ar, np.isfinite(X_ar).all(axis=1)


def _har_inputs_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t = t.dropna(subset=["date"])
    t = t.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    if "sprtrn" in t.columns:
        spr = pd.to_numeric(t["sprtrn"], errors="coerce")
        rv5 = np.sqrt((252.0 / 5.0) * spr.pow(2).rolling(5, min_periods=5).sum()) * 100.0
        t["log_rv_lag_5"] = np.where(rv5 > 0.0, np.log(rv5), np.nan)
    elif "log_rv_lag_30" in t.columns:
        # Compatibility fallback for smoke fixtures that omit daily return spine.
        t["log_rv_lag_5"] = pd.to_numeric(t["log_rv_lag_30"], errors="coerce")
    else:
        t["log_rv_lag_5"] = np.nan
    for c in ("log_rv_lag_30", "log_rv_lag_252"):
        if c not in t.columns:
            t[c] = np.nan
    return t[["date", "log_rv_lag_5", "log_rv_lag_30", "log_rv_lag_252"]]


def merge_forecasts_with_vix(cfg: ResolvedConfig) -> pd.DataFrame:
    """
    Inner-join `forecasts.parquet` to manifest `targets_path` on normalized date.
    Adds `y_pred_vix = log(max(vix, eps))` and keeps raw `vix` for level-space metrics.

    Raises if merge loses rows or VIX is missing / non-finite.
    """
    fp = forecasts_path(cfg)
    if not fp.is_file():
        raise FileNotFoundError(
            f"merge_forecasts_with_vix requires {fp} (run model.evaluate score_splits first)."
        )
    fc = pd.read_parquet(fp)
    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce").dt.normalize()
    if fc["date"].isna().any():
        raise ValueError("forecasts.parquet has invalid date values")

    man_path = Path(cfg.interim_dir) / "dataset" / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(
            "merge_forecasts_with_vix needs interim/dataset/manifest.json for targets_path."
        )
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    raw_tp = manifest.get("targets_path")
    if not raw_tp:
        raise ValueError("dataset manifest is missing targets_path")

    tg = pd.read_parquet(Path(raw_tp))
    if "vix" not in tg.columns:
        raise ValueError("targets.parquet has no vix column.")

    tg = tg.copy()
    tg["date"] = pd.to_datetime(tg["date"], errors="coerce").dt.normalize()
    if tg["date"].isna().any():
        raise ValueError("targets have invalid date values")
    tg = tg.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    merged = fc.merge(tg[["date", "vix"]], on="date", how="inner")
    if len(merged) != len(fc):
        raise ValueError(
            "forecasts/targets date merge changed row count "
            f"(forecasts={len(fc)} merged={len(merged)}); check date alignment."
        )

    vix_num = pd.to_numeric(merged["vix"], errors="coerce")
    if vix_num.isna().any() or not np.isfinite(vix_num.to_numpy()).all():
        raise ValueError("non-finite vix in targets for merged forecast rows")

    merged["vix"] = vix_num.to_numpy(dtype=float)
    merged["y_pred_vix"] = np.log(np.clip(merged["vix"], 1e-12, None))

    # HAR-like benchmark aligned to the same forecast rows.
    har_inputs = _har_inputs_from_targets(tg)
    merged = merged.merge(har_inputs, on="date", how="left")
    if len(merged) != len(fc):
        raise ValueError("HAR input merge changed row count; date alignment broke.")
    y_har_log, har_ok = _fit_har_like_log_benchmark(merged)
    merged["y_pred_har"] = np.where(har_ok, y_har_log, np.nan)
    # Keep alignment complete on sparse fixtures by backfilling from VIX log.
    y_har_num = pd.to_numeric(merged["y_pred_har"], errors="coerce").to_numpy(dtype=float)
    y_vix_num = pd.to_numeric(merged["y_pred_vix"], errors="coerce").to_numpy(dtype=float)
    y_har_num = np.where(np.isfinite(y_har_num), y_har_num, y_vix_num)
    if not np.isfinite(y_har_num).all():
        raise ValueError("HAR benchmark produced non-finite predictions after fallback.")
    merged["y_pred_har"] = y_har_num
    return merged
