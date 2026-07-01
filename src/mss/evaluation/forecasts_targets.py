"""Merge scored forecasts with targets (VIX) via dataset manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mss.evaluation.checks import forecasts_path
from mss.evaluation.config import load_model_evaluate_config
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

    X_ar = np.column_stack([np.ones(len(df)), x30])
    ok_ar = np.isfinite(y) & np.isfinite(X_ar).all(axis=1)
    train_ok_ar = ok_ar & is_train
    if int(train_ok_ar.sum()) < 20:
        pred = np.full(len(df), np.nan, dtype=float)
        return pred, np.zeros(len(df), dtype=bool)
    beta_ar, *_ = np.linalg.lstsq(X_ar[train_ok_ar], y[train_ok_ar], rcond=None)
    pred_ar = X_ar @ beta_ar
    return pred_ar, np.isfinite(X_ar).all(axis=1)


def _fit_affine_vix_log_calibration(
    y_true_log: np.ndarray,
    y_pred_vix_log: np.ndarray,
    is_train: np.ndarray,
) -> tuple[float, float]:
    ok = is_train & np.isfinite(y_true_log) & np.isfinite(y_pred_vix_log)
    if int(ok.sum()) < 5:
        return 0.0, 1.0
    X = np.column_stack([np.ones(int(ok.sum())), y_pred_vix_log[ok]])
    beta, *_ = np.linalg.lstsq(X, y_true_log[ok], rcond=None)
    return float(beta[0]), float(beta[1])


def _fit_vix_har_log_benchmark(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid implied-plus-historical volatility benchmark, fit train-only in log space:
      y_true_log ~ const + log_vix + log_rv_lag_5 + log_rv_lag_30 (+ log_rv_lag_252 if available)
    where ``log_vix`` is the raw spot VIX in logs (the same series as ``y_pred_vix``). This combines
    the market-implied signal with recent realized-vol history; it is not a standalone HAR model.
    Falls back to dropping the 252-day lag, then to a VIX-only affine fit, if data is scarce.
    """
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    lv = pd.to_numeric(df.get("y_pred_vix"), errors="coerce").to_numpy(dtype=float)
    x5 = pd.to_numeric(df.get("log_rv_lag_5"), errors="coerce").to_numpy(dtype=float)
    x30 = pd.to_numeric(df.get("log_rv_lag_30"), errors="coerce").to_numpy(dtype=float)
    x252 = pd.to_numeric(df.get("log_rv_lag_252"), errors="coerce").to_numpy(dtype=float)
    is_train = df["split"].astype(str).to_numpy() == "train"

    for cols, min_train in (
        ([np.ones(len(df)), lv, x5, x30, x252], 50),
        ([np.ones(len(df)), lv, x5, x30], 20),
        ([np.ones(len(df)), lv], 5),
    ):
        X = np.column_stack(cols)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if int((ok & is_train).sum()) >= min_train:
            beta, *_ = np.linalg.lstsq(X[ok & is_train], y[ok & is_train], rcond=None)
            return X @ beta, np.isfinite(X).all(axis=1)

    return np.full(len(df), np.nan, dtype=float), np.zeros(len(df), dtype=bool)


def _har_lag_columns_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t = t.dropna(subset=["date"])
    t = t.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    for c in ("log_rv_lag_5", "log_rv_lag_30", "log_rv_lag_252"):
        if c not in t.columns:
            t[c] = np.nan
    return t[["date", "log_rv_lag_5", "log_rv_lag_30", "log_rv_lag_252"]]


def merge_forecasts_with_vix(cfg: ResolvedConfig) -> pd.DataFrame:
    """
    Inner-join `forecasts.parquet` to manifest `targets_path` on normalized date and attach the
    v2 benchmark suite (all in log space, plus raw ``vix`` level for QLIKE).

    Benchmarks:
      - ``y_pred_vix`` = log(max(vix, eps)) -- raw spot VIX, the primary market-implied benchmark.
      - ``y_pred_calibrated_vix`` -- train-calibrated (bias-adjusted) VIX: affine fit
        ``y_true_log ~ a + b * log_vix`` on train rows, applied unchanged to val/test. This corrects
        a level/scale bias; it does not remove the volatility risk premium.
      - ``y_pred_vix_har`` -- hybrid implied-plus-historical volatility benchmark (log VIX plus
        lagged realized-vol terms), fit train-only.
      - ``y_pred_har`` -- standalone HAR-like benchmark from ``log_rv_lag_*`` only.
    """
    fp = forecasts_path(cfg)
    if not fp.is_file():
        raise FileNotFoundError(
            f"merge_forecasts_with_vix requires {fp} (run model.evaluate score_splits first)."
        )
    ecfg = load_model_evaluate_config(cfg.source_config_path)
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
    # raw_vix: primary market-implied benchmark (always raw spot VIX in logs).
    merged["y_pred_vix"] = np.log(np.clip(merged["vix"], 1e-12, None))

    y_true = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float)
    y_vix_log = pd.to_numeric(merged["y_pred_vix"], errors="coerce").to_numpy(dtype=float)
    is_train = merged["split"].astype(str).to_numpy() == "train"

    # calibrated_vix: train-calibrated (bias-adjusted) VIX, a separate benchmark column. The legacy
    # fit_vix_log_calibration flag additionally folds the affine map into the raw_vix column for
    # backward compatibility; raw_vix otherwise stays raw.
    a, b = _fit_affine_vix_log_calibration(y_true, y_vix_log, is_train)
    merged["y_pred_calibrated_vix"] = a + b * y_vix_log
    if ecfg.fit_vix_log_calibration:
        merged["y_pred_vix"] = a + b * y_vix_log

    har_inputs = _har_lag_columns_from_targets(tg)
    merged = merged.merge(har_inputs, on="date", how="left")
    if len(merged) != len(fc):
        raise ValueError("HAR input merge changed row count; date alignment broke.")

    # vix_har: hybrid implied-plus-historical benchmark (log VIX + realized-vol lags).
    y_vix_har_log, vix_har_ok = _fit_vix_har_log_benchmark(merged)
    merged["y_pred_vix_har"] = np.where(vix_har_ok, y_vix_har_log, np.nan)

    # har: standalone HAR-like benchmark from realized-vol lags only.
    y_har_log, har_ok = _fit_har_like_log_benchmark(merged)
    merged["y_pred_har"] = np.where(har_ok, y_har_log, np.nan)
    har_finite = int(np.isfinite(merged["y_pred_har"].to_numpy(dtype=float)).sum())
    if har_finite == 0:
        missing = [c for c in ("log_rv_lag_5", "log_rv_lag_30", "log_rv_lag_252") if c not in tg.columns]
        hint = (
            "Refresh shared targets (Phase 5 HAR lags): "
            "python scripts/run.py run --run default --pipeline data.prepare --overwrite data.prepare"
        )
        if missing:
            raise ValueError(
                f"HAR benchmark unavailable: targets.parquet missing columns {missing}. {hint}"
            )
        raise ValueError(
            "HAR benchmark unavailable: log_rv_lag_5/30/252 are all null or non-finite. "
            f"Likely stale shared targets.parquet. {hint}"
        )
    return merged
