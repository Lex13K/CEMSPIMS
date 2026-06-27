"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mss.model_train.qlike_mapping import level_variance_from_pred_log_numpy

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.square(d)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(d)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def smooth_l1_log(
    y_true: np.ndarray, y_pred: np.ndarray, *, beta: float = 1.0
) -> float:
    """Smooth L1 (Huber) loss on log-space residuals; matches `nn.SmoothL1Loss(beta=beta)` mean reduction."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_d = np.abs(d)
    return float(
        np.mean(np.where(abs_d < beta, 0.5 * (d * d) / beta, abs_d - 0.5 * beta))
    )


def qlike_level_from_log_preds(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    loss_eps: float,
    qlike_pred_transform: str,
    qlike_level_floor: float | None,
) -> float:
    """Mean QLIKE using level y and mapped level f from pred (matches training loss mapping)."""
    floor = float(loss_eps if qlike_level_floor is None else qlike_level_floor)
    y = np.clip(np.exp(np.asarray(y_true_log, dtype=float)), loss_eps, None)
    f = level_variance_from_pred_log_numpy(
        np.asarray(y_pred_log, dtype=float),
        transform=qlike_pred_transform,
        level_floor=floor,
    )
    f = np.clip(f, loss_eps, None)
    return float(np.mean(np.log(f) + (y / f)))


def aggregate_test_metric_value(
    metric: str,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    loss_eps: float = 1e-12,
    smooth_l1_beta: float = 1.0,
    loss_alpha: float = 0.5,
    qlike_pred_transform: str = "exp",
    qlike_level_floor: float | None = None,
) -> float:
    """Scalar test loss aligned with `mss.model_train.losses` ids; inputs are log-space targets/preds."""
    m = metric.strip().lower()
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    if m == "mse_log":
        return mse(yt, yp)
    if m == "mae_log":
        return mae(yt, yp)
    if m == "smooth_l1_log":
        return smooth_l1_log(yt, yp, beta=smooth_l1_beta)
    if m == "qlike_level":
        return qlike_level_from_log_preds(
            yt,
            yp,
            loss_eps=loss_eps,
            qlike_pred_transform=qlike_pred_transform,
            qlike_level_floor=qlike_level_floor,
        )
    if m == "hybrid_mse_qlike":
        if not 0.0 < loss_alpha < 1.0:
            raise ValueError("hybrid_mse_qlike metric requires 0 < loss_alpha < 1")
        mse_v = mse(yt, yp)
        q_v = qlike_level_from_log_preds(
            yt,
            yp,
            loss_eps=loss_eps,
            qlike_pred_transform=qlike_pred_transform,
            qlike_level_floor=qlike_level_floor,
        )
        return float(loss_alpha * mse_v + (1.0 - loss_alpha) * q_v)
    raise ValueError(f"Unknown aggregate test metric id: {metric!r}")


def qlike_level(y_true_level: np.ndarray, y_pred_level: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    QLIKE on variance/volatility-in-level space:
      mean(log(f_t) + y_t / f_t)
    """
    y = np.clip(np.asarray(y_true_level, dtype=float), eps, None)
    f = np.clip(np.asarray(y_pred_level, dtype=float), eps, None)
    return float(np.mean(np.log(f) + (y / f)))


def qlike_per_date(y_true_level: np.ndarray, y_pred_level: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    y = np.clip(np.asarray(y_true_level, dtype=float), eps, None)
    f = np.clip(np.asarray(y_pred_level, dtype=float), eps, None)
    return np.log(f) + (y / f)


def add_error_columns(df: pd.DataFrame, *, model_col: str, y_true_col: str = "y_true") -> pd.DataFrame:
    out = df.copy()
    pred = pd.to_numeric(out[model_col], errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(out[y_true_col], errors="coerce").to_numpy(dtype=float)
    suffix = "model" if model_col == "y_pred_model" else model_col.replace("y_pred_", "")
    err = true - pred
    out[f"se_log_{suffix}"] = np.square(err)
    out[f"ae_log_{suffix}"] = np.abs(err)
    true_lvl = np.exp(true)
    pred_lvl = np.exp(pred)
    out[f"se_level_{suffix}"] = np.square(true_lvl - pred_lvl)
    out[f"ae_level_{suffix}"] = np.abs(true_lvl - pred_lvl)
    out[f"qlike_{suffix}_t"] = qlike_per_date(true_lvl, pred_lvl)
    return out

