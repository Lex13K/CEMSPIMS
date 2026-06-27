"""Validation-set affine calibration in log space: ``y_true_log ≈ a + b * y_pred_raw``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

LOG_CALIBRATION_JSON = "log_calibration.json"


def collect_predictions_log_space(
    model: nn.Module,
    loader: object,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true_log, y_pred_log) arrays over the loader (e.g. val)."""
    model.eval()
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            y_pred_list.extend(out.detach().cpu().numpy().astype(float).tolist())
            y_true_list.extend(target.detach().cpu().numpy().astype(float).tolist())
    return (
        np.asarray(y_true_list, dtype=float),
        np.asarray(y_pred_list, dtype=float),
    )


def fit_affine_log_calibration(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> tuple[float, float, int]:
    """
    OLS: ``y_true_log = a + b * y_pred_log``.
    Returns ``(a, b, n_obs)``; raises if underdetermined.
    """
    y = np.asarray(y_true_log, dtype=float).reshape(-1)
    p = np.asarray(y_pred_log, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y_true_log and y_pred_log length mismatch")
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = int(y.size)
    if n < 2:
        raise ValueError("need at least 2 valid val points for log calibration")
    X = np.column_stack([np.ones(n), p])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), n


def write_log_calibration_json(
    path: Path,
    *,
    a: float,
    b: float,
    split: str,
    n_obs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "a": a,
        "b": b,
        "split": split,
        "n_obs": n_obs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_log_calibration_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
