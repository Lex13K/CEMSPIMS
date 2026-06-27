"""Map model outputs (log-space) to level variance `f` for QLIKE; shared numpy + transform ids."""

from __future__ import annotations

import numpy as np

ALLOWED_QLIKE_PRED_TRANSFORMS: frozenset[str] = frozenset({"exp", "softplus"})


def validate_qlike_pred_transform(name: str) -> str:
    n = name.strip().lower()
    if n not in ALLOWED_QLIKE_PRED_TRANSFORMS:
        raise ValueError(
            f"Unknown [model.train].qlike_pred_transform: {name!r}; "
            f"expected one of {sorted(ALLOWED_QLIKE_PRED_TRANSFORMS)}"
        )
    return n


def softplus_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus; aligned with ``torch.nn.functional.softplus``."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def level_variance_from_pred_log_numpy(
    pred_log: np.ndarray,
    *,
    transform: str,
    level_floor: float,
) -> np.ndarray:
    """Level-space forecast variance f_t from scalar model output (same tensor as MSE in log space)."""
    t = validate_qlike_pred_transform(transform)
    p = np.asarray(pred_log, dtype=float)
    if t == "exp":
        return np.clip(np.exp(p), level_floor, None)
    return np.maximum(softplus_np(p) + level_floor, level_floor)
