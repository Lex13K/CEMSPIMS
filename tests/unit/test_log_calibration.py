"""Affine log calibration (numpy)."""

from __future__ import annotations

import numpy as np

from mss.model_train.log_calibration import fit_affine_log_calibration


def test_fit_affine_log_calibration_recover_ab() -> None:
    rng = np.random.default_rng(0)
    p = rng.normal(0.1, 0.05, size=100)
    a_true, b_true = 0.05, 1.1
    y = a_true + b_true * p
    a, b, n = fit_affine_log_calibration(y, p)
    assert n == 100
    assert abs(a - a_true) < 1e-10
    assert abs(b - b_true) < 1e-10
