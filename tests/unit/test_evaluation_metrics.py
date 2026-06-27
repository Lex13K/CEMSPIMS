from __future__ import annotations

import numpy as np
import pytest

from mss.evaluation.metrics import aggregate_test_metric_value, mae, mse, qlike_level, rmse


def test_basic_metrics_values() -> None:
    y_true = np.array([1.0, 3.0, 2.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert mse(y_true, y_pred) == 5.0 / 3.0
    assert mae(y_true, y_pred) == 1.0
    assert rmse(y_true, y_pred) == np.sqrt(5.0 / 3.0)


def test_qlike_level_finite() -> None:
    y = np.array([0.01, 0.02, 0.03])
    f = np.array([0.011, 0.018, 0.031])
    q = qlike_level(y, f)
    assert np.isfinite(q)


def test_aggregate_test_metric_matches_mse() -> None:
    yt = np.array([0.0, 1.0, 2.0])
    yp = np.array([0.1, 1.2, 1.8])
    assert aggregate_test_metric_value("mse_log", yt, yp) == mse(yt, yp)
    assert aggregate_test_metric_value("mae_log", yt, yp) == mae(yt, yp)


def test_aggregate_qlike_matches_level_helper() -> None:
    yt = np.array([-0.5, 0.0])
    yp = np.array([-0.4, 0.1])
    ag = aggregate_test_metric_value(
        "qlike_level",
        yt,
        yp,
        loss_eps=1e-12,
        qlike_pred_transform="exp",
        qlike_level_floor=None,
    )
    ref = qlike_level(np.exp(yt), np.exp(yp))
    assert abs(ag - ref) < 1e-9


def test_aggregate_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown aggregate test metric"):
        aggregate_test_metric_value("bad", np.array([1.0]), np.array([1.0]))

