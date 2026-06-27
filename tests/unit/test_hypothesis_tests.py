"""Unit tests for formal hypothesis rows (schema + H1/H3/H2 p-value semantics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mss.evaluation.config import ModelEvaluateConfig
from mss.evaluation.hypothesis_tests import (
    PROC_AUDIT,
    PROC_DM,
    PROC_MZ,
    build_formal_subsample_specs,
    run_hypothesis_tests,
)
from mss.evaluation.metrics import qlike_per_date


def _panel_n_test_dates(*, n: int = 12) -> pd.DataFrame:
    """Synthetic test split with coherent log/level and loss differentials."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", periods=n)
    yt = rng.normal(0.12, 0.01, size=n)
    pm = yt + rng.normal(0.0, 0.02, size=n)
    vix_level = np.full(n, 25.0)
    pv = np.log(np.maximum(vix_level, 1e-12))
    y_level = np.exp(yt)
    pml = np.exp(pm)
    qg = qlike_per_date(y_level, pml, eps=1e-12)
    qv = qlike_per_date(y_level, vix_level, eps=1e-12)
    d_qlike = qv - qg
    mse_g = (yt - pm) ** 2
    mse_v = (yt - pv) ** 2
    d_mse = mse_v - mse_g
    return pd.DataFrame(
        {
            "date": dates,
            "split": ["test"] * n,
            "sample": ["test_full"] * n,
            "y_true_log": yt,
            "y_true_level": y_level,
            "y_pred_model_log": pm,
            "y_pred_model_level": pml,
            "y_pred_vix_log": pv,
            "y_pred_vix_level": vix_level.astype(float),
            "vix": vix_level.astype(float),
            "qlike_gnn_t": qg,
            "qlike_vix_t": qv,
            "d_qlike_t": d_qlike,
            "mse_log_gnn_t": mse_g,
            "mse_log_vix_t": mse_v,
            "d_mse_log_t": d_mse,
        }
    )


def _default_ecfg(dm_losses: tuple[str, ...] = ("qlike",)) -> ModelEvaluateConfig:
    return ModelEvaluateConfig(
        splits=("test",),
        batch_size=32,
        device="auto",
        verbose=False,
        summary_test_exclusion_years=(2020,),
        summary_train_crisis_excl_start="2008-09-01",
        summary_train_crisis_excl_end="2009-03-31",
        test_metric=None,
        hypothesis_hac_max_lags=5,
        hypothesis_dm_losses=dm_losses,
        summary_test_stress_excl_start="2008-09-01",
        summary_test_stress_excl_end="2009-03-31",
        hypothesis_h4_include_h1=False,
        apply_log_calibration=True,
    )


def test_hypothesis_test_rows_primary_test_h1_h3_two_sided_h2_one_sided_primary() -> None:
    panel = _panel_n_test_dates(n=12)
    ecfg = _default_ecfg(dm_losses=("qlike", "mse_log"))
    hrows, _, _ = run_hypothesis_tests(panel, ecfg, hac_max_lags=ecfg.hypothesis_hac_max_lags)
    primary = [r for r in hrows if r.get("hypothesis_id") != "insufficient_data" and r["sample"] == "test"]

    h1 = next(r for r in primary if r["hypothesis_id"] == "H1" and r["inference_procedure"] == PROC_MZ)
    assert h1["coefficient_tested"] == "y_pred_model_log"
    assert h1["test_scope"] == "slope_only"
    assert h1["joint_hypothesis"] == "none"
    assert h1["tail"] == "two_sided"
    assert h1["p_value_primary"] == h1["p_value_two_sided"]
    assert np.isfinite(h1["p_value_primary"])

    h3 = next(r for r in primary if r["hypothesis_id"] == "H3")
    assert h3["coefficient_tested"] == "y_pred_model_log"
    assert h3["test_scope"] == "slope_only"
    assert h3["tail"] == "two_sided"
    assert h3["p_value_primary"] == h3["p_value_two_sided"]
    assert h1["statistic_type"] == "hac_t"
    assert h3["statistic_type"] == "hac_t"

    h2_q = next(
        r for r in primary if r["hypothesis_id"] == "H2" and r["inference_procedure"] == PROC_DM and r["loss_name"] == "qlike"
    )
    assert h2_q["tail"] == "upper"
    assert h2_q["better_model"] == "gnn"
    assert h2_q["alternative_direction"] == "mean_d_gt_0_favors_gnn"
    assert h2_q["p_value_primary"] == h2_q["p_value_one_sided_upper"]
    assert np.isfinite(h2_q["p_value_two_sided"])
    assert 0.0 <= h2_q["p_value_primary"] <= 1.0
    assert h2_q["statistic_type"] == "hac_t_mean"

    h2_m = next(
        r for r in primary if r["hypothesis_id"] == "H2" and r["loss_name"] == "mse_log"
    )
    assert h2_m["loss_name"] == "mse_log"
    assert h2_m["p_value_primary"] == h2_m["p_value_one_sided_upper"]


def test_h2_row_self_describing_without_other_rows() -> None:
    panel = _panel_n_test_dates(n=10)
    ecfg = _default_ecfg(dm_losses=("qlike",))
    hrows, _, _ = run_hypothesis_tests(panel, ecfg, hac_max_lags=4)
    h2 = next(r for r in hrows if r.get("hypothesis_id") == "H2" and r["sample"] == "test" and r["loss_name"] == "qlike")
    required = (
        "alternative",
        "tail",
        "better_model",
        "loss_name",
        "loss_definition",
        "p_value_primary",
        "p_value_two_sided",
    )
    for key in required:
        assert key in h2
    assert h2["tail"] == "upper"
    assert h2["better_model"] == "gnn"
    assert str(h2["loss_definition"]).strip()


def test_skip_duplicate_subsamples_emits_audit_rows() -> None:
    """Stress window 2008-2009 does not overlap 2021 test dates: no duplicate H4 inference."""
    panel = _panel_n_test_dates(n=12)
    ecfg = _default_ecfg(dm_losses=("qlike",))
    hrows, mz, inc = run_hypothesis_tests(panel, ecfg, hac_max_lags=5)
    stress_h4 = [r for r in hrows if r["sample"] == "test_excl_stress"]
    assert any(r["inference_procedure"] == PROC_AUDIT for r in stress_h4)
    assert not any(
        r["inference_procedure"] == PROC_DM and r["sample"] == "test_excl_stress" for r in hrows
    )
    union_h4 = [r for r in hrows if r["sample"] == "test_excl_union"]
    assert any(r["inference_procedure"] == PROC_AUDIT for r in union_h4)
    assert not any(
        r["inference_procedure"] == PROC_DM and r["sample"] == "test_excl_union" for r in hrows
    )
    audit = next(r for r in hrows if r["sample"] == "test_excl_stress" and r["inference_procedure"] == PROC_AUDIT)
    assert audit["subsample_operational"] is False
    assert np.isnan(audit["p_value_primary"])
    # No regression rows for skipped stress subsample
    assert not any(r["sample"] == "test_excl_stress" for r in mz)
    assert not any(r["sample"] == "test_excl_stress" for r in inc)


def test_stress_exclusion_operative_changes_sample_size() -> None:
    """When test dates fall inside configured stress window, test_excl_stress is strictly smaller."""
    rng = np.random.default_rng(1)
    n_in = 8
    n_out = 8
    dates_in = pd.bdate_range("2008-10-01", periods=n_in)
    dates_out = pd.bdate_range("2021-01-04", periods=n_out)
    dates = pd.DatetimeIndex(list(dates_in) + list(dates_out))
    n = len(dates)
    yt = rng.normal(0.12, 0.01, size=n)
    pm = yt + rng.normal(0.0, 0.02, size=n)
    vix_level = np.full(n, 25.0)
    pv = np.log(np.maximum(vix_level, 1e-12))
    y_level = np.exp(yt)
    pml = np.exp(pm)
    qg = qlike_per_date(y_level, pml, eps=1e-12)
    qv = qlike_per_date(y_level, vix_level, eps=1e-12)
    d_qlike = qv - qg
    mse_g = (yt - pm) ** 2
    mse_v = (yt - pv) ** 2
    d_mse = mse_v - mse_g
    panel = pd.DataFrame(
        {
            "date": dates,
            "split": ["test"] * n,
            "sample": ["test_full"] * n,
            "y_true_log": yt,
            "y_true_level": y_level,
            "y_pred_model_log": pm,
            "y_pred_model_level": pml,
            "y_pred_vix_log": pv,
            "y_pred_vix_level": vix_level.astype(float),
            "vix": vix_level.astype(float),
            "qlike_gnn_t": qg,
            "qlike_vix_t": qv,
            "d_qlike_t": d_qlike,
            "mse_log_gnn_t": mse_g,
            "mse_log_vix_t": mse_v,
            "d_mse_log_t": d_mse,
        }
    )
    ecfg = _default_ecfg(dm_losses=("qlike",))
    specs = build_formal_subsample_specs(panel, ecfg)
    assert specs["test_excl_stress"].stress_exclusion_operative
    assert specs["test_excl_stress"].n_obs < specs["test"].n_obs
    hrows, _, _ = run_hypothesis_tests(panel, ecfg, hac_max_lags=5)
    stress_dm = [r for r in hrows if r["sample"] == "test_excl_stress" and r["inference_procedure"] == PROC_DM]
    assert len(stress_dm) == 1
    assert stress_dm[0]["n_obs"] < specs["test"].n_obs
