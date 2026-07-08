"""Formal HAC-aware hypothesis tests for thesis; reads only forecast_panel.parquet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from mss.evaluation.config import ModelEvaluateConfig

# Verbatim thesis statements (README + data_contracts must match).
H1_NULL = (
    "The GNN forecast contains no predictive information about future 30-calendar-day "
    "realized market volatility."
)
H1_ALT = (
    "The GNN forecast contains predictive information about future 30-calendar-day "
    "realized market volatility."
)

H2_NULL = (
    "The GNN does not achieve lower out-of-sample forecast loss than the raw VIX benchmark."
)
H2_ALT = "The GNN achieves lower out-of-sample forecast loss than the raw VIX benchmark."

H3_NULL = (
    "Conditional on the raw VIX benchmark, the GNN forecast adds no additional predictive "
    "information about future 30-calendar-day realized market volatility."
)
H3_ALT = (
    "Conditional on the raw VIX benchmark, the GNN forecast adds additional predictive "
    "information about future 30-calendar-day realized market volatility."
)

H4_NULL = (
    "Any apparent forecasting advantage of the GNN disappears once pre-specified extreme "
    "market episodes are excluded from the test sample."
)
H4_ALT = (
    "The GNN retains its relative forecasting advantage even when pre-specified extreme "
    "market episodes are excluded from the test sample."
)

H5_NULL = (
    "The GNN does not achieve lower out-of-sample forecast loss than the placebo benchmark."
)
H5_ALT = (
    "The GNN achieves lower out-of-sample forecast loss than the placebo benchmark."
)

DM_LOSS_COL = {"qlike": "d_qlike_t", "mse_log": "d_mse_log_t"}
DM_LOSS_COL_PLACEBO = {"qlike": "d_qlike_placebo_t", "mse_log": "d_mse_log_placebo_t"}

# v2 benchmark suite mappings. raw_vix is the canonical H2/H3 comparison; the others drive
# generalized DM (H2_benchmark) and incremental_vs_benchmark rows distinguished by the benchmark field.
BENCHMARK_DM_LOSS_COL = {
    "raw_vix": {"qlike": "d_qlike_t", "mse_log": "d_mse_log_t"},
    "calibrated_vix": {"qlike": "d_qlike_calibrated_vix_t", "mse_log": "d_mse_log_calibrated_vix_t"},
    "vix_har": {"qlike": "d_qlike_vix_har_t", "mse_log": "d_mse_log_vix_har_t"},
    "har": {"qlike": "d_qlike_har_t", "mse_log": "d_mse_log_har_t"},
}
BENCHMARK_LOG_COL = {
    "raw_vix": "y_pred_vix_log",
    "calibrated_vix": "y_pred_calibrated_vix_log",
    "vix_har": "y_pred_vix_har_log",
    "har": "y_pred_har_log",
}
BENCHMARK_DISPLAY = {
    "raw_vix": "raw spot VIX (primary market-implied benchmark)",
    "calibrated_vix": "train-calibrated (bias-adjusted) VIX",
    "vix_har": "hybrid implied-plus-historical (VIX-HAR) benchmark",
    "har": "standalone realized-vol HAR benchmark",
}
DM_LOSS_DEFINITION = {
    "qlike": "QLIKE level: d_t = L_VIX_t - L_GNN_t per date; positive => lower loss for GNN.",
    "mse_log": "MSE in log target: d_t = squared_err_VIX - squared_err_GNN; positive => GNN better.",
}
DM_LOSS_DEFINITION_PLACEBO = {
    "qlike": "QLIKE level: d_t = L_placebo_t - L_GNN_t per date; positive => lower loss for GNN.",
    "mse_log": "MSE in log target: d_t = squared_err_placebo - squared_err_GNN; positive => GNN better.",
}

PROC_MZ = "mz_gnn"
PROC_DM = "dm_loss_diff"
PROC_INC = "incremental_gnn_vix"
PROC_INC_BENCH = "incremental_vs_benchmark"
PROC_AUDIT = "subsample_audit"

STAT_HAC_T = "hac_t"
STAT_HAC_T_MEAN = "hac_t_mean"

REGRESSION_EXPORT_COLUMNS = (
    "sample",
    "dep_var",
    "spec",
    "term",
    "coef",
    "se_hac",
    "t",
    "p_two_sided",
    "ci_lo_95",
    "ci_hi_95",
    "r_squared",
    "n_obs",
    "hac_max_lags",
    "coefficient_tested_for_hypothesis",
    "test_scope",
    "rows_removed_vs_full_test",
    "effective_sample_start",
    "effective_sample_end",
    "stress_rows_removed",
    "year_rows_removed",
    "subsample_operational",
)


@dataclass(frozen=True)
class FormalSubsampleSpec:
    """Formal inference subsample: mask on panel rows + audit metadata."""

    name: str
    mask: pd.Series
    n_obs: int
    effective_sample_start: str
    effective_sample_end: str
    rows_removed_vs_full_test: int
    stress_rows_removed: int
    year_rows_removed: int
    stress_exclusion_operative: bool
    year_exclusion_operative: bool
    skip_inference: bool
    skip_reason: str


def _norm_sf(x: float) -> float:
    return float(scipy_stats.norm.sf(x))


def _dm_hac(
    d: np.ndarray,
    *,
    maxlags: int,
) -> tuple[float, float, float, float]:
    """Return mean_d, t_stat, p_two_sided, p_one_sided_upper (H1: mean d > 0)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    X = np.ones((n, 1))
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    mean_d = float(res.params[0])
    se = float(np.sqrt(res.cov_params()[0, 0]))
    if se <= 0 or not math.isfinite(se):
        return mean_d, float("nan"), float("nan"), float("nan")
    t_stat = mean_d / se
    p_two = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(t_stat))))
    p_upper = _norm_sf(t_stat)
    return mean_d, float(t_stat), p_two, p_upper


def _mz_incremental_rows(
    y: np.ndarray,
    x_mat: np.ndarray,
    names: list[str],
    *,
    maxlags: int,
    idx_test: int,
) -> tuple[float, float]:
    res = sm.OLS(y, x_mat).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[idx_test]), float(res.pvalues[idx_test])


def _audit_fields_from_spec(spec: FormalSubsampleSpec) -> dict[str, Any]:
    return {
        "rows_removed_vs_full_test": int(spec.rows_removed_vs_full_test),
        "effective_sample_start": spec.effective_sample_start,
        "effective_sample_end": spec.effective_sample_end,
        "stress_rows_removed": int(spec.stress_rows_removed),
        "year_rows_removed": int(spec.year_rows_removed),
        "subsample_operational": not spec.skip_inference,
    }


def _full_regression_export(
    y: np.ndarray,
    x_mat: np.ndarray,
    names: list[str],
    *,
    maxlags: int,
    sample: str,
    spec: str,
    dep_var: str,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    res = sm.OLS(y, x_mat).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    ci = res.conf_int()
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        tested = ""
        scope = "intercept"
        if i > 0:
            tested = name
            scope = "slope_only"
        row = {
            "sample": sample,
            "dep_var": dep_var,
            "spec": spec,
            "term": name,
            "coef": float(res.params[i]),
            "se_hac": float(res.bse[i]),
            "t": float(res.tvalues[i]),
            "p_two_sided": float(res.pvalues[i]),
            "ci_lo_95": float(ci[i, 0]),
            "ci_hi_95": float(ci[i, 1]),
            "r_squared": float(res.rsquared),
            "n_obs": int(res.nobs),
            "hac_max_lags": int(maxlags),
            "coefficient_tested_for_hypothesis": tested,
            "test_scope": scope,
        }
        row.update(audit)
        rows.append(row)
    return rows


def build_formal_subsample_specs(df: pd.DataFrame, ecfg: ModelEvaluateConfig) -> dict[str, FormalSubsampleSpec]:
    """Build masks and audit metadata for each formal inference subsample."""
    sp = df["split"].astype(str)
    test = sp == "test"
    dt = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    years = dt.dt.year
    stress_start = pd.Timestamp(ecfg.summary_test_stress_excl_start).normalize()
    stress_end = pd.Timestamp(ecfg.summary_test_stress_excl_end).normalize()
    in_stress = (dt >= stress_start) & (dt <= stress_end)
    ex_y = set(ecfg.summary_test_exclusion_years)

    masks: dict[str, pd.Series] = {
        "test": test,
        "test_excl_2020": test & ~years.isin(ex_y),
        "test_excl_stress": test & ~in_stress,
        "test_excl_union": test & ~years.isin(ex_y) & ~in_stress,
    }

    n_test = int(test.sum())
    stress_rows_removed = int((test & in_stress).sum())
    year_rows_removed = int((test & years.isin(ex_y)).sum())
    stress_exclusion_operative = stress_rows_removed > 0
    year_exclusion_operative = year_rows_removed > 0

    mask_ex2020 = masks["test_excl_2020"]
    out: dict[str, FormalSubsampleSpec] = {}

    for name, mask in masks.items():
        n_obs = int(mask.sum())
        eff_start, eff_end = "", ""
        if n_obs > 0:
            sub_dates = pd.to_datetime(df.loc[mask, "date"], errors="coerce")
            eff_start = sub_dates.min().strftime("%Y-%m-%d")
            eff_end = sub_dates.max().strftime("%Y-%m-%d")
        rows_removed = n_test - n_obs

        skip = False
        reason = ""
        if name == "test_excl_stress" and not stress_exclusion_operative:
            skip = True
            reason = (
                "Skipped inference: configured stress window [summary_test_stress_excl_start, "
                "summary_test_stress_excl_end] does not intersect any test dates; "
                "test_excl_stress is identical to full test."
            )
        elif name == "test_excl_union" and mask.equals(mask_ex2020):
            skip = True
            reason = (
                "Skipped inference: stress exclusion removed no test rows, so test_excl_union "
                "is identical to test_excl_2020 (duplicate of existing H4 rows)."
            )

        out[name] = FormalSubsampleSpec(
            name=name,
            mask=mask,
            n_obs=n_obs,
            effective_sample_start=eff_start,
            effective_sample_end=eff_end,
            rows_removed_vs_full_test=rows_removed,
            stress_rows_removed=stress_rows_removed,
            year_rows_removed=year_rows_removed,
            stress_exclusion_operative=stress_exclusion_operative,
            year_exclusion_operative=year_exclusion_operative,
            skip_inference=skip,
            skip_reason=reason,
        )

    return out


def _benchmark_dm_rows(
    sub: pd.DataFrame,
    ecfg: ModelEvaluateConfig,
    *,
    hac_max_lags: int,
    sample_name: str,
    audit: dict[str, Any],
    stress_notes: str,
) -> list[dict[str, Any]]:
    """Generalized DM (H2-style) rows for each secondary benchmark; raw_vix stays canonical H2."""
    rows: list[dict[str, Any]] = []
    for bench in ecfg.hypothesis_dm_benchmarks:
        loss_map = BENCHMARK_DM_LOSS_COL.get(bench, {})
        disp = BENCHMARK_DISPLAY.get(bench, bench)
        for loss in ecfg.hypothesis_dm_losses:
            col = loss_map.get(loss)
            if col is None or col not in sub.columns:
                continue
            d = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if len(d) < 5:
                continue
            _, t_dm, p_two, p_upper = _dm_hac(d, maxlags=hac_max_lags)
            rows.append(
                {
                    "hypothesis_id": "H2_benchmark",
                    "inference_procedure": PROC_DM,
                    "statistic_type": STAT_HAC_T_MEAN,
                    "null_hypothesis": (
                        f"The GNN does not achieve lower out-of-sample forecast loss than the {disp}."
                    ),
                    "alternative": (
                        f"The GNN achieves lower out-of-sample forecast loss than the {disp}."
                    ),
                    "test_name": "diebold_mariano_mean_loss_diff",
                    "sample": sample_name,
                    "benchmark": bench,
                    "coefficient_tested": "",
                    "test_scope": "mean_loss_difference",
                    "joint_hypothesis": "none",
                    "tail": "upper",
                    "alternative_direction": "mean_d_gt_0_favors_gnn",
                    "better_model": "gnn",
                    "loss_name": loss,
                    "loss_definition": (
                        f"d_t = L_{bench}_t - L_GNN_t per date; positive => lower loss for GNN."
                    ),
                    "statistic": t_dm,
                    "p_value_primary": p_upper,
                    "p_value_two_sided": p_two,
                    "p_value_one_sided_upper": p_upper,
                    "hac_max_lags": hac_max_lags,
                    "n_obs": int(len(d)),
                    **audit,
                    "notes": (
                        f"Generalized DM vs {disp}; raw_vix remains the primary H2 benchmark."
                        + stress_notes
                    ),
                }
            )
    return rows


def _benchmark_incremental_rows(
    sub: pd.DataFrame,
    ecfg: ModelEvaluateConfig,
    *,
    hac_max_lags: int,
    sample_name: str,
    dep: str,
    audit: dict[str, Any],
    stress_notes: str,
) -> list[dict[str, Any]]:
    """incremental_vs_benchmark rows: HAC t on the GNN slope conditional on each secondary benchmark."""
    rows: list[dict[str, Any]] = []
    for bench in ecfg.hypothesis_incremental_benchmarks:
        bcol = BENCHMARK_LOG_COL.get(bench)
        disp = BENCHMARK_DISPLAY.get(bench, bench)
        if bcol is None or bcol not in sub.columns:
            continue
        y = pd.to_numeric(sub[dep], errors="coerce").to_numpy(dtype=float)
        xm = pd.to_numeric(sub["y_pred_model_log"], errors="coerce").to_numpy(dtype=float)
        xb = pd.to_numeric(sub[bcol], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(xm) & np.isfinite(xb)
        y, xm, xb = y[ok], xm[ok], xb[ok]
        if len(y) < 5:
            continue
        X = np.column_stack([np.ones(len(y)), xm, xb])
        names = ["const", "y_pred_model_log", bcol]
        t_sl, p_sl = _mz_incremental_rows(y, X, names, maxlags=hac_max_lags, idx_test=1)
        rows.append(
            {
                "hypothesis_id": "H3_benchmark",
                "inference_procedure": PROC_INC_BENCH,
                "statistic_type": STAT_HAC_T,
                "null_hypothesis": (
                    f"Conditional on the {disp}, the GNN forecast adds no additional predictive "
                    "information about future 30-calendar-day realized market volatility."
                ),
                "alternative": (
                    f"Conditional on the {disp}, the GNN forecast adds additional predictive "
                    "information about future 30-calendar-day realized market volatility."
                ),
                "test_name": "incremental_gnn_slope_with_benchmark",
                "sample": sample_name,
                "benchmark": bench,
                "coefficient_tested": "y_pred_model_log",
                "test_scope": "slope_only",
                "joint_hypothesis": "none",
                "tail": "two_sided",
                "alternative_direction": "beta_gnn_neq_0_conditional_on_benchmark",
                "better_model": "",
                "loss_name": "",
                "loss_definition": "",
                "statistic": t_sl,
                "p_value_primary": p_sl,
                "p_value_two_sided": p_sl,
                "p_value_one_sided_upper": "",
                "hac_max_lags": hac_max_lags,
                "n_obs": int(len(y)),
                **audit,
                "notes": (
                    f"incremental_vs_benchmark: two-sided HAC t on GNN slope conditional on {disp}."
                    + stress_notes
                ),
            }
        )
    return rows


def run_hypothesis_tests(
    panel: pd.DataFrame,
    ecfg: ModelEvaluateConfig,
    *,
    hac_max_lags: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (hypothesis_test_rows, mz_regression_rows, incremental_regression_rows).

    H4 is implemented by re-running H2 and H3 (and optional H1 if hypothesis_h4_include_h1)
    on restricted test subsamples; hypothesis_id H4 with inference_procedure identifying the method.
    """
    hrows: list[dict[str, Any]] = []
    mz_all: list[dict[str, Any]] = []
    inc_all: list[dict[str, Any]] = []

    specs = build_formal_subsample_specs(panel, ecfg)
    dep = "y_true_log"

    stress_notes = (
        f" stress_preset={ecfg.stress_preset_active or 'override'}; "
        f"stress_window=[{ecfg.summary_test_stress_excl_start}, {ecfg.summary_test_stress_excl_end}]."
    )

    h4_note_suffix = (
        " H4 re-runs the same procedures as H1–H3 on restricted formal test subsamples "
        f"(optional H1 on non-primary subsamples when hypothesis_h4_include_h1=true in config)."
    )

    for sample_name in ("test", "test_excl_2020", "test_excl_stress", "test_excl_union"):
        spec = specs[sample_name]
        if spec.n_obs < 5:
            continue

        if spec.skip_inference:
            hrows.append(
                {
                    "hypothesis_id": "H4",
                    "inference_procedure": PROC_AUDIT,
                    "statistic_type": "",
                    "null_hypothesis": H4_NULL,
                    "alternative": H4_ALT,
                    "test_name": "subsample_skipped_duplicate",
                    "sample": sample_name,
                    "benchmark": "",
                    "coefficient_tested": "",
                    "test_scope": "subsample_audit",
                    "joint_hypothesis": "none",
                    "tail": "na",
                    "alternative_direction": "",
                    "better_model": "",
                    "loss_name": "",
                    "loss_definition": "",
                    "statistic": float("nan"),
                    "p_value_primary": float("nan"),
                    "p_value_two_sided": float("nan"),
                    "p_value_one_sided_upper": "",
                    "hac_max_lags": hac_max_lags,
                    "n_obs": spec.n_obs,
                    **_audit_fields_from_spec(spec),
                    "notes": spec.skip_reason + h4_note_suffix + stress_notes,
                }
            )
            continue

        sub = panel.loc[spec.mask].copy()
        sub = sub.sort_values("date")
        audit = _audit_fields_from_spec(spec)

        is_primary_sample = sample_name == "test"

        # --- MZ GNN (H1 / H4)
        if is_primary_sample or ecfg.hypothesis_h4_include_h1:
            y = pd.to_numeric(sub[dep], errors="coerce").to_numpy(dtype=float)
            x1 = pd.to_numeric(sub["y_pred_model_log"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(x1)
            y, x1 = y[ok], x1[ok]
            if len(y) >= 5:
                X = sm.add_constant(x1, has_constant="add")
                names = ["const", "y_pred_model_log"]
                t_sl, p_sl = _mz_incremental_rows(y, X, names, maxlags=hac_max_lags, idx_test=1)
                th = "H1" if is_primary_sample else "H4"
                base_notes = (
                    "H1: two-sided HAC t on slope of y_pred_model_log in "
                    f"y_true_log ~ const + y_pred_model_log; formal_inference subsample={sample_name}."
                )
                hrows.append(
                    {
                        "hypothesis_id": th,
                        "inference_procedure": PROC_MZ,
                        "statistic_type": STAT_HAC_T,
                        "null_hypothesis": H1_NULL if th == "H1" else H4_NULL,
                        "alternative": H1_ALT if th == "H1" else H4_ALT,
                        "test_name": "mincer_zarnowitz_gnn_slope",
                        "sample": sample_name,
                        "benchmark": "none",
                        "coefficient_tested": "y_pred_model_log",
                        "test_scope": "slope_only",
                        "joint_hypothesis": "none",
                        "tail": "two_sided",
                        "alternative_direction": "beta_neq_0",
                        "better_model": "",
                        "loss_name": "",
                        "loss_definition": "",
                        "statistic": t_sl,
                        "p_value_primary": p_sl,
                        "p_value_two_sided": p_sl,
                        "p_value_one_sided_upper": "",
                        "hac_max_lags": hac_max_lags,
                        "n_obs": int(len(y)),
                        **audit,
                        "notes": base_notes + (h4_note_suffix if th == "H4" else "") + stress_notes,
                    }
                )
                mz_all.extend(
                    _full_regression_export(
                        y,
                        X,
                        names,
                        maxlags=hac_max_lags,
                        sample=sample_name,
                        spec="mz_gnn",
                        dep_var=dep,
                        audit=audit,
                    )
                )

        # --- DM (H2 / H4)
        for loss in ecfg.hypothesis_dm_losses:
            col = DM_LOSS_COL.get(loss)
            if col is None or col not in sub.columns:
                continue
            d = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if len(d) < 5:
                continue
            _, t_dm, p_two, p_upper = _dm_hac(d, maxlags=hac_max_lags)
            th = "H2" if is_primary_sample else "H4"
            base_notes = (
                "H2: primary p_value is one-sided upper (E[d]>0, d=L_VIX-L_GNN); "
                "two-sided DM p in p_value_two_sided."
            )
            hrows.append(
                {
                    "hypothesis_id": th,
                    "inference_procedure": PROC_DM,
                    "statistic_type": STAT_HAC_T_MEAN,
                    "null_hypothesis": H2_NULL if th == "H2" else H4_NULL,
                    "alternative": H2_ALT if th == "H2" else H4_ALT,
                    "test_name": "diebold_mariano_mean_loss_diff",
                    "sample": sample_name,
                    "benchmark": "raw_vix",
                    "coefficient_tested": "",
                    "test_scope": "mean_loss_difference",
                    "joint_hypothesis": "none",
                    "tail": "upper",
                    "alternative_direction": "mean_d_gt_0_favors_gnn",
                    "better_model": "gnn",
                    "loss_name": loss,
                    "loss_definition": DM_LOSS_DEFINITION[loss],
                    "statistic": t_dm,
                    "p_value_primary": p_upper,
                    "p_value_two_sided": p_two,
                    "p_value_one_sided_upper": p_upper,
                    "hac_max_lags": hac_max_lags,
                    "n_obs": int(len(d)),
                    **audit,
                    "notes": base_notes + (h4_note_suffix if th == "H4" else "") + stress_notes,
                }
            )

        # --- DM vs placebo (H5, primary test only)
        if is_primary_sample:
            for loss in ecfg.hypothesis_dm_losses_vs_placebo:
                col = DM_LOSS_COL_PLACEBO.get(loss)
                if col is None or col not in sub.columns:
                    continue
                d = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                d = d[np.isfinite(d)]
                if len(d) < 5:
                    continue
                _, t_dm, p_two, p_upper = _dm_hac(d, maxlags=hac_max_lags)
                base_notes = (
                    "H5: primary p_value is one-sided upper (E[d]>0, d=L_placebo-L_GNN); "
                    "two-sided DM p in p_value_two_sided."
                )
                hrows.append(
                    {
                        "hypothesis_id": "H5",
                        "inference_procedure": PROC_DM,
                        "statistic_type": STAT_HAC_T_MEAN,
                        "null_hypothesis": H5_NULL,
                        "alternative": H5_ALT,
                        "test_name": "diebold_mariano_mean_loss_diff_vs_placebo",
                        "sample": sample_name,
                        "benchmark": "placebo",
                        "coefficient_tested": "",
                        "test_scope": "mean_loss_difference",
                        "joint_hypothesis": "none",
                        "tail": "upper",
                        "alternative_direction": "mean_d_gt_0_favors_gnn",
                        "better_model": "gnn",
                        "loss_name": loss,
                        "loss_definition": DM_LOSS_DEFINITION_PLACEBO[loss],
                        "statistic": t_dm,
                        "p_value_primary": p_upper,
                        "p_value_two_sided": p_two,
                        "p_value_one_sided_upper": p_upper,
                        "hac_max_lags": hac_max_lags,
                        "n_obs": int(len(d)),
                        **audit,
                        "notes": base_notes + stress_notes,
                    }
                )

        # --- Incremental (H3 / H4)
        y = pd.to_numeric(sub[dep], errors="coerce").to_numpy(dtype=float)
        xm = pd.to_numeric(sub["y_pred_model_log"], errors="coerce").to_numpy(dtype=float)
        xv = pd.to_numeric(sub["y_pred_vix_log"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(xm) & np.isfinite(xv)
        y, xm, xv = y[ok], xm[ok], xv[ok]
        if len(y) >= 5:
            X = np.column_stack([np.ones(len(y)), xm, xv])
            names = ["const", "y_pred_model_log", "y_pred_vix_log"]
            t_sl, p_sl = _mz_incremental_rows(y, X, names, maxlags=hac_max_lags, idx_test=1)
            th = "H3" if is_primary_sample else "H4"
            base_notes = (
                "H3: two-sided HAC t on y_pred_model_log in "
                "y_true_log ~ const + y_pred_model_log + y_pred_vix_log."
            )
            hrows.append(
                {
                    "hypothesis_id": th,
                    "inference_procedure": PROC_INC,
                    "statistic_type": STAT_HAC_T,
                    "null_hypothesis": H3_NULL if th == "H3" else H4_NULL,
                    "alternative": H3_ALT if th == "H3" else H4_ALT,
                    "test_name": "incremental_gnn_slope_with_vix",
                    "sample": sample_name,
                    "benchmark": "raw_vix",
                    "coefficient_tested": "y_pred_model_log",
                    "test_scope": "slope_only",
                    "joint_hypothesis": "none",
                    "tail": "two_sided",
                    "alternative_direction": "beta_gnn_neq_0_conditional_on_vix",
                    "better_model": "",
                    "loss_name": "",
                    "loss_definition": "",
                    "statistic": t_sl,
                    "p_value_primary": p_sl,
                    "p_value_two_sided": p_sl,
                    "p_value_one_sided_upper": "",
                    "hac_max_lags": hac_max_lags,
                    "n_obs": int(len(y)),
                    **audit,
                    "notes": base_notes + (h4_note_suffix if th == "H4" else "") + stress_notes,
                }
            )
            inc_all.extend(
                _full_regression_export(
                    y,
                    X,
                    names,
                    maxlags=hac_max_lags,
                    sample=sample_name,
                    spec="incremental_gnn_vix",
                    dep_var=dep,
                    audit=audit,
                )
            )

        # --- Generalized benchmark comparisons (primary test only).
        # raw_vix stays the canonical H2/H3 above; secondary benchmarks get DM rows
        # (hypothesis_id H2_benchmark) and incremental_vs_benchmark rows (H3_benchmark),
        # distinguished by the `benchmark` field. We deliberately do not relabel these as
        # "beyond VIX": each row states the benchmark it conditions on / competes against.
        if is_primary_sample:
            hrows.extend(
                _benchmark_dm_rows(
                    sub, ecfg, hac_max_lags=hac_max_lags, sample_name=sample_name,
                    audit=audit, stress_notes=stress_notes,
                )
            )
            hrows.extend(
                _benchmark_incremental_rows(
                    sub, ecfg, hac_max_lags=hac_max_lags, sample_name=sample_name,
                    dep=dep, audit=audit, stress_notes=stress_notes,
                )
            )

    if not hrows:
        hrows.append(
            {
                "hypothesis_id": "insufficient_data",
                "inference_procedure": "",
                "statistic_type": "",
                "null_hypothesis": "",
                "alternative": "",
                "test_name": "none",
                "sample": "",
                "benchmark": "",
                "coefficient_tested": "",
                "test_scope": "",
                "joint_hypothesis": "none",
                "tail": "na",
                "alternative_direction": "",
                "better_model": "",
                "loss_name": "",
                "loss_definition": "",
                "statistic": float("nan"),
                "p_value_primary": float("nan"),
                "p_value_two_sided": float("nan"),
                "p_value_one_sided_upper": "",
                "hac_max_lags": hac_max_lags,
                "n_obs": 0,
                "rows_removed_vs_full_test": 0,
                "effective_sample_start": "",
                "effective_sample_end": "",
                "stress_rows_removed": 0,
                "year_rows_removed": 0,
                "subsample_operational": False,
                "notes": (
                    "No formal hypothesis rows were produced: each formal inference subsample "
                    "needs at least 5 test dates after masking. Run on longer series for thesis inference."
                ),
            }
        )

    return hrows, mz_all, inc_all
