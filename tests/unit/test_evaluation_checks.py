from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mss.evaluation.checks import (
    check_forecasts,
    check_forecast_panel,
    check_hypothesis_tests,
    check_summary_table,
    check_test_loss,
    model_evaluate_step_semantically_complete,
)
from mss.io.config import ResolvedConfig


def _cfg(tmp: Path) -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    return ResolvedConfig(
        run_id="t",
        project_root=tmp,
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        source_config_path=cfg_toml.resolve(),
    )


def test_check_forecasts_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "forecasts.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "split": ["test", "test"],
            "sample": ["test_full", "test_full"],
            "y_true": [0.1, 0.2],
            "y_pred_model": [0.11, 0.22],
            "y_pred_placebo": [0.12, 0.21],
        }
    ).to_parquet(p, index=False)
    out = check_forecasts(p)
    assert out["passed"]


def test_check_forecasts_duplicate_dates_fail(tmp_path: Path) -> None:
    p = tmp_path / "forecasts.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
            "split": ["test", "test"],
            "sample": ["test_full", "test_full"],
            "y_true": [0.1, 0.2],
            "y_pred_model": [0.11, 0.22],
            "y_pred_placebo": [0.12, 0.21],
        }
    ).to_parquet(p, index=False)
    out = check_forecasts(p)
    assert not out["passed"]


def test_check_summary_table_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "summary_table.csv"
    p.write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n"
        "gnn,full_test,0.01,0.02,1.0,1.1,1.2,10\n"
        "vix,full_test,0.02,0.03,1.1,1.2,1.3,10\n",
        encoding="utf-8",
    )
    assert check_summary_table(p)["passed"]


def test_check_test_loss_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "test_loss.json"
    p.write_text(
        json.dumps(
            {
                "status": "completed",
                "metric": "mse_log",
                "split": "test",
                "n_rows": 10,
                "value": 0.1,
            }
        ),
        encoding="utf-8",
    )
    out = check_test_loss(p)
    assert out["passed"]


def test_check_test_loss_qlike_metric(tmp_path: Path) -> None:
    p = tmp_path / "test_loss.json"
    p.write_text(
        json.dumps(
            {
                "status": "completed",
                "metric": "qlike_level",
                "split": "test",
                "n_rows": 5,
                "value": 1.23,
            }
        ),
        encoding="utf-8",
    )
    assert check_test_loss(p)["passed"]


def _minimal_forecast_panel_df() -> pd.DataFrame:
    import numpy as np

    from mss.evaluation.metrics import qlike_per_date

    yt, pm, vx = 0.1, 0.11, 18.0
    pv = float(np.log(max(vx, 1e-12)))
    yl = np.exp(yt)
    pml = np.exp(pm)
    qg = qlike_per_date(np.array([yl]), np.array([pml]), eps=1e-12)
    qv = qlike_per_date(np.array([yl]), np.array([float(vx)]), eps=1e-12)
    dg = qv - qg
    mg = (yt - pm) ** 2
    mv = (yt - pv) ** 2
    dm = mv - mg
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
            "sample": ["test_full"],
            "y_true_log": [yt],
            "y_true_level": [yl],
            "y_pred_model_log": [pm],
            "y_pred_model_level": [pml],
            "y_pred_placebo_log": [pm],
            "y_pred_placebo_level": [pml],
            "y_pred_har_log": [pv],
            "y_pred_har_level": [float(vx)],
            "y_pred_vix_log": [pv],
            "y_pred_vix_level": [float(vx)],
            "vix": [float(vx)],
            "qlike_gnn_t": [float(qg[0])],
            "qlike_placebo_t": [float(qg[0])],
            "qlike_har_t": [float(qv[0])],
            "qlike_vix_t": [float(qv[0])],
            "d_qlike_t": [float(dg[0])],
            "mse_log_gnn_t": [mg],
            "mse_log_placebo_t": [mg],
            "mse_log_har_t": [mv],
            "mse_log_vix_t": [mv],
            "d_mse_log_t": [dm],
        }
    )


def _minimal_hypothesis_tests_row(hac_lags: int = 29) -> dict:
    return {
        "hypothesis_id": "H2",
        "inference_procedure": "dm_loss_diff",
        "statistic_type": "hac_t_mean",
        "null_hypothesis": "x",
        "alternative": "y",
        "test_name": "diebold_mariano_mean_loss_diff",
        "sample": "test",
        "coefficient_tested": "",
        "test_scope": "mean_loss_difference",
        "joint_hypothesis": "none",
        "tail": "upper",
        "alternative_direction": "mean_d_gt_0_favors_gnn",
        "better_model": "gnn",
        "loss_name": "qlike",
        "loss_definition": "def",
        "statistic": 1.0,
        "p_value_primary": 0.05,
        "p_value_two_sided": 0.1,
        "p_value_one_sided_upper": 0.05,
        "hac_max_lags": hac_lags,
        "n_obs": 100,
        "rows_removed_vs_full_test": 0,
        "effective_sample_start": "2020-01-02",
        "effective_sample_end": "2020-01-02",
        "stress_rows_removed": 0,
        "year_rows_removed": 0,
        "subsample_operational": True,
        "notes": "fixture",
    }


def test_check_test_loss_invalid_metric(tmp_path: Path) -> None:
    p = tmp_path / "test_loss.json"
    p.write_text(
        json.dumps(
            {
                "status": "completed",
                "metric": "not_a_metric",
                "split": "test",
                "n_rows": 5,
                "value": 1.0,
            }
        ),
        encoding="utf-8",
    )
    assert not check_test_loss(p)["passed"]


def test_model_evaluate_semantic_complete(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert not model_evaluate_step_semantically_complete(cfg)
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    fc = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
            "sample": ["test_full"],
            "y_true": [0.1],
            "y_pred_model": [0.2],
            "y_pred_placebo": [0.19],
        }
    )
    fc.to_parquet(proc / "forecasts.parquet", index=False)
    _minimal_forecast_panel_df().to_parquet(proc / "forecast_panel.parquet", index=False)
    assert check_forecast_panel(proc / "forecast_panel.parquet")["passed"]
    (proc / "test_loss.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "metric": "mse_log",
                "split": "test",
                "n_rows": 1,
                "value": 0.1,
            }
        ),
        encoding="utf-8",
    )
    summ = proc / "summaries"
    summ.mkdir(parents=True, exist_ok=True)
    (summ / "summary_table.csv").write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n"
        "gnn,full_test,0.01,0.02,1.0,1.1,1.2,10\n"
        "vix,full_test,0.02,0.03,1.1,1.2,1.3,10\n",
        encoding="utf-8",
    )
    (summ / "diagnostics_smoothing.csv").write_text(
        "sample,diagnostic,model,value,detail\n"
        "full_test,forecast_variance_log,gnn,0.01,\n",
        encoding="utf-8",
    )
    pd.DataFrame([_minimal_hypothesis_tests_row()]).to_csv(summ / "hypothesis_tests.csv", index=False)
    assert check_hypothesis_tests(summ / "hypothesis_tests.csv")["passed"]
    assert model_evaluate_step_semantically_complete(cfg)

