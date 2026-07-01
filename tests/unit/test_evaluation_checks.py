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
from tests.conftest import make_resolved_config


def _cfg(tmp: Path) -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    return make_resolved_config(tmp, source_config_path=cfg_toml.resolve())


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
            "d_qlike_placebo_t": [0.0],
            "mse_log_gnn_t": [mg],
            "mse_log_placebo_t": [mg],
            "mse_log_har_t": [mv],
            "mse_log_vix_t": [mv],
            "d_mse_log_t": [dm],
            "d_mse_log_placebo_t": [0.0],
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
        "benchmark": "raw_vix",
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


def test_check_forecast_panel_har_warmup_nan_on_train(tmp_path: Path) -> None:
    """Early train rows may lack 252-day HAR lags; val/test must still be finite."""
    import numpy as np

    from mss.evaluation.metrics import qlike_per_date

    rows: list[dict] = []
    train_dates = pd.bdate_range("2000-01-03", periods=5)
    test_dates = pd.bdate_range("2020-01-02", periods=2)
    for i, d in enumerate(train_dates):
        yt, pm, vx = 0.1 + i * 0.01, 0.11, 18.0
        pv = float(np.log(max(vx, 1e-12)))
        yl, pml = np.exp(yt), np.exp(pm)
        qg = float(qlike_per_date(np.array([yl]), np.array([pml]), eps=1e-12)[0])
        qv = float(qlike_per_date(np.array([yl]), np.array([float(vx)]), eps=1e-12)[0])
        har_log = np.nan if i < 2 else pv
        har_lvl = np.nan if i < 2 else float(vx)
        qhar = np.nan if i < 2 else qv
        mhar = np.nan if i < 2 else float((yt - pv) ** 2)
        rows.append(
            {
                "date": d,
                "split": "train",
                "sample": "train_full",
                "y_true_log": yt,
                "y_true_level": yl,
                "y_pred_model_log": pm,
                "y_pred_model_level": pml,
                "y_pred_placebo_log": pm,
                "y_pred_placebo_level": pml,
                "y_pred_har_log": har_log,
                "y_pred_har_level": har_lvl,
                "y_pred_vix_log": pv,
                "y_pred_vix_level": float(vx),
                "vix": float(vx),
                "qlike_gnn_t": qg,
                "qlike_placebo_t": qg,
                "qlike_har_t": qhar,
                "qlike_vix_t": qv,
                "d_qlike_t": qv - qg,
                "d_qlike_placebo_t": 0.0,
                "mse_log_gnn_t": (yt - pm) ** 2,
                "mse_log_placebo_t": (yt - pm) ** 2,
                "mse_log_har_t": mhar,
                "mse_log_vix_t": (yt - pv) ** 2,
                "d_mse_log_t": (yt - pv) ** 2 - (yt - pm) ** 2,
                "d_mse_log_placebo_t": 0.0,
            }
        )
    for d in test_dates:
        yt, pm, vx = 0.2, 0.21, 19.0
        pv = float(np.log(max(vx, 1e-12)))
        yl, pml = np.exp(yt), np.exp(pm)
        qg = float(qlike_per_date(np.array([yl]), np.array([pml]), eps=1e-12)[0])
        qv = float(qlike_per_date(np.array([yl]), np.array([float(vx)]), eps=1e-12)[0])
        rows.append(
            {
                "date": d,
                "split": "test",
                "sample": "test_full",
                "y_true_log": yt,
                "y_true_level": yl,
                "y_pred_model_log": pm,
                "y_pred_model_level": pml,
                "y_pred_placebo_log": pm,
                "y_pred_placebo_level": pml,
                "y_pred_har_log": pv,
                "y_pred_har_level": float(vx),
                "y_pred_vix_log": pv,
                "y_pred_vix_level": float(vx),
                "vix": float(vx),
                "qlike_gnn_t": qg,
                "qlike_placebo_t": qg,
                "qlike_har_t": qv,
                "qlike_vix_t": qv,
                "d_qlike_t": qv - qg,
                "d_qlike_placebo_t": 0.0,
                "mse_log_gnn_t": (yt - pm) ** 2,
                "mse_log_placebo_t": (yt - pm) ** 2,
                "mse_log_har_t": (yt - pv) ** 2,
                "mse_log_vix_t": (yt - pv) ** 2,
                "d_mse_log_t": (yt - pv) ** 2 - (yt - pm) ** 2,
                "d_mse_log_placebo_t": 0.0,
            }
        )
    p = tmp_path / "forecast_panel.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    assert check_forecast_panel(p)["passed"]


def test_check_forecast_panel_benchmark_diffs_finite_on_test(tmp_path: Path) -> None:
    """v2 benchmark loss differentials must be finite on val/test when present."""
    df = _minimal_forecast_panel_df()
    df["y_pred_calibrated_vix_log"] = df["y_pred_vix_log"]
    df["y_pred_calibrated_vix_level"] = df["y_pred_vix_level"]
    df["y_pred_vix_har_log"] = df["y_pred_vix_log"]
    df["y_pred_vix_har_level"] = df["y_pred_vix_level"]
    df["qlike_calibrated_vix_t"] = df["qlike_vix_t"]
    df["qlike_vix_har_t"] = df["qlike_vix_t"]
    df["mse_log_calibrated_vix_t"] = df["mse_log_vix_t"]
    df["mse_log_vix_har_t"] = df["mse_log_vix_t"]
    df["d_qlike_calibrated_vix_t"] = df["d_qlike_t"]
    df["d_qlike_vix_har_t"] = df["d_qlike_t"]
    df["d_qlike_har_t"] = df["d_qlike_t"]
    df["d_mse_log_calibrated_vix_t"] = df["d_mse_log_t"]
    df["d_mse_log_vix_har_t"] = df["d_mse_log_t"]
    df["d_mse_log_har_t"] = df["d_mse_log_t"]
    p = tmp_path / "forecast_panel.parquet"
    df.to_parquet(p, index=False)
    assert check_forecast_panel(p)["passed"]

    bad = df.copy()
    bad.loc[0, "d_qlike_vix_har_t"] = float("nan")
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    out = check_forecast_panel(tmp_path / "bad.parquet")
    assert not out["passed"]
    assert any("d_qlike_vix_har_t" in i for i in out["issues"])


def test_model_evaluate_semantic_complete(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert not model_evaluate_step_semantically_complete(cfg)
    proc = tmp_path / "processed"
    scoring = proc / "scoring"
    metrics_desc = proc / "metrics" / "descriptive"
    metrics_formal = proc / "metrics" / "formal"
    scoring.mkdir(parents=True)
    metrics_desc.mkdir(parents=True)
    metrics_formal.mkdir(parents=True)
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
    fc.to_parquet(scoring / "forecasts.parquet", index=False)
    _minimal_forecast_panel_df().to_parquet(scoring / "forecast_panel.parquet", index=False)
    assert check_forecast_panel(scoring / "forecast_panel.parquet")["passed"]
    (scoring / "test_loss.json").write_text(
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
    (metrics_desc / "summary_table.csv").write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n"
        "gnn,full_test,0.01,0.02,1.0,1.1,1.2,10\n"
        "vix,full_test,0.02,0.03,1.1,1.2,1.3,10\n",
        encoding="utf-8",
    )
    (metrics_desc / "diagnostics_smoothing.csv").write_text(
        "sample,diagnostic,model,value,detail\n"
        "full_test,forecast_variance_log,gnn,0.01,\n",
        encoding="utf-8",
    )
    pd.DataFrame([_minimal_hypothesis_tests_row()]).to_csv(
        metrics_formal / "hypothesis_tests.csv", index=False
    )
    assert check_hypothesis_tests(metrics_formal / "hypothesis_tests.csv")["passed"]
    assert model_evaluate_step_semantically_complete(cfg)

