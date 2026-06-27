"""Semantic completeness for pipeline skip/resume (not only file existence)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mss.graph.config import load_graph_config
from mss.io.config import ResolvedConfig
from mss.pipeline import completeness as comp


def _cfg(tmp: Path) -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text(
        "[graph]\nret_col = \"ret_used\"\nalign_feature_dates_with_targets = false\n\n"
        "[graph.rolling_window]\nlength = 1\nmin_obs_frac = 1.0\n\n"
        "[graph.universe]\nn_nodes = 1\n",
        encoding="utf-8",
    )
    return ResolvedConfig(
        run_id="t",
        project_root=tmp,
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        source_config_path=cfg_toml.resolve(),
    )


def _write_rp(path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "permno": [1, 1],
            "permco": [1, 1],
            "shrcd": [10, 10],
            "exchcd": [3, 3],
            "ret_used": [0.01, -0.02],
            "prc": [10.0, 9.0],
            "vol": [1, 1],
            "shrout": [1000, 1000],
            "dlstcd": [None, None],
            "dlret": [None, None],
            "dlretx": [None, None],
            "dlprc": [None, None],
            "dlpdt": [None, None],
            "ticker": ["A", "A"],
            "comnam": ["A", "A"],
            "ncusip": [None, None],
            "cusip": [None, None],
            "secstat": [None, None],
        }
    )
    df.to_parquet(path, index=False)


def test_edges_incomplete_when_fewer_dates_than_universe(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    interim = tmp_path / "interim"
    g = interim / "graphs"
    interim.mkdir(parents=True)
    g.mkdir(parents=True)
    _write_rp(interim / "returns_panel.parquet")

    u = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "permno": [1, 1],
            "rank": [1, 1],
            "mcap": [100.0, 100.0],
        }
    )
    u.to_parquet(g / "universe.parquet", index=False)

    e = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "src": [1],
            "dst": [2],
            "weight": [0.5],
        }
    )
    e.to_parquet(g / "edges.parquet", index=False)

    assert not comp.edges_step_semantically_complete(cfg)


def test_targets_incomplete_when_rowcount_mismatch_manifest(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    interim = tmp_path / "interim"
    interim.mkdir(parents=True)
    tg = interim / "targets.parquet"
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "rv_fwd_30": [0.1],
            "log_rv_fwd_30": [-2.0],
            "vix": [18.0],
        }
    )
    df.to_parquet(tg, index=False)
    man = interim / "targets_manifest.json"
    man.write_text(json.dumps({"stats": {"n_rows": 2}}), encoding="utf-8")
    assert not comp.targets_step_semantically_complete(cfg)


def test_feature_dates_matches_expected(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    interim = tmp_path / "interim"
    interim.mkdir(parents=True)
    _write_rp(interim / "returns_panel.parquet")
    from mss.graph.expected import compute_expected_feature_dates

    expected = compute_expected_feature_dates(
        interim / "returns_panel.parquet", load_graph_config(cfg.source_config_path), None
    )
    pd.DataFrame({"date": expected}).to_parquet(interim / "stage04_dates.parquet", index=False)
    assert comp.feature_dates_step_semantically_complete(cfg)


def test_expected_feature_dates_aligns_with_config(tmp_path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text(
        "[graph]\nret_col = \"ret_used\"\nalign_feature_dates_with_targets = true\n\n"
        "[graph.rolling_window]\nlength = 1\nmin_obs_frac = 1.0\n\n"
        "[graph.universe]\nn_nodes = 1\n",
        encoding="utf-8",
    )
    cfg = ResolvedConfig(
        run_id="t",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        source_config_path=cfg_toml.resolve(),
    )
    interim = tmp_path / "interim"
    interim.mkdir(parents=True)
    _write_rp(interim / "returns_panel.parquet")
    tg = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "rv_fwd_30": [0.1],
            "log_rv_fwd_30": [-2.0],
            "vix": [18.0],
        }
    )
    tg.to_parquet(interim / "targets.parquet", index=False)
    gc = load_graph_config(cfg.source_config_path)
    from mss.graph.expected import compute_expected_feature_dates

    exp = compute_expected_feature_dates(
        interim / "returns_panel.parquet", gc, interim / "targets.parquet"
    )
    assert len(exp) == 1
    pd.DataFrame({"date": exp}).to_parquet(interim / "stage04_dates.parquet", index=False)
    assert comp.feature_dates_step_semantically_complete(cfg)


def test_model_train_step_semantically_complete_delegates(tmp_path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    cfg = ResolvedConfig(
        run_id="t",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        source_config_path=cfg_toml.resolve(),
    )
    assert not comp.model_train_step_semantically_complete(cfg)
    d = tmp_path / "interim" / "model_train"
    d.mkdir(parents=True)
    (d / "final_metrics.json").write_text(
        json.dumps({"status": "completed", "best_epoch": 0, "best_val_loss": 0.5}),
        encoding="utf-8",
    )
    assert comp.model_train_step_semantically_complete(cfg)


def test_model_evaluate_step_semantically_complete_delegates(tmp_path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    cfg = ResolvedConfig(
        run_id="t",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        source_config_path=cfg_toml.resolve(),
    )
    assert not comp.model_evaluate_step_semantically_complete(cfg)
    p = tmp_path / "processed"
    p.mkdir(parents=True)
    fc = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
            "sample": ["test_full"],
            "y_true": [0.1],
            "y_pred_model": [0.2],
        }
    )
    fc.to_parquet(p / "forecasts.parquet", index=False)
    import numpy as np

    from mss.evaluation.metrics import qlike_per_date

    yt, pm, vx = 0.1, 0.11, 18.0
    pv = float(np.log(max(vx, 1e-12)))
    yl = float(np.exp(yt))
    pml = float(np.exp(pm))
    qg = qlike_per_date(np.array([yl]), np.array([pml]), eps=1e-12)
    qv = qlike_per_date(np.array([yl]), np.array([float(vx)]), eps=1e-12)
    dg = float(qv[0] - qg[0])
    mg = (yt - pm) ** 2
    mv = (yt - pv) ** 2
    dm = mv - mg
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
            "sample": ["test_full"],
            "y_true_log": [yt],
            "y_true_level": [yl],
            "y_pred_model_log": [pm],
            "y_pred_model_level": [pml],
            "y_pred_vix_log": [pv],
            "y_pred_vix_level": [float(vx)],
            "vix": [float(vx)],
            "qlike_gnn_t": [float(qg[0])],
            "qlike_vix_t": [float(qv[0])],
            "d_qlike_t": [dg],
            "mse_log_gnn_t": [mg],
            "mse_log_vix_t": [mv],
            "d_mse_log_t": [dm],
        }
    ).to_parquet(p / "forecast_panel.parquet", index=False)
    (p / "test_loss.json").write_text(
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
    summ = p / "summaries"
    summ.mkdir(parents=True, exist_ok=True)
    (summ / "summary_table.csv").write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n"
        "gnn,full_test,0.01,0.02,1.0,1.1,1.2,10\n"
        "vix,full_test,0.02,0.03,1.1,1.2,1.3,10\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
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
                "hac_max_lags": 29,
                "n_obs": 100,
                "rows_removed_vs_full_test": 0,
                "effective_sample_start": "2020-01-02",
                "effective_sample_end": "2020-01-02",
                "stress_rows_removed": 0,
                "year_rows_removed": 0,
                "subsample_operational": True,
                "notes": "fixture",
            }
        ]
    ).to_csv(summ / "hypothesis_tests.csv", index=False)
    assert comp.model_evaluate_step_semantically_complete(cfg)


def test_analysis_summarize_loss_figure_semantically_complete(tmp_path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text(
        '[model.evaluate]\nsplits = ["train", "val", "test"]\n\n[analysis.summarize]\ndpi = 100\n',
        encoding="utf-8",
    )
    cfg = ResolvedConfig(
        run_id="t",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        source_config_path=cfg_toml.resolve(),
    )
    assert not comp.analysis_summarize_loss_figure_semantically_complete(cfg)
    from mss.analysis.figures import expected_paths_for_summarize

    for p in expected_paths_for_summarize(cfg):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert comp.analysis_summarize_loss_figure_semantically_complete(cfg)
