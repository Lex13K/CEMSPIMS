from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mss.cli import main

pytest.importorskip("torch_geometric")


def _write_packaged_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    interim = tmp_path / "interim"
    ds_dir = interim / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    d_train = pd.Timestamp("2020-01-02")
    d_val = pd.Timestamp("2020-01-03")
    d_test = pd.Timestamp("2020-01-06")

    splits = pd.DataFrame(
        {"date": [d_train, d_val, d_test], "split": ["train", "val", "test"]}
    )
    targets = pd.DataFrame(
        {
            "date": [d_train, d_val, d_test],
            "log_rv_fwd_30": [0.10, 0.20, 0.30],
            "vix": [18.0, 19.0, 20.0],
        }
    )
    universe = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val, d_test, d_test],
            "permno": [1, 2, 1, 2, 1, 2],
            "rank": [1, 2, 1, 2, 1, 2],
            "mcap": [100.0, 99.0, 101.0, 98.0, 102.0, 97.0],
        }
    )
    node_features = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val, d_test, d_test],
            "permno": [1, 2, 1, 2, 1, 2],
            "rolling_mean": [0.1, 0.2, 0.15, 0.25, 0.05, 0.12],
            "rolling_vol": [0.9, 1.0, 0.85, 1.1, 0.8, 0.95],
        }
    )
    edges = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val, d_test, d_test],
            "src": [1, 2, 1, 2, 1, 2],
            "dst": [2, 1, 2, 1, 2, 1],
            "weight": [0.5, 0.5, 0.6, 0.6, 0.4, 0.4],
        }
    )
    scaler = {
        "scaler_type": "standard",
        "feature_columns": ["rolling_mean", "rolling_vol"],
        "params": {
            "rolling_mean": {"mean": 0.0, "scale": 1.0},
            "rolling_vol": {"mean": 0.0, "scale": 1.0},
        },
    }

    splits_path = ds_dir / "splits.parquet"
    targets_path = interim / "targets.parquet"
    gdir = interim / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    universe_path = gdir / "universe.parquet"
    node_features_path = gdir / "node_features.parquet"
    edges_path = gdir / "edges.parquet"
    scaler_path = ds_dir / "scaler_params.json"
    manifest_path = ds_dir / "manifest.json"

    splits.to_parquet(splits_path, index=False)
    targets.to_parquet(targets_path, index=False)
    universe.to_parquet(universe_path, index=False)
    node_features.to_parquet(node_features_path, index=False)
    edges.to_parquet(edges_path, index=False)
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    manifest = {
        "splits_path": str(splits_path.resolve()),
        "node_features_path": str(node_features_path.resolve()),
        "edges_path": str(edges_path.resolve()),
        "universe_path": str(universe_path.resolve()),
        "scaler_path": str(scaler_path.resolve()),
        "targets_path": str(targets_path.resolve()),
        "target_column": "log_rv_fwd_30",
        "feature_columns": ["rolling_mean", "rolling_vol"],
        "scale_target": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return interim, tmp_path / "processed", manifest_path


@pytest.mark.train
def test_model_evaluate_pipeline_writes_outputs(tmp_path: Path) -> None:
    interim, processed, _ = _write_packaged_dataset(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "evalsmoke.toml"
    cfg.write_text(
        f'[paths]\nraw = "{(tmp_path / "raw").as_posix()}"\n'
        f'interim = "{interim.as_posix()}"\n'
        f'processed = "{processed.as_posix()}"\n'
        "\n[model.train]\n"
        "seed = 42\nmax_epochs = 2\nlr = 0.001\nearly_stopping_patience = 2\n"
        "hidden_channels = 16\nnum_gnn_layers = 2\nmodel_type = \"graphsage\"\n"
        "pooling = \"mean\"\ndataloader_num_workers = 0\nbatch_size = 1\n"
        "show_progress = false\nverbose = false\ndevice = \"cpu\"\npin_memory = false\n"
        "parquet_date_pushdown = true\n"
        "\n[model.evaluate]\n"
        "splits = [\"train\", \"val\", \"test\"]\n"
        "batch_size = 1\ndevice = \"cpu\"\n"
        "verbose = false\n"
        "\n[analysis.summarize]\n"
        "verbose = false\ndpi = 100\n",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--configs-dir",
            str(configs_dir),
            "--run",
            "evalsmoke",
            "--pipeline",
            "model.train",
            "--pipeline",
            "model.evaluate",
            "--pipeline",
            "analysis.summarize",
        ]
    )
    assert code == 0
    f = processed / "forecasts.parquet"
    t = processed / "test_loss.json"
    figdir = processed / "figures"
    fv = figdir / "target_vs_model_vix"
    lv = figdir / "loss_over_epochs"
    assert f.is_file()
    assert t.is_file()
    assert (lv / "loss_over_epochs_with_test_line_linear.png").is_file()
    assert (lv / "loss_over_epochs_with_test_line_log.png").is_file()
    assert (fv / "target_vs_model_vix_all_splits.png").is_file()
    assert (fv / "target_vs_model_vix_train.png").is_file()
    assert (fv / "target_vs_model_vix_val.png").is_file()
    assert (fv / "target_vs_model_vix_test.png").is_file()
    assert (processed / "summaries" / "summary_table.csv").is_file()
    assert (processed / "summaries" / "diagnostics_smoothing.csv").is_file()
    assert (processed / "forecast_panel.parquet").is_file()
    assert (processed / "summaries" / "hypothesis_tests.csv").is_file()
    assert (processed / "summaries" / "regression_mz_gnn.csv").is_file()
    assert (processed / "summaries" / "regression_incremental.csv").is_file()
    assert (figdir / "general" / "summary_barplot.png").is_file()

    uc = figdir / "universe_churn"
    assert (uc / "universe_turnover_timeseries.png").is_file()
    assert (uc / "universe_rankbucket_replacement_heatmap.png").is_file()
    assert (uc / "universe_tenure_distribution.png").is_file()

    summ = processed / "summaries"
    assert (summ / "universe_turnover_timeseries.csv").is_file()
    assert (summ / "universe_rankbucket_replacement_long.csv").is_file()
    assert (summ / "universe_tenure_distribution.csv").is_file()
    assert (summ / "universe_churn_summary.csv").is_file()
    df = pd.read_parquet(f)
    assert {"date", "split", "sample", "y_true", "y_pred_model", "y_pred_placebo"}.issubset(
        df.columns
    )
    payload = json.loads(t.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metric"] == "mse_log"

