from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mss.cli import main

pytest.importorskip("torch_geometric")


def _write_packaged_dataset(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    ds_dir = interim / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    d_train = pd.Timestamp("2020-01-02")
    d_val = pd.Timestamp("2020-01-03")
    d_test = pd.Timestamp("2020-01-06")

    splits = pd.DataFrame({"date": [d_train, d_val, d_test], "split": ["train", "val", "test"]})
    targets = pd.DataFrame({"date": [d_train, d_val, d_test], "log_rv_fwd_30cal": [0.1, 0.2, 0.3], "vix": [18.0, 19.0, 20.0]})
    universe = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val, d_test, d_test],
            "permno": [1, 2, 1, 2, 1, 2],
            "rank": [1, 2, 1, 2, 1, 2],
            "mcap": [100.0, 99.0, 101.0, 98.0, 102.0, 97.0],
        }
    )
    nf = pd.DataFrame(
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
        "params": {"rolling_mean": {"mean": 0.0, "scale": 1.0}, "rolling_vol": {"mean": 0.0, "scale": 1.0}},
    }
    gdir = interim / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    splits_path = ds_dir / "splits.parquet"
    targets_path = interim / "targets.parquet"
    scaler_path = ds_dir / "scaler_params.json"
    manifest_path = ds_dir / "manifest.json"
    splits.to_parquet(splits_path, index=False)
    targets.to_parquet(targets_path, index=False)
    universe.to_parquet(gdir / "universe.parquet", index=False)
    nf.to_parquet(gdir / "node_features.parquet", index=False)
    edges.to_parquet(gdir / "edges.parquet", index=False)
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    manifest = {
        "splits_path": str(splits_path.resolve()),
        "node_features_path": str((gdir / "node_features.parquet").resolve()),
        "edges_path": str((gdir / "edges.parquet").resolve()),
        "universe_path": str((gdir / "universe.parquet").resolve()),
        "scaler_path": str(scaler_path.resolve()),
        "targets_path": str(targets_path.resolve()),
        "target_column": "log_rv_fwd_30cal",
        "feature_columns": ["rolling_mean", "rolling_vol"],
        "scale_target": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return interim, tmp_path / "processed"


@pytest.mark.train
def test_analysis_figures_pipeline_writes_pack(tmp_path: Path) -> None:
    interim, processed = _write_packaged_dataset(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "figsmoke.toml"
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
            "figsmoke",
            "--pipeline",
            "model.train",
            "--pipeline",
            "model.evaluate",
            "--pipeline",
            "analysis.summarize",
        ]
    )
    assert code == 0
    figdir = processed / "figures"
    assert (figdir / "diagnostics" / "loss_over_epochs_linear.png").is_file()
    assert (figdir / "diagnostics" / "loss_over_epochs_log.png").is_file()
    assert (figdir / "by_split" / "all" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "train" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "val" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "test" / "forecast_vs_benchmarks.png").is_file()
    assert (processed / "scoring" / "test_loss.json").is_file()
    assert (processed / "metrics" / "descriptive" / "summary_table.csv").is_file()
    assert (processed / "metrics" / "descriptive" / "diagnostics_smoothing.csv").is_file()
    assert (processed / "metrics" / "formal" / "hypothesis_tests.csv").is_file()
    assert (processed / "scoring" / "forecast_panel.parquet").is_file()
    assert (figdir / "diagnostics" / "summary_barplot.png").is_file()
    assert (figdir / "diagnostics" / "hypothesis_tests_table.png").is_file()

    uc = figdir / "diagnostics" / "universe_churn"
    assert (uc / "universe_turnover_timeseries.png").is_file()
    assert (uc / "universe_rankbucket_replacement_heatmap.png").is_file()
    assert (uc / "universe_tenure_distribution.png").is_file()

    univ = processed / "metrics" / "universe"
    assert (univ / "universe_turnover_timeseries.csv").is_file()
    assert (univ / "universe_rankbucket_replacement_long.csv").is_file()
    assert (univ / "universe_tenure_distribution.csv").is_file()
    assert (univ / "universe_churn_summary.csv").is_file()

