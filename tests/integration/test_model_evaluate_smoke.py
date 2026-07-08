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

    train_dates = pd.bdate_range("2020-01-02", periods=22)
    val_dates = pd.bdate_range("2020-02-03", periods=2)
    test_dates = pd.bdate_range("2020-02-06", periods=2)
    all_dates = list(train_dates) + list(val_dates) + list(test_dates)

    splits = pd.DataFrame(
        {
            "date": list(train_dates) + list(val_dates) + list(test_dates),
            "split": ["train"] * len(train_dates) + ["val"] * len(val_dates) + ["test"] * len(test_dates),
        }
    )
    n = len(all_dates)
    targets = pd.DataFrame(
        {
            "date": all_dates,
            "log_rv_fwd_30cal": [0.10 + 0.01 * i for i in range(n)],
            "log_rv_lag_5": [0.11 + 0.01 * i for i in range(n)],
            "log_rv_lag_30": [0.12 + 0.01 * i for i in range(n)],
            "log_rv_lag_252": [0.13 + 0.01 * i for i in range(n)],
            "vix": [18.0 + 0.1 * i for i in range(n)],
        }
    )
    universe_rows = []
    nf_rows = []
    edge_rows = []
    for i, d in enumerate(all_dates):
        for permno in (1, 2):
            universe_rows.append(
                {"date": d, "permno": permno, "rank": permno, "mcap": 100.0 + permno + 0.1 * i}
            )
            nf_rows.append(
                {
                    "date": d,
                    "permno": permno,
                    "rolling_mean": 0.1 * permno + 0.01 * i,
                    "rolling_vol": 0.9 + 0.01 * permno + 0.005 * i,
                }
            )
            edge_rows.append(
                {
                    "date": d,
                    "src": permno,
                    "dst": 3 - permno,
                    "weight": 0.5 + 0.01 * i,
                }
            )
    universe = pd.DataFrame(universe_rows)
    node_features = pd.DataFrame(nf_rows)
    edges = pd.DataFrame(edge_rows)
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
        "target_column": "log_rv_fwd_30cal",
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
    f = processed / "scoring" / "forecasts.parquet"
    t = processed / "scoring" / "test_loss.json"
    figdir = processed / "figures"
    assert f.is_file()
    assert t.is_file()
    assert (figdir / "diagnostics" / "loss_over_epochs_linear.png").is_file()
    assert (figdir / "diagnostics" / "loss_over_epochs_log.png").is_file()
    assert (figdir / "by_split" / "all" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "train" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "val" / "forecast_vs_benchmarks.png").is_file()
    assert (figdir / "by_split" / "test" / "forecast_vs_benchmarks.png").is_file()
    assert (processed / "metrics" / "descriptive" / "summary_table.csv").is_file()
    assert (processed / "metrics" / "descriptive" / "diagnostics_smoothing.csv").is_file()
    assert (processed / "scoring" / "forecast_panel.parquet").is_file()
    assert (processed / "metrics" / "formal" / "hypothesis_tests.csv").is_file()
    assert (processed / "metrics" / "formal" / "regression_mz_gnn.csv").is_file()
    assert (processed / "metrics" / "formal" / "regression_incremental.csv").is_file()
    assert (figdir / "diagnostics" / "summary_barplot.png").is_file()

    uc = figdir / "diagnostics" / "universe_churn"
    assert (uc / "universe_turnover_timeseries.png").is_file()
    assert (uc / "universe_rankbucket_replacement_heatmap.png").is_file()
    assert (uc / "universe_tenure_distribution.png").is_file()

    univ = processed / "metrics" / "universe"
    assert (univ / "universe_turnover_timeseries.csv").is_file()
    assert (univ / "universe_rankbucket_replacement_long.csv").is_file()
    assert (univ / "universe_tenure_distribution.csv").is_file()
    assert (univ / "universe_churn_summary.csv").is_file()
    df = pd.read_parquet(f)
    assert {"date", "split", "sample", "y_true", "y_pred_model", "y_pred_placebo"}.issubset(
        df.columns
    )
    payload = json.loads(t.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metric"] == "mse_log"

