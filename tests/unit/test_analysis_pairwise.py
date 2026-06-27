from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mss.analysis.pairwise import bootstrap_ci_loss_diff, diebold_mariano, run_pairwise_report


def test_diebold_mariano_basic() -> None:
    out = diebold_mariano([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert "mean_diff" in out
    assert "p_value_two_sided" in out


def test_bootstrap_ci_loss_diff_basic() -> None:
    out = bootstrap_ci_loss_diff([1.0, 2.0, 3.0], [1.2, 2.1, 3.1], reps=50)
    assert out["reps"] == 50
    assert out["ci_lower"] <= out["ci_upper"]


def test_run_pairwise_report_writes_outputs(tmp_path: Path) -> None:
    fc = tmp_path / "forecasts.parquet"
    pd.DataFrame(
        {
            "sample": ["test_full", "test_full", "train_full", "train_full"],
            "split": ["test", "test", "train", "train"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2010-01-04", "2010-01-05"]),
            "se_log_model": [1.0, 1.1, 0.9, 0.8],
            "se_log_vix": [1.2, 1.3, 1.0, 1.1],
            "ae_log_model": [1.0, 1.0, 1.0, 1.0],
            "ae_log_vix": [1.1, 1.1, 1.1, 1.1],
            "qlike_model_t": [0.5, 0.6, 0.4, 0.5],
            "qlike_vix_t": [0.7, 0.8, 0.6, 0.7],
            "se_level_model": [1.0, 1.0, 1.0, 1.0],
            "se_level_vix": [1.2, 1.2, 1.2, 1.2],
            "ae_level_model": [1.0, 1.0, 1.0, 1.0],
            "ae_level_vix": [1.1, 1.1, 1.1, 1.1],
        }
    ).to_parquet(fc, index=False)
    out_json = tmp_path / "pairwise_comparison.json"
    out_csv = tmp_path / "formal.csv"
    run_pairwise_report(fc, out_json, out_csv, reps=30)
    assert out_json.is_file()
    assert out_csv.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "samples" in payload

