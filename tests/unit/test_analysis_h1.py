from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.analysis.h1 import run_h1_regression_table


def test_run_h1_regression_table_writes_csv(tmp_path: Path) -> None:
    n = 40
    fc = tmp_path / "forecasts.parquet"
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 20 + ["val"] * 10 + ["test"] * 10,
            "sample": ["train_full"] * 20 + ["val_full"] * 10 + ["test_full"] * 10,
            "y_true": [0.01 * i for i in range(n)],
            "y_pred_model": [0.01 * i + 0.1 for i in range(n)],
            "y_pred_vix": [0.01 * i + 0.2 for i in range(n)],
        }
    ).to_parquet(fc, index=False)
    out = tmp_path / "h1_regression.csv"
    run_h1_regression_table(fc, out)
    assert out.is_file()
    df = pd.read_csv(out)
    assert {"variable", "coef", "se", "t", "pvalue"}.issubset(df.columns)

