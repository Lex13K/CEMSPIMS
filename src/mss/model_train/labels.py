"""Join graph dates with targets table for supervised labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def join_labels(
    graph_dates: pd.Series | list,
    targets_parquet_path: Path,
    *,
    target_column: str = "log_rv_fwd_30cal",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge target column from `targets.parquet` onto graph dates.
    Returns DataFrame with columns `date`, `y_t` (renamed from target_column), and optional extras.
    """
    if not targets_parquet_path.is_file():
        raise FileNotFoundError(f"Targets not found: {targets_parquet_path}")

    graph_dates = pd.Series(graph_dates).drop_duplicates().sort_values()
    cols = ["date", target_column]
    if extra_columns:
        for c in extra_columns:
            if c not in cols:
                cols.append(c)
    spine = pd.read_parquet(targets_parquet_path, columns=cols)
    spine["date"] = pd.to_datetime(spine["date"])
    df = pd.DataFrame({"date": pd.to_datetime(graph_dates)})
    df = df.merge(spine, on="date", how="left")
    df = df.rename(columns={target_column: "y_t"})
    return df
