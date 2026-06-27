"""Train/val/test date assignments for dataset packaging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_split_assignment(
    dates: pd.Series,
    train_end: str,
    val_end: str,
    test_end: str,
) -> pd.DataFrame:
    """
    Assign each date to train, val, or test by inclusive end boundaries.
    train: date <= train_end
    val: train_end < date <= val_end
    test: val_end < date <= test_end
    Dates outside (min, test_end] are dropped.
    Returns DataFrame with columns: date, split.
    """
    dates = pd.Series(pd.to_datetime(dates).drop_duplicates().sort_values())
    train_end_d = pd.Timestamp(train_end)
    val_end_d = pd.Timestamp(val_end)
    test_end_d = pd.Timestamp(test_end)
    if not (train_end_d <= val_end_d <= test_end_d):
        raise ValueError("Splits must be chronological: train_end <= val_end <= test_end")

    def assign(d: pd.Timestamp) -> str:
        if d <= train_end_d:
            return "train"
        if d <= val_end_d:
            return "val"
        if d <= test_end_d:
            return "test"
        return "drop"

    df = pd.DataFrame({"date": dates})
    df["split"] = df["date"].apply(assign)
    df = df[df["split"] != "drop"].reset_index(drop=True)
    return df[["date", "split"]]


def write_splits_parquet(
    universe_path: Path,
    out_path: Path,
    *,
    train_end: str,
    val_end: str,
    test_end: str,
) -> Path:
    """Read distinct dates from universe, assign splits, write Parquet."""
    if not universe_path.is_file():
        raise FileNotFoundError(f"Universe not found: {universe_path}")
    ud = pd.read_parquet(universe_path, columns=["date"])
    dates = ud["date"].drop_duplicates().sort_values()
    df = build_split_assignment(dates, train_end, val_end, test_end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
