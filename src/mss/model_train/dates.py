"""Date/label helpers shared between GraphDateDataset and cache materialization.

This module must remain torch-free so completion/step checks can run without the
train extras installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mss.model_train.labels import join_labels


def _normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


def split_unfiltered_dates(manifest: dict[str, Any], split: str) -> list[pd.Timestamp]:
    """All dates assigned to `split` in `splits.parquet` (normalized to midnight)."""
    splits_path = Path(manifest["splits_path"])
    sp = pd.read_parquet(splits_path)
    sp["date"] = _normalize_date_series(sp["date"])
    dates = sp[sp["split"] == split]["date"].sort_values().tolist()
    return dates


def split_filtered_dates_and_labels_df(
    manifest: dict[str, Any], split: str
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Filtered dates (non-NaN labels) plus labels_df to use for lookups."""
    dates_unfiltered = split_unfiltered_dates(manifest, split)
    targets_path = Path(manifest["targets_path"])
    labels_df = join_labels(
        dates_unfiltered,
        targets_path,
        target_column=str(manifest["target_column"]),
    )
    merged = pd.DataFrame({"date": dates_unfiltered}).merge(
        labels_df, on="date", how="left"
    )
    dates_filtered = merged[merged["y_t"].notna()]["date"].tolist()
    return dates_filtered, labels_df


def train_val_expected_dates_and_labels_df(
    manifest: dict[str, Any],
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Union of expected train+val dates (non-NaN labels) plus labels_df."""
    train_unfiltered = split_unfiltered_dates(manifest, "train")
    val_unfiltered = split_unfiltered_dates(manifest, "val")
    all_unfiltered = sorted(set(train_unfiltered).union(val_unfiltered))

    targets_path = Path(manifest["targets_path"])
    labels_df = join_labels(
        all_unfiltered,
        targets_path,
        target_column=str(manifest["target_column"]),
    )

    train_expected = (
        pd.DataFrame({"date": train_unfiltered})
        .merge(labels_df, on="date", how="left")
        .query("y_t.notna()")["date"]
        .tolist()
    )
    val_expected = (
        pd.DataFrame({"date": val_unfiltered})
        .merge(labels_df, on="date", how="left")
        .query("y_t.notna()")["date"]
        .tolist()
    )

    expected_dates = sorted(set(train_expected).union(val_expected))
    return expected_dates, labels_df

