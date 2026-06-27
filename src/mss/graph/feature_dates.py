"""Feature-date index from returns panel (optional alignment with targets spine)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.graph.config import GraphConfig
from mss.graph.expected import compute_expected_feature_dates


def build_feature_date_index(
    returns_panel_path: Path,
    out_path: Path,
    *,
    gc: GraphConfig,
    targets_parquet_path: Path | None,
    overwrite: bool = False,
) -> Path:
    """
    Valid feature dates = dates with a full rolling window of history in returns_panel.
    Writes a single-column parquet: date.
    When gc.align_feature_dates_with_targets and targets_parquet_path is set,
    intersects with dates present in targets.
    """
    if not returns_panel_path.is_file():
        raise FileNotFoundError(f"Returns panel not found: {returns_panel_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use overwrite=True.")

    expected = compute_expected_feature_dates(returns_panel_path, gc, targets_parquet_path)
    dates_df = pd.DataFrame({"date": expected})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dates_df.to_parquet(out_path, index=False)
    return out_path
