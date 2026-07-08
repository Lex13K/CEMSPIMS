"""Read-only expected feature dates; must stay aligned with `feature_dates.build_feature_date_index`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.config import GraphConfig


def compute_expected_feature_dates(
    returns_panel_path: Path,
    gc: GraphConfig,
    targets_parquet_path: Path | None,
) -> pd.Series:
    """
    Same date logic as `build_feature_date_index` without writing files.
    Returns sorted unique dates as datetime64[ns].
    """
    if not returns_panel_path.is_file():
        raise FileNotFoundError(f"Returns panel not found: {returns_panel_path}")

    cal = TradingCalendar.from_parquet_date_column(returns_panel_path)
    feature_dates = cal.feature_dates(gc.window_length)
    dates_df = pd.DataFrame({"date": feature_dates})

    if gc.align_feature_dates_with_targets and targets_parquet_path is not None:
        if not targets_parquet_path.is_file():
            raise FileNotFoundError(f"Targets not found: {targets_parquet_path}")
        tg = pd.read_parquet(targets_parquet_path, columns=["date"])
        tg["date"] = pd.to_datetime(tg["date"])
        dates_df["date"] = pd.to_datetime(dates_df["date"])
        dates_df = dates_df.merge(tg[["date"]].drop_duplicates(), on="date", how="inner")
        dates_df = dates_df.sort_values("date").reset_index(drop=True)

    out = pd.to_datetime(dates_df["date"])
    return out.reset_index(drop=True)


def normalized_date_set(series: pd.Series) -> set[pd.Timestamp]:
    """Compare dates as normalized timestamps (ns, UTC-naive)."""
    s = pd.to_datetime(series)
    return set(s.dt.normalize())
