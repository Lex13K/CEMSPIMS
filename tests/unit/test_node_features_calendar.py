"""Node features use the same trading-calendar windows as universe/edges."""

from __future__ import annotations

import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.node_features import (
    _build_node_features_vectorized,
    compute_node_features_for_date,
)


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-06", periods=8).delete(2)  # skip Wed
    rows = []
    rets = {1: 0.01, 2: 0.02, 3: -0.01}
    for i, d in enumerate(dates):
        for permno, base in rets.items():
            rows.append(
                {
                    "date": d,
                    "permno": permno,
                    "ret_used": base + 0.001 * i,
                    "prc": 100.0,
                    "vol": 1000.0,
                    "shrout": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_per_date_and_vectorized_agree_on_calendar_window(tmp_path) -> None:
    panel = _synthetic_panel()
    p = tmp_path / "rp.parquet"
    panel.to_parquet(p, index=False)
    cal = TradingCalendar.from_sorted_dates(panel["date"].unique())
    anchor = pd.Timestamp("2020-01-13")  # Mon week 2
    permnos = [1, 2, 3]
    per_date = compute_node_features_for_date(
        panel,
        permnos,
        anchor,
        window_length=3,
        calendar=cal,
    )
    universe = pd.DataFrame({"date": [anchor] * len(permnos), "permno": permnos})
    vectorized = _build_node_features_vectorized(
        p,
        universe,
        window_length=3,
        ret_col="ret_used",
        rolling_mean=True,
        rolling_vol=True,
        include_log_dollar_volume=False,
        include_turnover=False,
        min_periods=1,
        verbose=False,
        show_progress=False,
    )
    pd.testing.assert_frame_equal(
        per_date.sort_values("permno").reset_index(drop=True),
        vectorized.sort_values("permno").reset_index(drop=True),
        check_dtype=False,
    )


def test_window_excludes_timedelta_extra_dates() -> None:
    panel = _synthetic_panel()
    cal = TradingCalendar.from_sorted_dates(panel["date"].unique())
    anchor = pd.Timestamp("2020-01-10")
    window_dates = cal.window_dates_inclusive(anchor, 3)
    assert pd.Timestamp("2020-01-06") not in window_dates
