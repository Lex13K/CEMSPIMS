"""Edges use trading-calendar windows (not calendar Timedelta buffers)."""

from __future__ import annotations

import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.edges import compute_edges_for_date


def _synthetic_panel() -> pd.DataFrame:
    # Mon-Fri minus Wed holiday -> 4 trading days
    dates = pd.bdate_range("2020-01-06", periods=5).delete(2)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "permno": 1, "ret_used": 0.01 * (i + 1)})
        rows.append({"date": d, "permno": 2, "ret_used": 0.02 * (i + 1) + 0.001})
    return pd.DataFrame(rows)


def test_compute_edges_uses_calendar_window_not_timedelta() -> None:
    panel = _synthetic_panel()
    cal = TradingCalendar.from_sorted_dates(panel["date"].unique())
    anchor = pd.Timestamp("2020-01-10")  # Fri
    window_dates = set(cal.window_dates_inclusive(anchor, 3))

    # Timedelta buffer would include Mon 1/6; calendar window is Tue-Thu-Fri only.
    assert pd.Timestamp("2020-01-06") not in window_dates
    assert len(window_dates) == 3

    edges = compute_edges_for_date(
        panel,
        [1, 2],
        anchor,
        window_length=3,
        top_k=1,
        calendar=cal,
    )
    assert not edges.empty
    assert edges["date"].nunique() == 1
    assert pd.to_datetime(edges["date"].iloc[0]).normalize() == anchor.normalize()
