"""Partitioned returns reads for parallel edge builds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.edges import _read_returns_for_chunk, compute_edges_for_date


def test_read_returns_for_chunk_subset(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=30)
    rows = []
    for d in dates:
        for permno in (1, 2, 3):
            rows.append({"date": d, "permno": permno, "ret_used": 0.01 * permno})
    p = tmp_path / "rp.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    cal = TradingCalendar.from_parquet_date_column(p)
    chunk = [dates[20], dates[21]]
    got = _read_returns_for_chunk(
        str(p),
        chunk,
        ret_col="ret_used",
        window_length=5,
        calendar=cal,
    )
    full = pd.read_parquet(p)
    full["date"] = pd.to_datetime(full["date"])
    window_dates: set[pd.Timestamp] = set()
    for d in chunk:
        window_dates.update(cal.window_dates_inclusive(d, 5))
    expected = full[full["date"].isin(window_dates)]
    assert len(got) == len(expected)
    assert len(got) < len(full)


def test_partitioned_read_matches_full_for_edges(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=15)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "permno": 1, "ret_used": 0.01 * (i + 1)})
        rows.append({"date": d, "permno": 2, "ret_used": 0.02 * (i + 1) + 0.001})
    p = tmp_path / "rp.parquet"
    panel = pd.DataFrame(rows)
    panel.to_parquet(p, index=False)
    cal = TradingCalendar.from_parquet_date_column(p)
    anchor = dates[-1]
    full_edges = compute_edges_for_date(
        panel,
        [1, 2],
        anchor,
        window_length=5,
        calendar=cal,
    )
    part_panel = _read_returns_for_chunk(
        str(p),
        [anchor],
        ret_col="ret_used",
        window_length=5,
        calendar=cal,
    )
    part_edges = compute_edges_for_date(
        part_panel,
        [1, 2],
        anchor,
        window_length=5,
        calendar=cal,
    )
    pd.testing.assert_frame_equal(
        full_edges.sort_values(["src", "dst"]).reset_index(drop=True),
        part_edges.sort_values(["src", "dst"]).reset_index(drop=True),
        check_dtype=False,
    )
