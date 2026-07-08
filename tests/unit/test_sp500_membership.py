"""Unit tests for the optional S&P 500 membership normalizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.data.ingest import normalize_sp500_membership


def test_normalize_spans_from_start_end(tmp_path: Path) -> None:
    raw = tmp_path / "sp500_membership.csv"
    pd.DataFrame(
        {
            "permno": [10001, 10002],
            "start": ["2010-01-01", "2015-06-01"],
            "ending": ["2020-12-31", ""],  # not detected; only 'end'-like names
            "end_date": ["2020-12-31", ""],
        }
    ).to_csv(raw, index=False)
    out = tmp_path / "spans.parquet"
    normalize_sp500_membership(raw, out, overwrite=True)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["permno", "start_date", "end_date"]
    assert set(df["permno"]) == {10001, 10002}
    assert str(df.loc[df["permno"] == 10001, "start_date"].iloc[0]) == "2010-01-01"


def test_normalize_spans_from_date_panel(tmp_path: Path) -> None:
    raw = tmp_path / "sp500_constituents.csv"
    dates = pd.bdate_range("2018-01-02", periods=5)
    rows = []
    for d in dates:
        for permno in (1, 2):
            rows.append({"date": d.date(), "permno": permno})
    pd.DataFrame(rows).to_csv(raw, index=False)
    out = tmp_path / "spans.parquet"
    normalize_sp500_membership(raw, out, overwrite=True)
    df = pd.read_parquet(out).sort_values("permno").reset_index(drop=True)
    assert set(df["permno"]) == {1, 2}
    # Collapsed to contiguous [min, max] span per permno.
    assert str(df.loc[0, "start_date"]) == "2018-01-02"
