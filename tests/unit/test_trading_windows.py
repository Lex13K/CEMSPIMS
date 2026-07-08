"""Unit tests for mss.calendar.trading_windows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mss.calendar.trading_windows import TradingCalendar, window_dates_sql


def test_window_dates_inclusive_with_gap() -> None:
    # Mon-Fri week, skip Wed (holiday) -> 4 trading days
    dates = pd.bdate_range("2020-01-06", periods=5)  # Mon-Fri
    dates = dates.delete(2)  # remove Wed
    cal = TradingCalendar.from_sorted_dates(dates)

    anchor = pd.Timestamp("2020-01-10")  # Fri
    w3 = cal.window_dates_inclusive(anchor, 3)
    assert len(w3) == 3
    assert w3[-1] == anchor.normalize()
    assert w3[0] == pd.Timestamp("2020-01-07").normalize()  # Tue (Wed holiday skipped)


def test_feature_dates_matches_iloc_semantics() -> None:
    dates = pd.bdate_range("2020-01-02", periods=25)
    cal = TradingCalendar.from_sorted_dates(dates)
    wl = 20
    expected = pd.Series(dates[wl - 1 :])
    got = cal.feature_dates(wl)
    pd.testing.assert_series_equal(
        got.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_pos_of_missing_returns_none() -> None:
    cal = TradingCalendar.from_sorted_dates(pd.bdate_range("2020-01-02", periods=5))
    assert cal.pos_of("2020-01-01") is None


def test_from_parquet_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "rp.parquet"
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=10),
            "permno": [1] * 10,
            "ret_used": [0.01] * 10,
        }
    )
    df.to_parquet(p, index=False)
    cal = TradingCalendar.from_parquet_date_column(p)
    assert len(cal) == 10


def test_window_dates_sql_fragment() -> None:
    sql = window_dates_sql("DATE '2020-01-15'", 20)
    assert "dates_index" in sql
    assert "20" in sql


def test_window_dates_sql_matches_python(tmp_path: Path) -> None:
    """Universe SQL fragment selects the same dates as TradingCalendar."""
    dates = pd.bdate_range("2020-01-02", periods=25)
    p = tmp_path / "rp.parquet"
    pd.DataFrame(
        {"date": dates, "permno": [1] * len(dates), "ret_used": [0.01] * len(dates)}
    ).to_parquet(p, index=False)
    import duckdb

    cal = TradingCalendar.from_parquet_date_column(p)
    anchor = dates[22]
    py_window = set(cal.window_dates_inclusive(anchor, 20))
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE TABLE returns_panel AS SELECT * FROM read_parquet('{p.as_posix()}')"
    )
    con.execute(
        """
        CREATE TEMP TABLE dates_index AS
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS pos
        FROM (SELECT DISTINCT date FROM returns_panel)
        """
    )
    sql = window_dates_sql(f"DATE '{anchor.date()}'", 20)
    sql_dates = set(
        con.execute(f"SELECT date FROM ({sql})").df()["date"].map(lambda d: pd.Timestamp(d).normalize())
    )
    con.close()
    assert sql_dates == py_window


def test_window_length_zero_raises() -> None:
    cal = TradingCalendar.from_sorted_dates(pd.bdate_range("2020-01-02", periods=5))
    with pytest.raises(ValueError, match="window_length"):
        cal.window_dates_inclusive("2020-01-06", 0)
