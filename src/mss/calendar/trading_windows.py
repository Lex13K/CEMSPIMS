"""Trading-day window indexing shared by graph, targets audit, and universe SQL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


def _normalize_dates(dates: pd.Series | list) -> pd.Series:
    s = pd.to_datetime(pd.Series(dates)).dt.normalize().drop_duplicates().sort_values()
    return s.reset_index(drop=True)


@dataclass(frozen=True)
class TradingCalendar:
    """1-based positions on a sorted distinct trading-date spine."""

    dates: tuple[pd.Timestamp, ...]

    @classmethod
    def from_sorted_dates(cls, dates: pd.Series | list) -> TradingCalendar:
        s = _normalize_dates(dates)
        if len(s) == 0:
            raise ValueError("TradingCalendar requires at least one date")
        return cls(dates=tuple(pd.Timestamp(d) for d in s))

    @classmethod
    def from_parquet_date_column(cls, parquet_path: Path, *, date_column: str = "date") -> TradingCalendar:
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        con = duckdb.connect(database=":memory:")
        pq = parquet_path.resolve().as_posix().replace("'", "''")
        rows = con.execute(
            f"SELECT DISTINCT {date_column} AS date FROM read_parquet('{pq}') ORDER BY date"
        ).fetchall()
        con.close()
        return cls.from_sorted_dates([r[0] for r in rows])

    def __len__(self) -> int:
        return len(self.dates)

    def pos_of(self, anchor: pd.Timestamp | str) -> int | None:
        """1-based position of anchor on the calendar, or None if missing."""
        a = pd.Timestamp(anchor).normalize()
        for i, d in enumerate(self.dates, start=1):
            if d == a:
                return i
        return None

    @staticmethod
    def window_start_pos(anchor_pos: int, window_length: int) -> int:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        return max(1, anchor_pos - window_length + 1)

    def window_dates_inclusive(
        self,
        anchor: pd.Timestamp | str,
        window_length: int,
    ) -> list[pd.Timestamp]:
        """Trading dates in [anchor_pos - wl + 1, anchor_pos] (inclusive)."""
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        pos = self.pos_of(anchor)
        if pos is None:
            return []
        start = self.window_start_pos(pos, window_length)
        return list(self.dates[start - 1 : pos])

    def feature_dates(self, window_length: int) -> pd.Series:
        """Dates with a full `window_length` history on the trading calendar."""
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if len(self.dates) < window_length:
            raise RuntimeError(
                f"Not enough dates to form a rolling window (have {len(self.dates)}, need > {window_length})."
            )
        return pd.Series(self.dates[window_length - 1 :], dtype="datetime64[ns]")


def load_trading_calendar_from_parquet(parquet_path: Path, *, date_column: str = "date") -> TradingCalendar:
    return TradingCalendar.from_parquet_date_column(parquet_path, date_column=date_column)


def window_dates_sql(anchor_date_sql: str, window_length: int) -> str:
    """
    DuckDB subquery selecting window dates ending at ``anchor_date_sql``.

    Requires ``dates_index(date, pos)`` temp table (same as universe.py).
    """
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    return f"""
        SELECT di.date FROM dates_index di
        WHERE di.pos BETWEEN
            GREATEST(1, (SELECT pos FROM dates_index WHERE date = {anchor_date_sql} LIMIT 1) - {window_length} + 1)
            AND (SELECT pos FROM dates_index WHERE date = {anchor_date_sql} LIMIT 1)
    """
