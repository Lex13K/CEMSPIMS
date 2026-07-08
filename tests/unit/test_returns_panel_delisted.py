"""Delisting-adjusted return column when enabled in returns panel build."""

from __future__ import annotations

import duckdb
import pandas as pd

from mss.data.returns_panel import build_returns_panel


def _panel_row(date: str, permno: int, *, retx: float, dlstcd=None, dlret=None) -> dict:
    return {
        "date": pd.Timestamp(date),
        "permno": permno,
        "permco": 20001,
        "shrcd": 10,
        "exchcd": 3,
        "ret": retx,
        "retx": retx,
        "prc": 100.0,
        "vol": 1000.0,
        "shrout": 1_000_000,
        "dlstcd": dlstcd,
        "dlret": dlret,
        "dlretx": dlret,
        "dlprc": None,
        "dlpdt": None,
        "ticker": "ACME",
        "comnam": "Acme Inc",
        "ncusip": None,
        "cusip": None,
        "secstat": None,
    }


def test_ret_delisted_compounds_dlret_on_delist_date(tmp_path) -> None:
    crsp = tmp_path / "crsp_parquet" / "year=2020"
    crsp.mkdir(parents=True)
    df = pd.DataFrame(
        [
            _panel_row("2020-01-02", 10001, retx=0.01),
            _panel_row("2020-01-03", 10001, retx=0.02, dlstcd=500, dlret=-0.30),
        ]
    )
    df.to_parquet(crsp / "data.parquet", index=False)
    out = tmp_path / "returns_panel.parquet"
    build_returns_panel(crsp.parent, out, apply_delisting_adjustment=True, overwrite=False)

    con = duckdb.connect(database=":memory:")
    row = con.execute(
        f"SELECT ret_used, ret_delisted FROM read_parquet('{out.as_posix()}') "
        "WHERE dlstcd IS NOT NULL"
    ).fetchone()
    con.close()
    assert row is not None
    ret_used, ret_delisted = row
    expected = (1.0 + ret_used) * (1.0 + (-0.30)) - 1.0
    assert abs(ret_delisted - expected) < 1e-12


def test_ret_delisted_null_when_flag_off(tmp_path) -> None:
    crsp = tmp_path / "crsp_parquet" / "year=2020"
    crsp.mkdir(parents=True)
    df = pd.DataFrame([_panel_row("2020-01-02", 10001, retx=0.01)])
    df.to_parquet(crsp / "data.parquet", index=False)
    out = tmp_path / "returns_panel.parquet"
    build_returns_panel(crsp.parent, out, apply_delisting_adjustment=False, overwrite=False)
    con = duckdb.connect(database=":memory:")
    n_null = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}') WHERE ret_delisted IS NULL"
    ).fetchone()[0]
    con.close()
    assert n_null == 1
