from __future__ import annotations

import pandas as pd

from mss.data.returns_panel import build_returns_panel, check_returns_panel


def _panel_row(
    date: str,
    permno: int,
    *,
    prc: float = 100.0,
    vol: float = 1000.0,
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "permno": permno,
        "permco": 20001,
        "shrcd": 10,
        "exchcd": 3,
        "ret": 0.01,
        "retx": 0.01,
        "prc": prc,
        "vol": vol,
        "shrout": 1_000_000,
        "dlstcd": None,
        "dlret": None,
        "dlretx": None,
        "dlprc": None,
        "dlpdt": None,
        "ticker": "ACME",
        "comnam": "Acme Inc",
        "ncusip": None,
        "cusip": None,
        "secstat": None,
    }


def test_build_and_check_returns_panel_ok(tmp_path) -> None:
    crsp = tmp_path / "crsp_parquet" / "year=2020"
    crsp.mkdir(parents=True)
    df = pd.DataFrame(
        [
            _panel_row("2020-01-02", 10001),
            _panel_row("2020-01-03", 10001, prc=98.0, vol=1100.0),
        ]
    )
    df.to_parquet(crsp / "data.parquet", index=False)

    out = tmp_path / "returns_panel.parquet"
    build_returns_panel(crsp.parent, out, overwrite=False)
    result = check_returns_panel(out)
    assert result["passed"]
    assert result["duplicates"]["n_duplicate_pairs"] == 0


def test_check_returns_panel_detects_duplicate_keys(tmp_path) -> None:
    crsp = tmp_path / "crsp_parquet" / "year=2020"
    crsp.mkdir(parents=True)
    df = pd.DataFrame(
        [
            _panel_row("2020-01-02", 10001, prc=100.0),
            _panel_row("2020-01-02", 10001, prc=99.0),
        ]
    )
    df.to_parquet(crsp / "data.parquet", index=False)

    out = tmp_path / "returns_panel.parquet"
    build_returns_panel(crsp.parent, out, overwrite=False)
    result = check_returns_panel(out)
    assert not result["passed"]
    assert result["duplicates"]["n_duplicate_pairs"] == 1
