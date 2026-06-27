from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from mss.data.targets import build_targets, check_targets, parse_trailing_windows


def test_parse_trailing_windows_ok() -> None:
    assert parse_trailing_windows("30, 252") == (30, 252)
    assert parse_trailing_windows("") == ()


def test_parse_trailing_windows_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_trailing_windows("0")


def _bday_sp500_vix(tmp_path: Path, *, n: int = 50) -> tuple[Path, Path]:
    idx = pd.bdate_range("2020-01-02", periods=n)
    sp = pd.DataFrame({"date": idx.date, "sprtrn": 0.001})
    vx = pd.DataFrame({"date": idx.date, "vix": 18.0 + (idx.dayofyear % 5) * 0.1})
    sp_path = tmp_path / "sp500_returns.parquet"
    vx_path = tmp_path / "vix.parquet"
    sp.to_parquet(sp_path, index=False)
    vx.to_parquet(vx_path, index=False)
    return sp_path, vx_path


def test_build_targets_and_check_ok(tmp_path) -> None:
    sp_path, vx_path = _bday_sp500_vix(tmp_path, n=50)
    out = tmp_path / "targets.parquet"
    man = tmp_path / "targets_manifest.json"
    build_targets(
        sp_path,
        out,
        man,
        vix_parquet_path=vx_path,
        overwrite=False,
    )
    assert out.is_file() and man.is_file()
    r = check_targets(out)
    assert r["passed"]


def test_build_targets_without_vix(tmp_path) -> None:
    sp_path, _ = _bday_sp500_vix(tmp_path, n=50)
    out = tmp_path / "t.parquet"
    man = tmp_path / "t.json"
    build_targets(
        sp_path,
        out,
        man,
        include_vix=False,
        overwrite=False,
    )
    r = check_targets(out, expect_vix_column=False)
    assert r["passed"]


def test_check_targets_fails_on_duplicate_dates(tmp_path) -> None:
    d = dt.date(2020, 1, 2)
    sp = pd.DataFrame(
        {
            "date": [d, d],
            "sprtrn": [0.01, 0.02],
        }
    )
    sp_path = tmp_path / "sp.parquet"
    sp.to_parquet(sp_path, index=False)
    vx = pd.DataFrame({"date": [d], "vix": [20.0]})
    vx_path = tmp_path / "vx.parquet"
    vx.to_parquet(vx_path, index=False)
    out = tmp_path / "t.parquet"
    man = tmp_path / "m.json"
    build_targets(sp_path, out, man, vix_parquet_path=vx_path, overwrite=False)
    r = check_targets(out)
    assert not r["passed"]
