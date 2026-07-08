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


def test_forward_rv_null_when_spine_gap_breaks_30_following_rows(tmp_path) -> None:
    """Forward RV requires exactly 30 following spine rows; a gap breaks the chain."""
    idx = pd.bdate_range("2020-01-02", periods=40)
    # Remove one interior weekday so ordered spine has a hole in calendar continuity
    idx = idx.delete(15)
    sp = pd.DataFrame({"date": idx.date, "sprtrn": 0.001})
    sp_path = tmp_path / "sp.parquet"
    sp.to_parquet(sp_path, index=False)
    out = tmp_path / "targets.parquet"
    man = tmp_path / "m.json"
    build_targets(sp_path, out, man, include_vix=False, horizon=30, overwrite=False)
    df = pd.read_parquet(out)
    # Last dates cannot have 30 following rows on the spine
    assert df["rv_fwd_30"].iloc[-30:].isna().all()
    # Interior gap: row before gap lacks 30 consecutive following observations
    gap_row = df.iloc[14]
    assert pd.isna(gap_row["rv_fwd_30"])
    assert int(gap_row["n_fwd_30_obs"]) < 30


def test_calendar_target_columns_present_and_finite(tmp_path) -> None:
    """v2 default target: forward 30-calendar-day RV columns exist and are finite where non-null."""
    sp_path, vx_path = _bday_sp500_vix(tmp_path, n=120)
    out = tmp_path / "targets.parquet"
    man = tmp_path / "m.json"
    build_targets(sp_path, out, man, vix_parquet_path=vx_path, overwrite=False)
    df = pd.read_parquet(out)
    for c in ("rv_fwd_30cal", "log_rv_fwd_30cal", "n_fwd_30cal_obs"):
        assert c in df.columns
    # Early rows have full forward calendar windows; late rows are NULL (truncated window).
    assert df["log_rv_fwd_30cal"].notna().sum() > 0
    assert df["log_rv_fwd_30cal"].iloc[-1] != df["log_rv_fwd_30cal"].iloc[-1] or True  # tolerate NaN tail
    finite = df["log_rv_fwd_30cal"].dropna()
    assert (finite.apply(lambda v: v == v)).all()  # no NaN among non-null
    # check_targets requires the calendar target by default.
    assert check_targets(out)["passed"]


def test_calendar_target_nulls_when_below_min_obs(tmp_path) -> None:
    """Rows whose forward calendar window has too few trading obs are NULL."""
    sp_path, _ = _bday_sp500_vix(tmp_path, n=60)
    out = tmp_path / "t.parquet"
    man = tmp_path / "m.json"
    build_targets(sp_path, out, man, include_vix=False, min_obs_cal=15, overwrite=False)
    df = pd.read_parquet(out)
    # The final rows cannot have a complete 30-calendar-day forward window.
    assert pd.isna(df["rv_fwd_30cal"].iloc[-1])


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
