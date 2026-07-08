"""Universe mode semantics: fixed_replace vs monthly_rebalance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.graph.universe import build_universe_per_date


def _write_liquidity_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Panel where mcap and dollar-volume rankings disagree; includes a vol column."""
    dates = pd.bdate_range("2020-01-02", periods=8)
    specs = {
        1: dict(prc=10.0, vol=1_000.0, shrout=100.0),
        2: dict(prc=20.0, vol=5_000.0, shrout=100.0),
        3: dict(prc=100.0, vol=10.0, shrout=100.0),   # high mcap, tiny volume
        4: dict(prc=5.0, vol=20_000.0, shrout=100.0),  # low mcap, huge volume
    }
    rows = []
    for d in dates:
        for permno, s in specs.items():
            rows.append({"date": d, "permno": permno, "ret_used": 0.01, **s})
    rp = tmp_path / "returns_panel.parquet"
    pd.DataFrame(rows).to_parquet(rp, index=False)
    feat = tmp_path / "stage04_dates.parquet"
    pd.DataFrame({"date": dates}).to_parquet(feat, index=False)
    return rp, feat


def test_selection_rule_dollar_volume_differs_from_mcap(tmp_path: Path) -> None:
    rp, feat = _write_liquidity_fixture(tmp_path)
    common = dict(
        window_length=3, min_obs=1, n_nodes=2, universe_mode="monthly_rebalance",
        rebalance_freq="daily", show_progress=False, verbose=False,
    )
    u_mcap = build_universe_per_date(rp, feat, selection_rule="mcap", **common)
    u_dv = build_universe_per_date(rp, feat, selection_rule="dollar_volume", **common)
    last = u_mcap["date"].max()
    mcap_top = set(u_mcap[u_mcap["date"] == last]["permno"])
    dv_top = set(u_dv[u_dv["date"] == last]["permno"])
    assert mcap_top == {2, 3}       # ranked by prc*shrout
    assert dv_top == {2, 4}         # ranked by prc*vol
    # mcap column is still emitted for schema continuity under dollar_volume.
    assert "mcap" in u_dv.columns


def test_restrict_to_sp500_membership_filter(tmp_path: Path) -> None:
    rp, feat = _write_liquidity_fixture(tmp_path)
    spans = pd.DataFrame(
        {
            "permno": [1, 2],
            "start_date": [pd.Timestamp("2019-01-01").date()] * 2,
            "end_date": [pd.NaT, pd.NaT],
        }
    )
    membership = tmp_path / "sp500_membership.parquet"
    spans.to_parquet(membership, index=False)
    u = build_universe_per_date(
        rp, feat, window_length=3, min_obs=1, n_nodes=4,
        universe_mode="monthly_rebalance", rebalance_freq="daily",
        restrict_to_sp500=True, sp500_membership_path=str(membership),
        show_progress=False, verbose=False,
    )
    assert set(u["permno"].unique()).issubset({1, 2})


def test_fixed_replace_sp500_seed_respects_membership(tmp_path: Path) -> None:
    """fixed_replace seed cohort must honor restrict_to_sp500 on the first feature date."""
    rp, feat = _write_liquidity_fixture(tmp_path)
    spans = pd.DataFrame(
        {
            "permno": [1, 2],
            "start_date": [pd.Timestamp("2019-01-01").date()] * 2,
            "end_date": [pd.NaT, pd.NaT],
        }
    )
    membership = tmp_path / "sp500_membership.parquet"
    spans.to_parquet(membership, index=False)
    u = build_universe_per_date(
        rp,
        feat,
        window_length=3,
        min_obs=1,
        n_nodes=2,
        universe_mode="fixed_replace",
        rebalance_freq="monthly",
        restrict_to_sp500=True,
        sp500_membership_path=str(membership),
        show_progress=False,
        verbose=False,
    )
    first = u.loc[u["date"] == u["date"].min(), "permno"].tolist()
    # permno 3 has highest mcap but is not in the membership file; seed must exclude it.
    assert set(first) == {2, 1}
    assert 3 not in first


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Three months of daily data; mcap ranking shifts by calendar month."""
    dates = pd.bdate_range("2020-01-02", "2020-03-31")
    rows = []
    for d in dates:
        month = d.month
        for permno in range(1, 7):
            # Jan: low permnos win; Feb/Mar: high permnos win -> monthly rebalance churns.
            prc = float((7 - permno) if month == 1 else permno) * 100.0
            rows.append(
                {
                    "date": d,
                    "permno": permno,
                    "ret_used": 0.01,
                    "prc": prc,
                    "shrout": 1_000_000.0,
                }
            )
    rp = tmp_path / "returns_panel.parquet"
    pd.DataFrame(rows).to_parquet(rp, index=False)
    feat = tmp_path / "stage04_dates.parquet"
    pd.DataFrame({"date": dates}).to_parquet(feat, index=False)
    return rp, feat


def test_fixed_replace_carries_cohort_forward(tmp_path: Path) -> None:
    rp, feat = _write_fixture(tmp_path)
    out = build_universe_per_date(
        rp,
        feat,
        window_length=1,
        min_obs=1,
        n_nodes=3,
        universe_mode="fixed_replace",
        rebalance_freq="daily",
        show_progress=False,
        verbose=False,
        limit_dates=30,
    )
    first = set(out.loc[out["date"] == out["date"].min(), "permno"])
    # Sticky cohort: February dates should still use January seed permnos.
    feb = out[pd.to_datetime(out["date"]).dt.month == 2]
    if not feb.empty:
        feb_set = set(feb.loc[feb["date"] == feb["date"].min(), "permno"])
        assert feb_set == first


def test_monthly_rebalance_higher_turnover_than_fixed(tmp_path: Path) -> None:
    rp, feat = _write_fixture(tmp_path)
    common = dict(
        window_length=1,
        min_obs=1,
        n_nodes=3,
        show_progress=False,
        verbose=False,
        limit_dates=45,
    )
    fixed = build_universe_per_date(
        rp,
        feat,
        universe_mode="fixed_replace",
        rebalance_freq="daily",
        **common,
    )
    monthly = build_universe_per_date(
        rp,
        feat,
        universe_mode="monthly_rebalance",
        rebalance_freq="monthly",
        **common,
    )
    # Monthly mode should differ from sticky cohort at least once across months.
    fixed_sets = {
        frozenset(fixed.loc[fixed["date"] == d, "permno"])
        for d in fixed["date"].drop_duplicates()
    }
    monthly_sets = {
        frozenset(monthly.loc[monthly["date"] == d, "permno"])
        for d in monthly["date"].drop_duplicates()
    }
    assert len(monthly_sets) > 1
    assert monthly_sets != fixed_sets


def test_monthly_rebalance_constant_within_calendar_month(tmp_path: Path) -> None:
    rp, feat = _write_fixture(tmp_path)
    out = build_universe_per_date(
        rp,
        feat,
        window_length=1,
        min_obs=1,
        n_nodes=3,
        universe_mode="monthly_rebalance",
        rebalance_freq="monthly",
        show_progress=False,
        verbose=False,
        limit_dates=25,
    )
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"])
    for _, grp in out.groupby(out["date"].dt.to_period("M")):
        dates = grp["date"].drop_duplicates().sort_values()
        if len(dates) < 2:
            continue
        sets = [
            set(out.loc[out["date"] == d, "permno"]) for d in dates
        ]
        assert sets[0] == sets[-1]
