from __future__ import annotations

import pandas as pd
import numpy as np

from mss.analysis.figures import (
    _compute_rankbucket_replacement_matrix,
    _compute_tenure_streak_lengths,
    _compute_turnover_rates,
)


def _universe_fixture_df() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    d1 = pd.Timestamp("2020-01-02")
    d2 = pd.Timestamp("2020-01-03")
    d3 = pd.Timestamp("2020-01-06")

    rows = [
        # d1: permnos 1..4 with ranks 1..4
        {"date": d1, "permno": 1, "rank": 1},
        {"date": d1, "permno": 2, "rank": 2},
        {"date": d1, "permno": 3, "rank": 3},
        {"date": d1, "permno": 4, "rank": 4},
        # d2: permnos {2,3,4,5} with ranks 1..4
        {"date": d2, "permno": 2, "rank": 1},
        {"date": d2, "permno": 3, "rank": 2},
        {"date": d2, "permno": 4, "rank": 3},
        {"date": d2, "permno": 5, "rank": 4},
        # d3: permnos {3,4,5,6} with ranks 1..4
        {"date": d3, "permno": 3, "rank": 1},
        {"date": d3, "permno": 4, "rank": 2},
        {"date": d3, "permno": 5, "rank": 3},
        {"date": d3, "permno": 6, "rank": 4},
    ]
    df = pd.DataFrame(rows)
    feature_dates = [d1, d2, d3]
    return df, feature_dates


def test_compute_turnover_rates_basic() -> None:
    df, feature_dates = _universe_fixture_df()
    t = _compute_turnover_rates(df, feature_dates)

    assert list(t.columns) == ["date", "universe_size", "replaced_count", "turnover_rate"]
    assert len(t) == 2

    # d2: replaced permno=5 (only), universe_size=4 => 0.25
    assert pd.Timestamp(t.loc[0, "date"]) == feature_dates[1]
    assert int(t.loc[0, "universe_size"]) == 4
    assert int(t.loc[0, "replaced_count"]) == 1
    assert np.isclose(float(t.loc[0, "turnover_rate"]), 0.25)

    # d3: replaced permno=6, universe_size=4 => 0.25
    assert pd.Timestamp(t.loc[1, "date"]) == feature_dates[2]
    assert int(t.loc[1, "universe_size"]) == 4
    assert int(t.loc[1, "replaced_count"]) == 1
    assert np.isclose(float(t.loc[1, "turnover_rate"]), 0.25)


def test_rankbucket_replacement_matrix_bucketized() -> None:
    df, feature_dates = _universe_fixture_df()
    mat, bucket_ranges, bucket_size = _compute_rankbucket_replacement_matrix(df, feature_dates, n_buckets=2)

    assert bucket_size == 2  # max_rank=4, 2 buckets => bucket_size=2
    assert bucket_ranges == [(1, 2), (3, 4)]
    assert mat.shape == (2, 2)

    # With our construction each bucket loses one member and gains one member each transition => 1/2 = 0.5
    assert np.allclose(mat, np.full((2, 2), 0.5))

    # Long-form export sanity (matches the CSV writer logic).
    dates_cols = feature_dates[1:]
    rows = []
    for col_idx, d_cur in enumerate(dates_cols):
        for b, (lo, hi) in enumerate(bucket_ranges):
            rows.append(
                {
                    "date": pd.Timestamp(d_cur),
                    "rank_bucket": int(b),
                    "rank_lo": int(lo),
                    "rank_hi": int(hi),
                    "bucket_size": int(bucket_size),
                    "replacement_fraction": float(mat[b, col_idx]),
                }
            )
    long_df = pd.DataFrame(rows)
    assert list(long_df.columns) == [
        "date",
        "rank_bucket",
        "rank_lo",
        "rank_hi",
        "bucket_size",
        "replacement_fraction",
    ]
    assert len(long_df) == 2 * 2  # 2 buckets x 2 transitions
    assert np.allclose(long_df["replacement_fraction"].to_numpy(), 0.5)


def test_tenure_streak_lengths_counts() -> None:
    df, feature_dates = _universe_fixture_df()
    streaks = _compute_tenure_streak_lengths(df, feature_dates)

    # Presence over steps:
    # permno1: [d1] => 1
    # permno2: [d1,d2] => 2
    # permno3: [d1,d2,d3] => 3
    # permno4: [d1,d2,d3] => 3
    # permno5: [d2,d3] => 2
    # permno6: [d3] => 1
    assert sorted(streaks) == [1, 1, 2, 2, 3, 3]

