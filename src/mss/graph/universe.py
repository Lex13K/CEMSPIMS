"""Universe selection per feature date."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from mss.calendar.trading_windows import window_dates_sql
from mss.graph.progress_util import maybe_tqdm


def _sql_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/").replace("'", "''")
    return f"'{s}'"


def _create_returns_panel_view(con: duckdb.DuckDBPyConnection, pq: str) -> None:
    """Create the returns_panel view; tolerate panels without a `vol` column (dollar_volume off)."""
    cols = [
        str(r[0]).lower()
        for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({pq})").fetchall()
    ]
    vol_expr = "vol" if "vol" in cols else "CAST(NULL AS DOUBLE) AS vol"
    con.execute(
        f"""CREATE VIEW returns_panel AS
        SELECT date, permno, ret_used, prc, shrout, {vol_expr}
        FROM read_parquet({pq})"""
    )


def _last_trading_day_of_prev_month(con: duckdb.DuckDBPyConnection, date_str: str) -> str | None:
    q = f"""
        SELECT MAX(date) AS d
        FROM returns_panel
        WHERE date <= last_day(date '{date_str}' - INTERVAL '1 month')
    """
    row = con.execute(q).fetchone()
    return row[0].isoformat() if row and row[0] is not None else None


def _last_trading_day_of_prev_quarter(con: duckdb.DuckDBPyConnection, date_str: str) -> str | None:
    q = f"""
        SELECT MAX(date) AS d
        FROM returns_panel
        WHERE date <= last_day(date '{date_str}' - INTERVAL '3 month')
    """
    row = con.execute(q).fetchone()
    return row[0].isoformat() if row and row[0] is not None else None


def _rebalance_anchor_date(
    con: duckdb.DuckDBPyConnection,
    feature_date: str,
    rebalance_freq: str,
) -> str:
    freq = str(rebalance_freq).strip().lower()
    if freq == "daily":
        return feature_date
    if freq == "monthly":
        anchor = _last_trading_day_of_prev_month(con, feature_date)
        return anchor if anchor is not None else feature_date
    if freq == "quarterly":
        anchor = _last_trading_day_of_prev_quarter(con, feature_date)
        return anchor if anchor is not None else feature_date
    raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq!r}")


_VALID_SELECTION_RULES = ("mcap", "dollar_volume")


def _register_sp500_spans(con: duckdb.DuckDBPyConnection, membership_path: str | None) -> bool:
    """Register the optional S&P 500 membership spans table; return True if available."""
    if not membership_path:
        return False
    p = Path(membership_path)
    if not p.is_file():
        raise FileNotFoundError(f"sp500_membership_path not found: {p}")
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW sp500_spans AS
        SELECT CAST(permno AS BIGINT) AS permno,
               CAST(start_date AS DATE) AS start_date,
               TRY_CAST(end_date AS DATE) AS end_date
        FROM read_parquet({_sql_path(p)})"""
    )
    return True


def _membership_clause(asof_date: str, restrict_to_sp500: bool) -> str:
    """AND-clause restricting permnos to S&P 500 members as-of `asof_date` (or empty)."""
    if not restrict_to_sp500:
        return ""
    return (
        f"AND permno IN (SELECT permno FROM sp500_spans "
        f"WHERE start_date <= DATE '{asof_date}' "
        f"AND (end_date IS NULL OR end_date >= DATE '{asof_date}'))"
    )


def _selection_metric_column(selection_rule: str) -> str:
    """Column to rank the universe by. ``mcap`` is canonical; ``dollar_volume`` is robustness-only."""
    rule = str(selection_rule).strip().lower()
    if rule == "dollar_volume":
        return "dvol"
    return "mcap"


def _eligible_at_date(
    con: duckdb.DuckDBPyConnection,
    date_str: str,
    *,
    window_length: int,
    min_obs: int,
    selection_rule: str = "mcap",
    restrict_to_sp500: bool = False,
) -> pd.DataFrame:
    window_dates_fragment = window_dates_sql(f"DATE '{date_str}'", window_length)
    metric = _selection_metric_column(selection_rule)
    member_clause = _membership_clause(date_str, restrict_to_sp500)
    q = f"""
        WITH window_dates AS ({window_dates_fragment}),
        eligible AS (
            SELECT
                permno,
                COUNT(ret_used) AS n_obs
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            {member_clause}
            GROUP BY permno
            HAVING n_obs >= {min_obs}
        ),
        mcap_at_date AS (
            SELECT
                permno,
                ABS(COALESCE(prc, 0)) * NULLIF(COALESCE(shrout, 0), 0) AS mcap
            FROM returns_panel
            WHERE date = '{date_str}'
        ),
        dvol_window AS (
            SELECT
                permno,
                AVG(ABS(COALESCE(prc, 0)) * COALESCE(vol, 0)) AS dvol
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            GROUP BY permno
        )
        SELECT
            e.permno,
            m.mcap,
            COALESCE(d.dvol, 0) AS dvol,
            e.n_obs
        FROM eligible e
        JOIN mcap_at_date m USING (permno)
        LEFT JOIN dvol_window d USING (permno)
        WHERE m.mcap IS NOT NULL AND m.mcap > 0
        ORDER BY {metric} DESC
    """
    return con.execute(q).df()


def universe_for_date_fixed_replace(
    con: duckdb.DuckDBPyConnection,
    feature_date: str,
    current_permno_set: set[int],
    *,
    window_length: int,
    min_obs: int,
    n_nodes: int,
    selection_rule: str = "mcap",
    restrict_to_sp500: bool = False,
) -> pd.DataFrame:
    metric = _selection_metric_column(selection_rule)
    elig = _eligible_at_date(
        con,
        feature_date,
        window_length=window_length,
        min_obs=min_obs,
        selection_rule=selection_rule,
        restrict_to_sp500=restrict_to_sp500,
    )
    if elig.empty:
        return pd.DataFrame(columns=["permno", "rank", "mcap"])
    # `elig` is already ordered by the selection metric (mcap or dollar_volume).
    elig_permnos = set(elig["permno"].tolist())
    still_in = current_permno_set & elig_permnos
    need = n_nodes - len(still_in)
    candidates = elig[~elig["permno"].isin(still_in)]
    replacements = candidates.head(need)["permno"].tolist()
    new_set = list(still_in) + replacements
    out = elig[elig["permno"].isin(new_set)].copy()
    out = out.sort_values(metric, ascending=False).reset_index(drop=True)
    out = out.head(n_nodes)
    out["rank"] = range(1, len(out) + 1)
    return out[["permno", "rank", "mcap"]]


def _build_universe_chunk_monthly_rebalance(
    returns_panel_path: str,
    date_chunk: list[str],
    *,
    window_length: int,
    min_obs: int,
    n_nodes: int,
    selection_rule: str = "mcap",
    rebalance_freq: str = "monthly",
    restrict_to_sp500: bool = False,
    sp500_membership_path: str | None = None,
) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    pq = _sql_path(Path(returns_panel_path))
    _create_returns_panel_view(con, pq)
    con.execute("""
        CREATE TEMP TABLE dates_index AS
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS pos
        FROM (SELECT DISTINCT date FROM returns_panel)
    """)
    if restrict_to_sp500:
        _register_sp500_spans(con, sp500_membership_path)
    rows = []
    for d in date_chunk:
        u = universe_for_date(
            con,
            d,
            window_length=window_length,
            min_obs=min_obs,
            n_nodes=n_nodes,
            selection_rule=selection_rule,
            rebalance_freq=rebalance_freq,
            restrict_to_sp500=restrict_to_sp500,
        )
        if u.empty:
            continue
        u = u.copy()
        u["date"] = pd.to_datetime(d)
        rows.append(u)
    con.close()
    if not rows:
        return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
    return pd.concat(rows, ignore_index=True)[["date", "permno", "rank", "mcap"]]


def universe_for_date(
    con: duckdb.DuckDBPyConnection,
    feature_date: str,
    *,
    window_length: int,
    min_obs: int,
    n_nodes: int,
    selection_rule: str = "mcap",
    rebalance_freq: str = "monthly",
    restrict_to_sp500: bool = False,
) -> pd.DataFrame:
    rebalance_date = _rebalance_anchor_date(con, feature_date, rebalance_freq)

    # mcap is canonical; dollar_volume is an optional robustness ranking. The mcap column is always
    # emitted for schema continuity; only the ranking metric changes.
    metric = _selection_metric_column(selection_rule)
    order_col = f"{metric} DESC"
    member_clause = _membership_clause(rebalance_date, restrict_to_sp500)

    window_dates_fragment = window_dates_sql(f"DATE '{rebalance_date}'", window_length)
    q = f"""
        WITH window_dates AS ({window_dates_fragment}),
        eligible AS (
            SELECT
                permno,
                COUNT(ret_used) AS n_obs
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            {member_clause}
            GROUP BY permno
            HAVING n_obs >= {min_obs}
        ),
        mcap_at_date AS (
            SELECT
                permno,
                ABS(COALESCE(prc, 0)) * NULLIF(COALESCE(shrout, 0), 0) AS mcap
            FROM returns_panel
            WHERE date = '{rebalance_date}'
        ),
        dvol_window AS (
            SELECT
                permno,
                AVG(ABS(COALESCE(prc, 0)) * COALESCE(vol, 0)) AS dvol
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            GROUP BY permno
        )
        SELECT
            e.permno,
            m.mcap,
            ROW_NUMBER() OVER (ORDER BY {order_col}) AS rank
        FROM eligible e
        JOIN mcap_at_date m USING (permno)
        LEFT JOIN dvol_window d USING (permno)
        WHERE m.mcap IS NOT NULL AND m.mcap > 0
        ORDER BY rank
        LIMIT {n_nodes}
    """
    df = con.execute(q).df()
    return df


def build_universe_per_date(
    returns_panel_path: Path,
    feature_dates_path: Path,
    *,
    window_length: int,
    min_obs: int,
    n_nodes: int,
    selection_rule: str = "mcap",
    universe_mode: str = "fixed_replace",
    rebalance_freq: str = "monthly",
    restrict_to_sp500: bool = False,
    sp500_membership_path: str | None = None,
    progress_every: int = 50,
    limit_dates: int | None = None,
    n_jobs: int = 1,
    show_progress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if not returns_panel_path.is_file():
        raise FileNotFoundError(f"Returns panel not found: {returns_panel_path}")
    if not feature_dates_path.is_file():
        raise FileNotFoundError(f"Feature dates not found: {feature_dates_path}")

    mode = str(universe_mode).strip().lower()
    if mode not in ("fixed_replace", "monthly_rebalance"):
        raise ValueError(
            f"universe_mode must be 'fixed_replace' or 'monthly_rebalance'; got {universe_mode!r}"
        )

    con = duckdb.connect(database=":memory:")
    pq = _sql_path(returns_panel_path)
    _create_returns_panel_view(con, pq)
    con.execute("""
        CREATE TEMP TABLE dates_index AS
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS pos
        FROM (SELECT DISTINCT date FROM returns_panel)
    """)
    if restrict_to_sp500:
        _register_sp500_spans(con, sp500_membership_path)

    dates_df = pd.read_parquet(feature_dates_path)
    feature_dates = dates_df["date"].astype(str).tolist()
    if limit_dates is not None:
        feature_dates = feature_dates[:limit_dates]

    rows: list[pd.DataFrame] = []
    n_dates = len(feature_dates)

    if mode == "fixed_replace":
        if not feature_dates:
            con.close()
            return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
        first_date = feature_dates[0]
        if not show_progress and verbose and (1 % progress_every == 0 or n_dates == 1):
            print(f"  [    1/{n_dates}] universe for date {first_date}")
        elig0 = _eligible_at_date(
            con,
            first_date,
            window_length=window_length,
            min_obs=min_obs,
            selection_rule=selection_rule,
            restrict_to_sp500=restrict_to_sp500,
        )
        seed_df = elig0.head(n_nodes)
        if seed_df.empty:
            con.close()
            return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
        seed_df = seed_df.copy()
        seed_df["rank"] = range(1, len(seed_df) + 1)
        seed_df = seed_df[["permno", "rank", "mcap"]]
        seed_df["date"] = pd.to_datetime(first_date)
        rows.append(seed_df)
        current_set = set(seed_df["permno"].tolist())
        rest_idx = range(1, len(feature_dates))
        rest_iter = maybe_tqdm(
            rest_idx,
            desc="universe",
            total=max(0, len(feature_dates) - 1),
            show=show_progress,
        )
        for j in rest_iter:
            d = feature_dates[j]
            i = j + 1
            if not show_progress and verbose and (i == 2 or i % progress_every == 0 or i == n_dates):
                print(f"  [{i:5d}/{n_dates}] universe for date {d}")
            u = universe_for_date_fixed_replace(
                con,
                d,
                current_set,
                window_length=window_length,
                min_obs=min_obs,
                n_nodes=n_nodes,
                selection_rule=selection_rule,
                restrict_to_sp500=restrict_to_sp500,
            )
            if u.empty:
                continue
            u = u.copy()
            u["date"] = pd.to_datetime(d)
            rows.append(u)
            current_set = set(u["permno"].tolist())
    elif mode == "monthly_rebalance":
        if n_jobs > 1:
            chunk_size = max(1, (n_dates + n_jobs - 1) // n_jobs)
            chunks = [feature_dates[i : i + chunk_size] for i in range(0, n_dates, chunk_size)]
            rp_str = str(returns_panel_path.resolve())
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(
                        _build_universe_chunk_monthly_rebalance,
                        rp_str,
                        chunk,
                        window_length=window_length,
                        min_obs=min_obs,
                        n_nodes=n_nodes,
                        selection_rule=selection_rule,
                        rebalance_freq=rebalance_freq,
                        restrict_to_sp500=restrict_to_sp500,
                        sp500_membership_path=sp500_membership_path,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                chunk_results: list[pd.DataFrame | None] = [None] * len(chunks)
                for f in as_completed(futures):
                    idx = futures[f]
                    chunk_results[idx] = f.result()
                rows = [df for df in chunk_results if df is not None and not df.empty]
        else:
            idx_iter = range(len(feature_dates))
            idx_iter = maybe_tqdm(idx_iter, desc="universe", total=n_dates, show=show_progress)
            for j in idx_iter:
                d = feature_dates[j]
                i = j + 1
                if not show_progress and verbose and (i == 1 or i % progress_every == 0 or i == n_dates):
                    print(f"  [{i:5d}/{n_dates}] universe for date {d}")
                u = universe_for_date(
                    con,
                    d,
                    window_length=window_length,
                    min_obs=min_obs,
                    n_nodes=n_nodes,
                    selection_rule=selection_rule,
                    rebalance_freq=rebalance_freq,
                    restrict_to_sp500=restrict_to_sp500,
                )
                if u.empty:
                    continue
                u = u.copy()
                u["date"] = pd.to_datetime(d)
                rows.append(u)
    else:
        con.close()
        raise ValueError(f"Unsupported universe_mode: {universe_mode!r}")

    con.close()
    if not rows:
        return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "permno", "rank", "mcap"]]
