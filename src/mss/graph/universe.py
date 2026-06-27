"""Universe selection per feature date."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from mss.graph.progress_util import maybe_tqdm


def _sql_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/").replace("'", "''")
    return f"'{s}'"


def _last_trading_day_of_prev_month(con: duckdb.DuckDBPyConnection, date_str: str) -> str | None:
    q = f"""
        SELECT MAX(date) AS d
        FROM returns_panel
        WHERE date <= last_day(date '{date_str}' - INTERVAL '1 month')
    """
    row = con.execute(q).fetchone()
    return row[0].isoformat() if row and row[0] is not None else None


def _eligible_at_date(
    con: duckdb.DuckDBPyConnection,
    date_str: str,
    *,
    window_length: int,
    min_obs: int,
) -> pd.DataFrame:
    window_dates_sql = f"""
        SELECT di.date FROM dates_index di
        WHERE di.pos BETWEEN
            GREATEST(1, (SELECT pos FROM dates_index WHERE date = '{date_str}' LIMIT 1) - {window_length} + 1)
            AND (SELECT pos FROM dates_index WHERE date = '{date_str}' LIMIT 1)
    """
    q = f"""
        WITH window_dates AS ({window_dates_sql}),
        eligible AS (
            SELECT
                permno,
                COUNT(ret_used) AS n_obs
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            GROUP BY permno
            HAVING n_obs >= {min_obs}
        ),
        mcap_at_date AS (
            SELECT
                permno,
                ABS(COALESCE(prc, 0)) * NULLIF(COALESCE(shrout, 0), 0) AS mcap
            FROM returns_panel
            WHERE date = '{date_str}'
        )
        SELECT
            e.permno,
            m.mcap,
            e.n_obs
        FROM eligible e
        JOIN mcap_at_date m USING (permno)
        WHERE m.mcap IS NOT NULL AND m.mcap > 0
        ORDER BY m.mcap DESC
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
) -> pd.DataFrame:
    del selection_rule  # mcap-only ranking in this path
    elig = _eligible_at_date(con, feature_date, window_length=window_length, min_obs=min_obs)
    if elig.empty:
        return pd.DataFrame(columns=["permno", "rank", "mcap"])
    elig_permnos = set(elig["permno"].tolist())
    still_in = current_permno_set & elig_permnos
    need = n_nodes - len(still_in)
    candidates = elig[~elig["permno"].isin(still_in)]
    replacements = candidates.head(need)["permno"].tolist()
    new_set = list(still_in) + replacements
    out = elig[elig["permno"].isin(new_set)].copy()
    out = out.sort_values("mcap", ascending=False).reset_index(drop=True)
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
) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    pq = _sql_path(Path(returns_panel_path))
    con.execute(
        f"""CREATE VIEW returns_panel AS SELECT date, permno, ret_used, prc, shrout
        FROM read_parquet({pq})"""
    )
    con.execute("""
        CREATE TEMP TABLE dates_index AS
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS pos
        FROM (SELECT DISTINCT date FROM returns_panel)
    """)
    rows = []
    for d in date_chunk:
        u = universe_for_date(
            con,
            d,
            window_length=window_length,
            min_obs=min_obs,
            n_nodes=n_nodes,
            selection_rule=selection_rule,
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
) -> pd.DataFrame:
    rebalance_date = _last_trading_day_of_prev_month(con, feature_date)
    if rebalance_date is None:
        rebalance_date = feature_date

    if selection_rule.lower() == "mcap":
        order_col = "mcap DESC"
    else:
        order_col = "mcap DESC"

    window_dates_sql = f"""
        SELECT di.date FROM dates_index di
        WHERE di.pos BETWEEN
            GREATEST(1, (SELECT pos FROM dates_index WHERE date = '{rebalance_date}' LIMIT 1) - {window_length} + 1)
            AND (SELECT pos FROM dates_index WHERE date = '{rebalance_date}' LIMIT 1)
    """
    q = f"""
        WITH window_dates AS ({window_dates_sql}),
        eligible AS (
            SELECT
                permno,
                COUNT(ret_used) AS n_obs
            FROM returns_panel
            WHERE date IN (SELECT date FROM window_dates)
            GROUP BY permno
            HAVING n_obs >= {min_obs}
        ),
        mcap_at_date AS (
            SELECT
                permno,
                ABS(COALESCE(prc, 0)) * NULLIF(COALESCE(shrout, 0), 0) AS mcap
            FROM returns_panel
            WHERE date = '{rebalance_date}'
        )
        SELECT
            e.permno,
            m.mcap,
            ROW_NUMBER() OVER (ORDER BY {order_col}) AS rank
        FROM eligible e
        JOIN mcap_at_date m USING (permno)
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

    con = duckdb.connect(database=":memory:")
    pq = _sql_path(returns_panel_path)
    con.execute(
        f"""CREATE VIEW returns_panel AS SELECT date, permno, ret_used, prc, shrout
        FROM read_parquet({pq})"""
    )
    con.execute("""
        CREATE TEMP TABLE dates_index AS
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS pos
        FROM (SELECT DISTINCT date FROM returns_panel)
    """)

    dates_df = pd.read_parquet(feature_dates_path)
    feature_dates = dates_df["date"].astype(str).tolist()
    if limit_dates is not None:
        feature_dates = feature_dates[:limit_dates]

    rows: list[pd.DataFrame] = []
    n_dates = len(feature_dates)

    if universe_mode == "fixed_replace":
        if not feature_dates:
            con.close()
            return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
        first_date = feature_dates[0]
        if not show_progress and verbose and (1 % progress_every == 0 or n_dates == 1):
            print(f"  [    1/{n_dates}] universe for date {first_date}")
        elig0 = _eligible_at_date(con, first_date, window_length=window_length, min_obs=min_obs)
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
            )
            if u.empty:
                continue
            u = u.copy()
            u["date"] = pd.to_datetime(d)
            rows.append(u)
            current_set = set(u["permno"].tolist())
    else:
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
                )
                if u.empty:
                    continue
                u = u.copy()
                u["date"] = pd.to_datetime(d)
                rows.append(u)

    con.close()
    if not rows:
        return pd.DataFrame(columns=["date", "permno", "rank", "mcap"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "permno", "rank", "mcap"]]
