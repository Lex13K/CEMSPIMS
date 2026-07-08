from __future__ import annotations

import sys
from pathlib import Path

import duckdb


def build_returns_panel(
    crsp_parquet_dir: Path,
    out_path: Path,
    *,
    use_ret: str = "retx",
    apply_delisting_adjustment: bool = False,
    common_shares_only: bool = True,
    major_exchanges_only: bool = False,
    min_date: str | None = None,
    max_date: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Create a cleaned daily returns panel suitable for rolling-window dependence estimation.
    Keeps delisting variables for later adjustment and avoids survivorship bias by not
    requiring continuous listing beyond each date.

    Parameters:
      use_ret: choose "ret" or "retx" as the main return column.
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Use overwrite=True to replace.")

    con = duckdb.connect(database=":memory:")
    # DuckDB's TTY progress bar corrupts output when mixed with tqdm / Windows consoles.
    con.execute("PRAGMA enable_progress_bar=false;")
    print("returns_panel: scanning CRSP shards and writing parquet…", file=sys.stderr, flush=True)

    ret_col = "retx" if use_ret.lower() == "retx" else "ret"

    where_clauses = ["permno IS NOT NULL", "date IS NOT NULL"]

    if common_shares_only:
        where_clauses.append("shrcd IN (10, 11)")

    if major_exchanges_only:
        where_clauses.append("exchcd IN (1, 2, 3)")

    if min_date:
        where_clauses.append(f"date >= DATE '{min_date}'")
    if max_date:
        where_clauses.append(f"date <= DATE '{max_date}'")

    where_sql = " AND ".join(where_clauses)
    glob_sql = (
        str((crsp_parquet_dir / "**" / "*.parquet").resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )
    out_sql = str(out_path.resolve()).replace("\\", "/").replace("'", "''")

    ret_delisted_expr = (
        "CASE "
        "WHEN dlstcd IS NOT NULL AND dlret IS NOT NULL "
        f"THEN (1.0 + {ret_col}) * (1.0 + dlret) - 1.0 "
        f"ELSE {ret_col} "
        "END AS ret_delisted"
        if apply_delisting_adjustment
        else f"CAST(NULL AS DOUBLE) AS ret_delisted"
    )

    query = f"""
    COPY (
        SELECT DISTINCT
            date,
            permno,
            permco,
            shrcd,
            exchcd,
            {ret_col} AS ret_used,
            {ret_delisted_expr},
            prc,
            vol,
            shrout,
            dlstcd,
            dlret,
            dlretx,
            dlprc,
            dlpdt,
            ticker,
            comnam,
            ncusip,
            cusip,
            secstat
        FROM read_parquet('{glob_sql}')
        WHERE {where_sql}
    )
    TO '{out_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(query)
    con.close()
    print(f"returns_panel: wrote {out_path}", file=sys.stderr, flush=True)
    return out_path


def check_returns_panel(
    returns_panel_path: Path,
    *,
    head_n: int = 5,
) -> dict[str, object]:
    """
    Lightweight diagnostics for the returns panel.

    Returns a dict with:
      - summary: row count, date range, distinct permnos
      - head: first few rows ordered by date
      - delisting: counts of rows with delisting codes/returns
      - duplicates: count of duplicate (date, permno) pairs and sample rows
      - passed: True only if there are zero duplicate (date, permno) pairs
    """
    if not returns_panel_path.exists():
        raise FileNotFoundError(f"Missing returns panel: {returns_panel_path}")

    con = duckdb.connect(database=":memory:")
    p = str(returns_panel_path)

    summary = con.execute(
        """
        SELECT COUNT(*) n_rows,
               MIN(date) min_date,
               MAX(date) max_date,
               COUNT(DISTINCT permno) n_permno
        FROM read_parquet(?)
        """,
        [p],
    ).fetchdf()

    head = con.execute(
        """
        SELECT date, permno, shrcd, exchcd, ret_used, dlstcd, dlret
        FROM read_parquet(?)
        ORDER BY date
        LIMIT ?
        """,
        [p, int(head_n)],
    ).fetchdf()

    delisting = con.execute(
        """
        SELECT
          SUM(CASE WHEN dlstcd IS NOT NULL THEN 1 ELSE 0 END) AS n_delist_rows,
          SUM(CASE WHEN dlret  IS NOT NULL THEN 1 ELSE 0 END) AS n_dlret_rows
        FROM read_parquet(?)
        """,
        [p],
    ).fetchdf()

    dup_count = con.execute(
        """
        SELECT COUNT(*) AS n_duplicate_pairs
        FROM (
            SELECT date, permno
            FROM read_parquet(?)
            GROUP BY date, permno
            HAVING COUNT(*) > 1
        )
        """,
        [p],
    ).fetchone()[0]

    dup_sample = con.execute(
        """
        SELECT date, permno, COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY date, permno
        HAVING COUNT(*) > 1
        ORDER BY date, permno
        LIMIT 10
        """,
        [p],
    ).fetchdf()

    con.close()
    return {
        "summary": summary,
        "head": head,
        "delisting": delisting,
        "duplicates": {"n_duplicate_pairs": int(dup_count), "sample": dup_sample},
        "passed": int(dup_count) == 0,
    }
