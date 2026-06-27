from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb


def parse_trailing_windows(s: str) -> tuple[int, ...]:
    """
    Parse a comma-separated string of positive integers (e.g. "22,252").
    Used for lagged RV trailing windows.
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return ()
    out: list[int] = []
    for p in parts:
        w = int(p)
        if w <= 0:
            raise ValueError(f"Trailing window must be positive: {w}")
        out.append(w)
    return tuple(out)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_overwritable_file(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Path exists and overwrite=False: {path}")
        path.unlink()


def _sql_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    s = s.replace("'", "''")
    return f"'{s}'"


def _create_vix_view_from_parquet(con: duckdb.DuckDBPyConnection, vix_parquet_path: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW vix_typed AS
        SELECT
            try_cast(date AS DATE) AS date,
            try_cast(vix AS DOUBLE) AS vix
        FROM read_parquet({_sql_path(vix_parquet_path)})
        WHERE try_cast(date AS DATE) IS NOT NULL
        """
    )


def build_targets(
    sp500_parquet_path: Path,
    out_path: Path,
    manifest_path: Path,
    *,
    vix_parquet_path: Path | None = None,
    horizon: int = 30,
    trailing_windows: tuple[int, ...] = (30, 252),
    annualization: int = 252,
    scale: float = 100.0,
    include_vix: bool = True,
    include_log: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Build daily targets: forward realized vol over `horizon` trading days plus lag RV controls.

    Inputs:
      - sp500_parquet_path: parquet with columns date, sprtrn (from ingest).
      - vix_parquet_path: parquet with date, vix (from ingest), if include_vix.

    Outputs:
      - out_path: targets.parquet (ZSTD)
      - manifest_path: JSON metadata
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if any(w <= 0 for w in trailing_windows):
        raise ValueError("All trailing_windows must be positive")
    if annualization <= 0:
        raise ValueError("annualization must be positive")

    if include_vix:
        if vix_parquet_path is None:
            raise ValueError("include_vix=True requires vix_parquet_path")
        if not vix_parquet_path.is_file():
            raise FileNotFoundError(f"include_vix=True but file not found: {vix_parquet_path}")

    _ensure_parent_dir(out_path)
    _ensure_overwritable_file(out_path, overwrite=overwrite)
    _ensure_overwritable_file(manifest_path, overwrite=overwrite)

    print("targets: building from S&P 500 returns…", file=sys.stderr, flush=True)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW sp500 AS
        SELECT
            try_cast(date AS DATE) AS date,
            try_cast(sprtrn AS DOUBLE) AS sprtrn
        FROM read_parquet({_sql_path(sp500_parquet_path)})
        WHERE try_cast(date AS DATE) IS NOT NULL
        ORDER BY date
        """
    )

    fwd_frame = f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING"
    fwd_sum = f"SUM(sprtrn * sprtrn) OVER (ORDER BY date {fwd_frame})"
    fwd_cnt = f"COUNT(sprtrn) OVER (ORDER BY date {fwd_frame})"
    fwd_rv_expr = (
        f"CASE WHEN {fwd_cnt} = {horizon} "
        f"THEN SQRT(({annualization}::DOUBLE / {horizon}::DOUBLE) * {fwd_sum}) * {scale} "
        f"ELSE NULL END"
    )

    base_cols: list[str] = [
        "date",
        "sprtrn",
        f"{fwd_rv_expr} AS rv_fwd_{horizon}",
        f"{fwd_cnt} AS n_fwd_{horizon}_obs",
    ]

    for w in trailing_windows:
        lag_frame = f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        lag_sum = f"SUM(sprtrn * sprtrn) OVER (ORDER BY date {lag_frame})"
        lag_cnt = f"COUNT(sprtrn) OVER (ORDER BY date {lag_frame})"
        lag_rv_expr = (
            f"CASE WHEN {lag_cnt} = {w} "
            f"THEN SQRT(({annualization}::DOUBLE / {w}::DOUBLE) * {lag_sum}) * {scale} "
            f"ELSE NULL END"
        )
        base_cols.append(f"{lag_rv_expr} AS rv_lag_{w}")
        base_cols.append(f"{lag_cnt} AS n_lag_{w}_obs")

    base_sql = "SELECT\n    " + ",\n    ".join(base_cols) + "\nFROM sp500"

    if include_log:
        log_cols: list[str] = [
            f"CASE WHEN rv_fwd_{horizon} > 0 THEN LN(rv_fwd_{horizon}) ELSE NULL END AS log_rv_fwd_{horizon}"
        ]
        for w in trailing_windows:
            log_cols.append(
                f"CASE WHEN rv_lag_{w} > 0 THEN LN(rv_lag_{w}) ELSE NULL END AS log_rv_lag_{w}"
            )

        with_sql = f"""
            WITH base AS (
                {base_sql}
            ),
            spine AS (
                SELECT
                    base.*,
                    {", ".join(log_cols)}
                FROM base
            )
        """
        spine_table = "spine"
    else:
        with_sql = f"""
            WITH spine AS (
                {base_sql}
            )
        """
        spine_table = "spine"

    if include_vix:
        _create_vix_view_from_parquet(con, vix_parquet_path)

        final_sql = f"""
            {with_sql}
            SELECT
                s.*,
                v.vix
            FROM {spine_table} s
            LEFT JOIN vix_typed v
            USING (date)
            ORDER BY s.date
        """
    else:
        final_sql = f"""
            {with_sql}
            SELECT *
            FROM {spine_table}
            ORDER BY date
        """

    con.execute(
        f"""
        COPY (
            {final_sql}
        )
        TO {_sql_path(out_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    out_file = str(out_path.resolve()).replace("\\", "/")
    out_sql = out_file.replace("'", "''")

    stats_df = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(CASE WHEN rv_fwd_{horizon} IS NULL THEN 1 ELSE 0 END) AS n_missing_rv_fwd
        FROM read_parquet('{out_sql}')
        """
    ).df()
    stats_row = stats_df.iloc[0].to_dict()

    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_sql}')").df()
    out_cols = cols_df["column_name"].tolist()

    manifest: dict[str, Any] = {
        "inputs": {
            "sp500_parquet_path": str(sp500_parquet_path),
            "vix_parquet_path": str(vix_parquet_path) if include_vix else None,
        },
        "parameters": {
            "horizon": horizon,
            "trailing_windows": list(trailing_windows),
            "annualization": annualization,
            "scale": scale,
            "include_vix": include_vix,
            "include_log": include_log,
        },
        "outputs": {
            "targets_path": str(out_path),
            "manifest_path": str(manifest_path),
            "columns": out_cols,
        },
        "stats": {
            "n_rows": int(stats_row["n_rows"]) if stats_row.get("n_rows") is not None else None,
            "min_date": str(stats_row["min_date"]) if stats_row.get("min_date") is not None else None,
            "max_date": str(stats_row["max_date"]) if stats_row.get("max_date") is not None else None,
            "n_missing_rv_fwd": int(stats_row["n_missing_rv_fwd"])
            if stats_row.get("n_missing_rv_fwd") is not None
            else None,
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    con.close()
    print(f"targets: wrote {out_path}", file=sys.stderr, flush=True)
    return out_path


def check_targets(
    targets_path: Path,
    *,
    horizon: int = 30,
    expect_vix_column: bool = True,
) -> dict[str, Any]:
    """
    Validate targets.parquet against docs/data_contracts.md (required columns, keys, basic numeric integrity).
    """
    if not targets_path.is_file():
        raise FileNotFoundError(f"Missing targets table: {targets_path}")

    rv_fwd = f"rv_fwd_{horizon}"
    log_rv_fwd = f"log_rv_fwd_{horizon}"

    con = duckdb.connect(database=":memory:")
    p = str(targets_path.resolve()).replace("\\", "/").replace("'", "''")

    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").df()
    col_names = [str(c).lower() for c in cols_df["column_name"].tolist()]

    required = ["date", rv_fwd.lower(), log_rv_fwd.lower()]
    missing = [c for c in required if c not in col_names]
    if missing:
        con.close()
        return {
            "passed": False,
            "error": f"Missing required columns: {missing}",
            "columns": col_names,
        }

    if expect_vix_column and "vix" not in col_names:
        con.close()
        return {
            "passed": False,
            "error": "Expected vix column (include_vix=True build)",
            "columns": col_names,
        }

    dup = con.execute(
        f"""
        SELECT COUNT(*) - COUNT(DISTINCT date) AS n_dup_dates
        FROM read_parquet('{p}')
        """
    ).fetchone()[0]

    order_check = con.execute(
        f"""
        SELECT COUNT(*) AS violations
        FROM (
            SELECT date,
                   LAG(date) OVER (ORDER BY date) AS prev_date
            FROM read_parquet('{p}')
        ) t
        WHERE prev_date IS NOT NULL AND date <= prev_date
        """
    ).fetchone()[0]

    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]

    bad_log = con.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM read_parquet('{p}')
        WHERE "{log_rv_fwd}" IS NOT NULL
          AND NOT isfinite("{log_rv_fwd}")
        """
    ).fetchone()[0]

    summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(CASE WHEN "{rv_fwd}" IS NULL THEN 1 ELSE 0 END) AS n_null_rv_fwd,
            SUM(CASE WHEN "{log_rv_fwd}" IS NULL THEN 1 ELSE 0 END) AS n_null_log_rv_fwd
        FROM read_parquet('{p}')
        """
    ).fetchdf()

    con.close()

    passed = (
        int(dup) == 0
        and int(order_check) == 0
        and int(bad_log) == 0
        and len(missing) == 0
    )

    issues: list[str] = []
    if int(dup) != 0:
        issues.append(f"duplicate dates: {int(dup)} extra rows vs distinct dates")
    if int(order_check) != 0:
        issues.append("date column not strictly increasing")
    if int(bad_log) != 0:
        issues.append(f"non-finite {log_rv_fwd} where non-null: {int(bad_log)} rows")

    return {
        "passed": passed,
        "summary": summary,
        "issues": issues,
        "n_rows": int(n_rows),
        "n_duplicate_date_rows": int(dup),
        "n_nonfinite_log_rv_fwd": int(bad_log),
    }
