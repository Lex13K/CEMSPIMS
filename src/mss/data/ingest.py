from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

# ---------------------------------------------------------------------
# Progress: tqdm on stderr when available; else explicit i/n lines on stderr.
# Avoid wrapping the stage loop in tqdm while CRSP uses an inner bar + prints.
# ---------------------------------------------------------------------

try:
    from tqdm import tqdm as _tqdm_cls  # type: ignore
except Exception:  # pragma: no cover
    _tqdm_cls = None


def _progress_line(msg: str) -> None:
    """Emit a status line without corrupting an active tqdm bar."""
    if _tqdm_cls is not None:
        _tqdm_cls.write(msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr, flush=True)


def _track(iterable, *, desc: str):
    items = list(iterable)
    n = len(items)
    if _tqdm_cls is None:
        for i, x in enumerate(items, start=1):
            print(f"{desc}: {i}/{n}", file=sys.stderr, flush=True)
            yield x
        return
    yield from _tqdm_cls(
        items,
        desc=desc,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.25,
        smoothing=0.1,
    )


# ---------------------------------------------------------------------
# Path bundle (injectable for tests / config)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IngestPaths:
    raw_dir: Path
    interim_dir: Path

    @classmethod
    def from_resolved_config(cls, cfg: ResolvedConfig) -> IngestPaths:
        return cls(raw_dir=cfg.raw_dir, interim_dir=cfg.interim_dir)


# ---------------------------------------------------------------------
# Dataclasses for outputs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CrspIngestOutputs:
    crsp_parquet_dir: Path
    years_written: list[int]


@dataclass(frozen=True)
class Sp500IngestOutputs:
    sp500_parquet_path: Path


@dataclass(frozen=True)
class VixIngestOutputs:
    vix_parquet_path: Path


@dataclass(frozen=True)
class IngestOutputs:
    crsp_parquet_dir: Path
    sp500_parquet_path: Path
    vix_parquet_path: Path
    manifest_path: Path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )


def normalize_columns_lower(cols: Iterable[str]) -> list[str]:
    return [str(c).strip().lower() for c in cols]


def _read_csv_header(path: Path) -> list[str]:
    df0 = pd.read_csv(path, nrows=0)
    return list(df0.columns)


def _detect_column(header_cols: list[str], candidates: list[str]) -> str:
    lower = [c.lower() for c in header_cols]
    for cand in candidates:
        if cand.lower() in lower:
            return header_cols[lower.index(cand.lower())]
    raise KeyError(f"Could not find any of columns {candidates} in CSV header.")


def _write_json(path: Path, obj: dict) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def is_close_numeric(a: Any, b: Any, *, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    """Approximate equality for validation summaries (ints, floats, None/NaN, strings, dates)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        if (a is None and isinstance(b, float) and math.isnan(b)) or (
            b is None and isinstance(a, float) and math.isnan(a)
        ):
            return True
        return False
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return True
    if isinstance(a, (int, str)) or isinstance(b, (int, str)):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= (atol + rtol * abs(float(b)))
        except Exception:
            return a == b
    return a == b


# ---------------------------------------------------------------------
# CRSP ingestion
# ---------------------------------------------------------------------


def _crsp_typed_select_exprs(in_path: Path) -> tuple[str, list[str]]:
    header_cols = _read_csv_header(in_path)
    out_cols_lower = normalize_columns_lower(header_cols)

    date_col_raw = _detect_column(header_cols, ["date", "caldt", "tradedate"])

    want_int = {"permno", "dlstcd", "exchcd", "shrcd", "siccd", "hsiccd"}
    want_double = {"ret", "retx", "dlret", "dlretx", "prc", "vol", "shrout"}

    select_exprs: list[str] = []

    date_expr = f"""
        CASE
          WHEN regexp_matches(CAST("{date_col_raw}" AS VARCHAR), '^[0-9]{{8}}$')
            THEN strptime(CAST("{date_col_raw}" AS VARCHAR), '%Y%m%d')::DATE
          ELSE CAST("{date_col_raw}" AS DATE)
        END AS date
    """.strip()
    select_exprs.append(date_expr)

    for raw_name, lower_name in zip(header_cols, out_cols_lower):
        if raw_name == date_col_raw:
            continue
        if lower_name == "date":
            continue

        if lower_name in want_int:
            select_exprs.append(f'TRY_CAST("{raw_name}" AS BIGINT) AS "{lower_name}"')
        elif lower_name in want_double:
            select_exprs.append(f'TRY_CAST("{raw_name}" AS DOUBLE) AS "{lower_name}"')
        else:
            select_exprs.append(f'"{raw_name}" AS "{lower_name}"')

    return ", ".join(select_exprs), out_cols_lower


def ingest_crsp_wrds_to_parquet(paths: IngestPaths, *, overwrite: bool = False) -> CrspIngestOutputs:
    in_path = paths.raw_dir / "WRDS.csv"
    out_dir = paths.interim_dir / "crsp_parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw input: {in_path}")

    _assert_can_write(out_dir, overwrite=overwrite)
    if out_dir.exists() and overwrite:
        _remove_if_exists(out_dir)
    _safe_mkdir(out_dir)

    select_sql, out_cols_lower = _crsp_typed_select_exprs(in_path)

    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        CREATE VIEW crsp_raw AS
        SELECT {select_sql}
        FROM read_csv_auto(
          '{in_path.as_posix()}',
          header=true,
          all_varchar=true
        )
    """)

    years = con.execute("SELECT DISTINCT year(date) AS y FROM crsp_raw ORDER BY y").df()["y"].tolist()
    years_int = [int(y) for y in years if pd.notna(y)]

    if len(years_int) == 0:
        raise RuntimeError("No years detected from CRSP date parsing. Check the raw date column format.")

    has_permno = "permno" in out_cols_lower

    for y in _track(years_int, desc="CRSP: writing year partitions"):
        y_dir = out_dir / f"year={y}"
        _safe_mkdir(y_dir)
        out_file = y_dir / "data.parquet"
        order_clause = "ORDER BY date, permno" if has_permno else "ORDER BY date"

        con.execute(f"""
            COPY (
              SELECT *
              FROM crsp_raw
              WHERE year(date) = {y}
              {order_clause}
            )
            TO '{out_file.as_posix()}'
            (FORMAT PARQUET);
        """)

    con.close()

    return CrspIngestOutputs(crsp_parquet_dir=out_dir, years_written=years_int)


# ---------------------------------------------------------------------
# S&P 500 ingestion
# ---------------------------------------------------------------------


def ingest_sp500_returns_to_parquet(paths: IngestPaths, *, overwrite: bool = False) -> Sp500IngestOutputs:
    in_path = paths.raw_dir / "snp500_volatility.csv"
    out_path = paths.interim_dir / "sp500_returns.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw input: {in_path}")

    _assert_can_write(out_path, overwrite=overwrite)
    if out_path.exists() and overwrite:
        _remove_if_exists(out_path)
    _safe_mkdir(out_path.parent)

    df = pd.read_csv(in_path)
    df.columns = normalize_columns_lower(df.columns)

    date_col = None
    for cand in ["date", "caldt", "trading_date", "observation_date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise KeyError(f"Could not find a date column in {in_path.name}. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col], errors="raise").dt.date
    if date_col != "date":
        df = df.drop(columns=[date_col])

    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    return Sp500IngestOutputs(sp500_parquet_path=out_path)


# ---------------------------------------------------------------------
# VIX ingestion
# ---------------------------------------------------------------------


def ingest_vix_to_parquet(paths: IngestPaths, *, overwrite: bool = False) -> VixIngestOutputs:
    in_path = paths.raw_dir / "VIX.csv"
    out_path = paths.interim_dir / "vix.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw input: {in_path}")

    _assert_can_write(out_path, overwrite=overwrite)
    if out_path.exists() and overwrite:
        _remove_if_exists(out_path)
    _safe_mkdir(out_path.parent)

    df = pd.read_csv(in_path)
    df.columns = normalize_columns_lower(df.columns)

    date_col = None
    for cand in ["observation_date", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise KeyError(f"Could not find a date column in {in_path.name}. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col], errors="raise").dt.date
    if date_col != "date":
        df = df.drop(columns=[date_col])

    if "vixcls" in df.columns:
        df = df.rename(columns={"vixcls": "vix"})
    elif "vix" not in df.columns:
        raise KeyError(f"Could not find VIX column (expected VIXCLS) in {in_path.name}. Columns: {list(df.columns)}")

    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    return VixIngestOutputs(vix_parquet_path=out_path)


# ---------------------------------------------------------------------
# Ingest orchestration (`data.prepare` / ingest step)
# ---------------------------------------------------------------------


def ingest_raw_to_parquet(paths: IngestPaths, *, overwrite: bool = False) -> IngestOutputs:
    tasks = [
        ("CRSP/WRDS", "crsp"),
        ("S&P 500 series", "sp500"),
        ("VIX", "vix"),
    ]

    results: dict[str, Any] = {}

    n_tasks = len(tasks)
    for i, (label, key) in enumerate(tasks, start=1):
        _progress_line(f"Ingest {i}/{n_tasks} — {label}")
        if key == "crsp":
            results["crsp"] = ingest_crsp_wrds_to_parquet(paths, overwrite=overwrite)
        elif key == "sp500":
            results["sp500"] = ingest_sp500_returns_to_parquet(paths, overwrite=overwrite)
        elif key == "vix":
            results["vix"] = ingest_vix_to_parquet(paths, overwrite=overwrite)
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown task key: {key}")

    crsp_out: CrspIngestOutputs = results["crsp"]
    sp500_out: Sp500IngestOutputs = results["sp500"]
    vix_out: VixIngestOutputs = results["vix"]

    manifest_path = paths.interim_dir / "ingest_manifest.json"

    manifest = {
        "stage": "01_ingest_raw_data",
        "created_at": _utc_now_iso(),
        "inputs": {
            "crsp_wrds_csv": str((paths.raw_dir / "WRDS.csv").as_posix()),
            "sp500_csv": str((paths.raw_dir / "snp500_volatility.csv").as_posix()),
            "vix_csv": str((paths.raw_dir / "VIX.csv").as_posix()),
        },
        "outputs": {
            "crsp_parquet_dir": str(crsp_out.crsp_parquet_dir.as_posix()),
            "sp500_parquet_path": str(sp500_out.sp500_parquet_path.as_posix()),
            "vix_parquet_path": str(vix_out.vix_parquet_path.as_posix()),
            "manifest_path": str(manifest_path.as_posix()),
        },
        "crsp": {
            "partition": "year",
            "years_written": crsp_out.years_written,
            "n_years": int(len(crsp_out.years_written)),
        },
        "flags": {"overwrite": bool(overwrite)},
    }

    _write_json(manifest_path, manifest)

    return IngestOutputs(
        crsp_parquet_dir=crsp_out.crsp_parquet_dir,
        sp500_parquet_path=sp500_out.sp500_parquet_path,
        vix_parquet_path=vix_out.vix_parquet_path,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _table_columns(con: duckdb.DuckDBPyConnection, relation: str) -> list[str]:
    df = con.execute(f"DESCRIBE {relation}").df()
    return [str(c).lower() for c in df["column_name"].tolist()]


def _summary_query_sql(relation: str, numeric_cols: list[str]) -> str:
    parts: list[str] = [
        f"'{relation}' AS relation",
        "COUNT(*) AS n_rows",
        "COUNT(DISTINCT date) AS n_dates",
        "MIN(date) AS min_date",
        "MAX(date) AS max_date",
    ]

    for c in numeric_cols:
        parts += [
            f"COUNT({c}) AS {c}__n_nonnull",
            f"MIN({c}) AS {c}__min",
            f"MAX({c}) AS {c}__max",
            f"AVG({c}) AS {c}__mean",
        ]
    return f"SELECT {', '.join(parts)} FROM {relation}"


def _summarize_relation(con: duckdb.DuckDBPyConnection, relation: str, preferred_numeric: list[str]) -> dict[str, Any]:
    cols = _table_columns(con, relation)
    numeric_cols = [c for c in preferred_numeric if c in cols]
    q = _summary_query_sql(relation, numeric_cols)
    row = con.execute(q).df().iloc[0].to_dict()
    return row


def validate_ingest(paths: IngestPaths, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    crsp_csv = paths.raw_dir / "WRDS.csv"
    sp500_csv = paths.raw_dir / "snp500_volatility.csv"
    vix_csv = paths.raw_dir / "VIX.csv"

    crsp_parquet_dir = paths.interim_dir / "crsp_parquet"
    sp500_parquet = paths.interim_dir / "sp500_returns.parquet"
    vix_parquet = paths.interim_dir / "vix.parquet"

    for p in [crsp_csv, sp500_csv, vix_csv, crsp_parquet_dir, sp500_parquet, vix_parquet]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path for validation: {p}")

    con = duckdb.connect(database=":memory:")

    select_sql, _ = _crsp_typed_select_exprs(crsp_csv)

    con.execute(f"""
        CREATE VIEW crsp_csv_typed AS
        SELECT {select_sql}
        FROM read_csv_auto(
          '{crsp_csv.as_posix()}',
          header=true,
          all_varchar=true
        )
    """)

    con.execute(f"""
        CREATE VIEW crsp_parquet AS
        SELECT *
        FROM read_parquet('{(crsp_parquet_dir / "**" / "*.parquet").as_posix()}')
    """)

    crsp_numeric_pref = ["permno", "ret", "retx", "dlret", "dlretx", "prc", "vol", "shrout", "dlstcd", "exchcd", "shrcd"]
    crsp_csv_sum = _summarize_relation(con, "crsp_csv_typed", crsp_numeric_pref)
    crsp_pq_sum = _summarize_relation(con, "crsp_parquet", crsp_numeric_pref)

    sp500_header = _read_csv_header(sp500_csv)
    sp500_cols_lower = normalize_columns_lower(sp500_header)

    date_col_raw = None
    for cand in ["date", "caldt", "trading_date", "observation_date"]:
        if cand in [c.lower() for c in sp500_header]:
            date_col_raw = sp500_header[[c.lower() for c in sp500_header].index(cand)]
            break
    if date_col_raw is None:
        raise KeyError(f"Could not find a date column in {sp500_csv.name} during validation.")

    sp500_select = [f'CAST("{date_col_raw}" AS DATE) AS date']
    for raw_name, lower_name in zip(sp500_header, sp500_cols_lower):
        if raw_name == date_col_raw:
            continue
        if lower_name == "date":
            continue
        sp500_select.append(f'"{raw_name}" AS "{lower_name}"')

    con.execute(f"""
        CREATE VIEW sp500_csv_norm AS
        SELECT {", ".join(sp500_select)}
        FROM read_csv_auto('{sp500_csv.as_posix()}', header=true, all_varchar=false)
    """)
    con.execute(f"""
        CREATE VIEW sp500_parquet AS
        SELECT *
        FROM read_parquet('{sp500_parquet.as_posix()}')
    """)

    sp500_numeric_pref = ["sprtrn", "ret", "return", "sp500_return", "spx_ret", "spxreturn", "index", "level"]
    sp500_csv_sum = _summarize_relation(con, "sp500_csv_norm", sp500_numeric_pref)
    sp500_pq_sum = _summarize_relation(con, "sp500_parquet", sp500_numeric_pref)

    vix_header = _read_csv_header(vix_csv)
    vix_cols_lower = normalize_columns_lower(vix_header)

    vix_date_raw = None
    for cand in ["observation_date", "date"]:
        if cand in [c.lower() for c in vix_header]:
            vix_date_raw = vix_header[[c.lower() for c in vix_header].index(cand)]
            break
    if vix_date_raw is None:
        raise KeyError(f"Could not find a date column in {vix_csv.name} during validation.")

    vix_val_raw = None
    for cand in ["vixcls", "vix"]:
        if cand in [c.lower() for c in vix_header]:
            vix_val_raw = vix_header[[c.lower() for c in vix_header].index(cand)]
            break
    if vix_val_raw is None:
        raise KeyError(f"Could not find VIX value column (VIXCLS) in {vix_csv.name} during validation.")

    con.execute(f"""
        CREATE VIEW vix_csv_norm AS
        SELECT
          CAST("{vix_date_raw}" AS DATE) AS date,
          TRY_CAST("{vix_val_raw}" AS DOUBLE) AS vix
        FROM read_csv_auto('{vix_csv.as_posix()}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE VIEW vix_parquet AS
        SELECT *
        FROM read_parquet('{vix_parquet.as_posix()}')
    """)

    vix_numeric_pref = ["vix"]
    vix_csv_sum = _summarize_relation(con, "vix_csv_norm", vix_numeric_pref)
    vix_pq_sum = _summarize_relation(con, "vix_parquet", vix_numeric_pref)

    con.close()

    comparisons = [
        ("CRSP", crsp_csv_sum, crsp_pq_sum),
        ("SP500", sp500_csv_sum, sp500_pq_sum),
        ("VIX", vix_csv_sum, vix_pq_sum),
    ]

    failures: list[str] = []
    for name, a, b in comparisons:
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            if k == "relation":
                continue
            av = a.get(k)
            bv = b.get(k)
            ok = is_close_numeric(av, bv, atol=atol, rtol=rtol)
            if not ok:
                failures.append(f"{name}: {k}: csv={av} parquet={bv}")

    if failures:
        msg = "\n".join(failures[:200])
        raise AssertionError(
            "Ingest validation failed (CSV vs Parquet summary mismatch).\n"
            f"Showing up to 200 mismatches:\n{msg}"
        )

    _progress_line("Ingest validation passed: CSV and Parquet summaries match within tolerance.")
