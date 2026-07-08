"""List column names for raw CSV inputs.

By default this inspects the canonical raw files under ``data/shared/raw``:
- WRDS.csv
- snp500_volatility.csv
- VIX.csv

Usage:
    python scripts/list_raw_columns.py
    python scripts/list_raw_columns.py --raw-dir data/shared/raw --files WRDS.csv VIX.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = ("WRDS.csv", "snp500_volatility.csv", "VIX.csv")
DATE_COLUMN_BY_FILE = {
    "wrds.csv": "date",
    "snp500_volatility.csv": "caldt",
    "vix.csv": "observation_date",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read raw CSV files and print their columns."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/shared/raw"),
        help="Directory containing raw CSV files (default: data/shared/raw).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="CSV filenames to inspect (default: WRDS.csv snp500_volatility.csv VIX.csv).",
    )
    return parser


def list_columns(csv_path: Path) -> list[str]:
    # Read only headers for speed and low memory use on large WRDS files.
    frame = pd.read_csv(csv_path, nrows=0)
    return frame.columns.tolist()


def get_date_bounds(csv_path: Path, date_column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.read_csv(csv_path, usecols=[date_column])[date_column]
    parsed = pd.to_datetime(dates, errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        raise ValueError(f"date column '{date_column}' has no parseable values")
    return parsed.min(), parsed.max()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    filenames: list[str] = args.files

    print(f"Raw directory: {raw_dir.resolve()}")
    print()

    missing = False
    for name in filenames:
        csv_path = raw_dir / name
        print(f"File: {csv_path}")

        if not csv_path.exists():
            print("  [missing] file not found")
            print()
            missing = True
            continue

        try:
            columns = list_columns(csv_path)
        except Exception as exc:  # pragma: no cover - utility script
            print(f"  [error] could not read columns: {exc}")
            print()
            missing = True
            continue

        if not columns:
            print("  [warning] no columns found")
        else:
            for idx, col in enumerate(columns, start=1):
                print(f"  {idx:>3}. {col}")

        date_column = DATE_COLUMN_BY_FILE.get(name.lower())
        if date_column is not None:
            if date_column not in columns:
                print(f"  [warning] expected date column '{date_column}' not found")
                missing = True
            else:
                try:
                    min_date, max_date = get_date_bounds(csv_path, date_column)
                    print(f"  date column: {date_column}")
                    print(f"  min date: {min_date.date().isoformat()}")
                    print(f"  max date: {max_date.date().isoformat()}")
                except Exception as exc:  # pragma: no cover - utility script
                    print(f"  [error] could not compute date bounds: {exc}")
                    missing = True
        print()

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
