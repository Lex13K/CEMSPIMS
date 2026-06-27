"""Validate graph artifacts against data contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def check_universe(path: Path) -> dict:
    issues: list[str] = []
    if not path.is_file():
        return {"passed": False, "issues": [f"missing file: {path}"]}
    df = pd.read_parquet(path)
    req = {"date", "permno", "rank", "mcap"}
    if not req.issubset(df.columns):
        issues.append(f"missing columns: {req - set(df.columns)}")
    if not issues:
        dup = df.duplicated(subset=["date", "permno"]).sum()
        if dup:
            issues.append(f"duplicate (date, permno) rows: {dup}")
    return {"passed": len(issues) == 0, "issues": issues}


def check_node_features(path: Path, *, feature_cols: tuple[str, ...] | None = None) -> dict:
    issues: list[str] = []
    if not path.is_file():
        return {"passed": False, "issues": [f"missing file: {path}"]}
    df = pd.read_parquet(path)
    if "date" not in df.columns or "permno" not in df.columns:
        issues.append("need date and permno columns")
        return {"passed": False, "issues": issues}
    cols = feature_cols or tuple(c for c in df.columns if c not in ("date", "permno"))
    if not cols:
        issues.append("no feature columns")
    dup = df.duplicated(subset=["date", "permno"]).sum()
    if dup:
        issues.append(f"duplicate (date, permno) rows: {dup}")
    for c in cols:
        if df[c].isna().any():
            issues.append(f"null values in column {c!r}")
    return {"passed": len(issues) == 0, "issues": issues}


def check_edges(path: Path, *, allow_self_loops: bool = False) -> dict:
    issues: list[str] = []
    if not path.is_file():
        return {"passed": False, "issues": [f"missing file: {path}"]}
    df = pd.read_parquet(path)
    req = {"date", "src", "dst", "weight"}
    if not req.issubset(df.columns):
        issues.append(f"missing columns: {req - set(df.columns)}")
        return {"passed": False, "issues": issues}
    if len(df) > 0:
        dup = df.duplicated(subset=["date", "src", "dst"]).sum()
        if dup:
            issues.append(f"duplicate (date, src, dst) rows: {dup}")
        if not allow_self_loops and (df["src"] == df["dst"]).any():
            issues.append("self-loops present (forbidden by default)")
        w = df["weight"]
        if not np.isfinite(w.astype(float)).all():
            issues.append("non-finite weights")
    return {"passed": len(issues) == 0, "issues": issues}
