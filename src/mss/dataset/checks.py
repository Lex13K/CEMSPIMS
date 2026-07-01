"""Contract checks for dataset packaging artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mss.dataset.config import DatasetConfig, load_dataset_config
from mss.dataset.splits import build_split_assignment
from mss.io.config import ResolvedConfig


def check_splits(path: Path) -> dict[str, Any]:
    """Validate splits.parquet schema and split labels."""
    result: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        result["issues"].append(f"missing file: {path}")
        return result
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        result["issues"].append(f"read failed: {e}")
        return result
    for col in ("date", "split"):
        if col not in df.columns:
            result["issues"].append(f"missing column: {col}")
    if result["issues"]:
        return result
    allowed = {"train", "val", "test"}
    bad = set(df["split"].astype(str).unique()) - allowed
    if bad:
        result["issues"].append(f"invalid split values: {bad}")
    if df["date"].duplicated().any():
        result["issues"].append("duplicate dates in splits")
    result["passed"] = len(result["issues"]) == 0
    return result


def check_scaler_json(path: Path, feature_columns: tuple[str, ...]) -> dict[str, Any]:
    """Validate scaler_params.json structure."""
    result: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        result["issues"].append(f"missing file: {path}")
        return result
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        result["issues"].append(f"invalid JSON: {e}")
        return result
    fc = data.get("feature_columns")
    if fc != list(feature_columns):
        result["issues"].append(
            f"feature_columns mismatch: got {fc}, expected {list(feature_columns)}"
        )
    params = data.get("params") or {}
    for col in feature_columns:
        if col not in params:
            result["issues"].append(f"missing scaler params for column: {col}")
        else:
            p = params[col]
            if "mean" not in p or "scale" not in p:
                result["issues"].append(f"invalid params for {col}")
    result["passed"] = len(result["issues"]) == 0
    return result


def check_manifest(path: Path) -> dict[str, Any]:
    """Validate manifest JSON: required keys, paths exist, targets column present."""
    result: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        result["issues"].append(f"missing file: {path}")
        return result
    try:
        with path.open(encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        result["issues"].append(f"invalid JSON: {e}")
        return result

    required = [
        "splits_path",
        "node_features_path",
        "edges_path",
        "universe_path",
        "scaler_path",
        "targets_path",
        "target_column",
        "feature_columns",
    ]
    for k in required:
        if k not in m:
            result["issues"].append(f"missing manifest key: {k}")

    if result["issues"]:
        return result

    for key in (
        "splits_path",
        "node_features_path",
        "edges_path",
        "universe_path",
        "scaler_path",
        "targets_path",
    ):
        p = Path(m[key])
        if not p.is_file():
            result["issues"].append(f"path not found for {key}: {p}")

    nfs = m.get("node_features_scaled_path")
    if nfs:
        p = Path(nfs)
        if not p.is_file():
            result["issues"].append(f"path not found for node_features_scaled_path: {p}")

    tc = str(m["target_column"])
    tpath = Path(m["targets_path"])
    if tpath.is_file() and not result["issues"]:
        try:
            cols = pd.read_parquet(tpath, columns=[tc])
        except Exception as e:
            result["issues"].append(f"targets missing column {tc!r}: {e}")
        else:
            if tc not in cols.columns:
                result["issues"].append(f"targets missing column: {tc}")

    result["passed"] = len(result["issues"]) == 0
    return result


def splits_matches_universe_and_config(
    cfg: ResolvedConfig,
    splits_path: Path,
    universe_path: Path,
    dc: DatasetConfig,
) -> bool:
    """True if splits.parquet equals recomputation from universe dates and split boundaries."""
    if not splits_path.is_file() or not universe_path.is_file():
        return False
    try:
        ud = pd.read_parquet(universe_path, columns=["date"])
        dates = ud["date"].drop_duplicates().sort_values()
        expected = build_split_assignment(
            dates, dc.train_end, dc.val_end, dc.test_end
        )
        actual = pd.read_parquet(splits_path)
        expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
        actual["date"] = pd.to_datetime(actual["date"]).dt.normalize()
        expected = expected.sort_values("date").reset_index(drop=True)
        actual = actual.sort_values("date").reset_index(drop=True)
        if len(expected) != len(actual):
            return False
        if not expected["split"].astype(str).equals(actual["split"].astype(str)):
            return False
        return bool(expected["date"].equals(actual["date"]))
    except Exception:
        return False


def splits_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    paths = splits_paths(cfg)
    if not paths["splits"].is_file():
        return False
    ch = check_splits(paths["splits"])
    if not ch.get("passed"):
        return False
    try:
        dc = load_dataset_config(cfg.source_config_path)
    except OSError:
        return False
    return splits_matches_universe_and_config(
        cfg, paths["splits"], paths["universe"], dc
    )


def scaler_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    paths = splits_paths(cfg)
    if not paths["scaler"].is_file():
        return False
    try:
        dc = load_dataset_config(cfg.source_config_path)
    except OSError:
        return False
    ch = check_scaler_json(paths["scaler"], dc.feature_columns)
    return bool(ch.get("passed"))


def manifest_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    paths = splits_paths(cfg)
    if not paths["manifest"].is_file():
        return False
    try:
        dc = load_dataset_config(cfg.source_config_path)
    except OSError:
        return False
    ch = check_manifest(paths["manifest"])
    if not ch.get("passed"):
        return False
    with paths["manifest"].open(encoding="utf-8") as f:
        m = json.load(f)
    if dc.write_scaled_features:
        nfs = m.get("node_features_scaled_path")
        if not nfs or not Path(nfs).is_file():
            return False
    return True


def splits_paths(cfg: ResolvedConfig) -> dict[str, Path]:
    run_interim = cfg.run_interim_dir
    ddir = run_interim / "dataset"
    gdir = run_interim / "graphs"
    return {
        "dataset_dir": ddir,
        "splits": ddir / "splits.parquet",
        "scaler": ddir / "scaler_params.json",
        "manifest": ddir / "manifest.json",
        "scaled_nf": ddir / "node_features_scaled.parquet",
        "universe": gdir / "universe.parquet",
        "node_features": gdir / "node_features.parquet",
        "edges": gdir / "edges.parquet",
        "targets": cfg.shared_interim_dir / "targets.parquet",
    }
