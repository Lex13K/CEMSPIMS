"""Load and validate dataset package manifest JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_KEYS = (
    "splits_path",
    "node_features_path",
    "edges_path",
    "universe_path",
    "scaler_path",
    "targets_path",
    "target_column",
    "feature_columns",
)


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    with path.open(encoding="utf-8") as f:
        m = json.load(f)
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in m]
    if missing:
        raise ValueError(f"Manifest missing keys: {missing}")
    fc = m.get("feature_columns")
    if not isinstance(fc, list) or not fc:
        raise ValueError("feature_columns must be a non-empty list")
    for k in (
        "splits_path",
        "node_features_path",
        "edges_path",
        "universe_path",
        "targets_path",
        "scaler_path",
    ):
        p = Path(m[k])
        if not p.is_file():
            raise FileNotFoundError(f"Manifest path missing for {k}: {p}")
    scaled = m.get("node_features_scaled_path")
    if scaled:
        sp = Path(scaled)
        if not sp.is_file():
            raise FileNotFoundError(f"Manifest path missing for node_features_scaled_path: {sp}")
    return m


def node_features_path_from_manifest(m: dict[str, Any]) -> Path:
    p = m.get("node_features_scaled_path") or m["node_features_path"]
    return Path(p)
