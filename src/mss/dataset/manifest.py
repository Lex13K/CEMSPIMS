"""Dataset manifest and optional scaled node features for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mss.dataset.scaler import apply_scaler


def write_manifest(
    out_path: Path,
    *,
    splits_path: Path,
    node_features_path: Path,
    node_features_scaled_path: Path | None,
    edges_path: Path,
    universe_path: Path,
    scaler_path: Path,
    targets_path: Path,
    target_column: str,
    feature_columns: tuple[str, ...],
    scale_target: bool = False,
) -> Path:
    """Write dataset manifest JSON with absolute-path strings for portability within a machine."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "splits_path": splits_path.resolve().as_posix(),
        "node_features_path": node_features_path.resolve().as_posix(),
        "edges_path": edges_path.resolve().as_posix(),
        "universe_path": universe_path.resolve().as_posix(),
        "scaler_path": scaler_path.resolve().as_posix(),
        "targets_path": targets_path.resolve().as_posix(),
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "scale_target": scale_target,
    }
    if node_features_scaled_path is not None:
        manifest["node_features_scaled_path"] = node_features_scaled_path.resolve().as_posix()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def package_scaled_and_manifest(
    *,
    splits_path: Path,
    node_features_path: Path,
    scaler_path: Path,
    edges_path: Path,
    universe_path: Path,
    targets_path: Path,
    out_manifest_path: Path,
    out_scaled_path: Path | None,
    target_column: str,
    feature_columns: tuple[str, ...],
    write_scaled_features: bool,
) -> tuple[Path, Path | None]:
    """
    Optionally write scaled node features from scaler JSON, then write manifest.
    Returns (manifest_path, node_features_scaled_path or None).
    """
    if not splits_path.is_file():
        raise FileNotFoundError(f"Splits not found: {splits_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not node_features_path.is_file():
        raise FileNotFoundError(f"Node features not found: {node_features_path}")

    with scaler_path.open(encoding="utf-8") as f:
        scaler_params = json.load(f)

    scaled_path_for_manifest: Path | None = None
    if write_scaled_features:
        if out_scaled_path is None:
            raise ValueError("out_scaled_path required when write_scaled_features is True")
        nf = pd.read_parquet(node_features_path)
        nf_scaled = apply_scaler(nf, scaler_params, inplace=False)
        nf_scaled = nf_scaled.sort_values("date")
        out_scaled_path.parent.mkdir(parents=True, exist_ok=True)
        nf_scaled.to_parquet(out_scaled_path, index=False)
        scaled_path_for_manifest = out_scaled_path

    write_manifest(
        out_manifest_path,
        splits_path=splits_path,
        node_features_path=node_features_path,
        node_features_scaled_path=scaled_path_for_manifest,
        edges_path=edges_path,
        universe_path=universe_path,
        scaler_path=scaler_path,
        targets_path=targets_path,
        target_column=target_column,
        feature_columns=feature_columns,
        scale_target=False,
    )
    return (out_manifest_path, scaled_path_for_manifest)
