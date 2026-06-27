"""Train-only scaling for node features (standard = mean/std)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def fit_node_feature_scaler(
    node_features: pd.DataFrame,
    train_dates: pd.Series | set,
    feature_columns: tuple[str, ...],
    *,
    scaler_type: str = "standard",
) -> dict[str, Any]:
    """
    Fit scaler on train node features only.
    Returns a dict suitable for JSON: type, feature_columns, and per-column mean/scale.
    """
    if not feature_columns:
        return {"type": scaler_type, "feature_columns": [], "params": {}}
    train_dates_set = {pd.to_datetime(d) for d in train_dates}
    nf = node_features.copy()
    nf["date"] = pd.to_datetime(nf["date"])
    train_nf = nf[nf["date"].isin(train_dates_set)]
    if train_nf.empty:
        raise ValueError(
            "No rows in node features for split==train dates: adjust dataset split boundaries "
            "or ensure universe dates include at least one training date."
        )
    missing = [c for c in feature_columns if c not in train_nf.columns]
    if missing:
        raise ValueError(f"Node features missing columns: {missing}")
    params: dict[str, dict[str, float]] = {}
    for col in feature_columns:
        vals = train_nf[col].values.astype(np.float64)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        if std <= 0 or not np.isfinite(std):
            std = 1.0
        params[col] = {"mean": mean, "scale": std}
    return {
        "type": scaler_type,
        "feature_columns": list(feature_columns),
        "params": params,
    }


def apply_scaler(
    node_features: pd.DataFrame,
    scaler_params: dict[str, Any],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply fitted standard scaler: (x - mean) / scale."""
    df = node_features if inplace else node_features.copy()
    for col in scaler_params.get("feature_columns", []):
        if col not in df.columns:
            continue
        p = scaler_params.get("params", {}).get(col, {})
        mean = p.get("mean", 0.0)
        scale = p.get("scale", 1.0)
        if scale <= 0:
            scale = 1.0
        df[col] = (df[col].astype(np.float64) - mean) / scale
    return df


def fit_and_save_scaler(
    node_features_path: Path,
    splits_path: Path,
    out_path: Path,
    *,
    feature_columns: tuple[str, ...],
    scaler_type: str = "standard",
) -> Path:
    """Load node features and splits, fit on train dates, write JSON."""
    if not node_features_path.is_file():
        raise FileNotFoundError(f"Node features not found: {node_features_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Splits not found: {splits_path}")
    nf = pd.read_parquet(node_features_path)
    sp = pd.read_parquet(splits_path)
    train_dates = sp[sp["split"] == "train"]["date"]
    params = fit_node_feature_scaler(
        nf, train_dates, feature_columns, scaler_type=scaler_type
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return out_path
