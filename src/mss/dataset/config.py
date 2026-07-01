"""Dataset packaging parameters from TOML `[dataset]` (and default target from `[graph.output]`)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    train_end: str
    val_end: str
    test_end: str
    scaler_type: str
    feature_columns: tuple[str, ...]
    write_scaled_features: bool
    target_column: str


def load_dataset_config(config_path: Path) -> DatasetConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    d = data.get("dataset") or {}
    g_out = (data.get("graph") or {}).get("output") or {}

    train_end = str(d.get("train_end", "2014-12-31"))
    val_end = str(d.get("val_end", "2018-12-31"))
    test_end = str(d.get("test_end", "2024-12-31"))
    scaler_type = str(d.get("scaler_type", "standard"))
    fc = d.get("feature_columns")
    if fc is None:
        feature_columns = ("rolling_mean", "rolling_vol")
    else:
        feature_columns = tuple(str(x) for x in list(fc))
    write_scaled_features = bool(d.get("write_scaled_features", True))
    tc = d.get("target_column")
    if tc is not None:
        target_column = str(tc)
    else:
        target_column = str(g_out.get("target_column", "log_rv_fwd_30cal"))

    return DatasetConfig(
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        scaler_type=scaler_type,
        feature_columns=feature_columns,
        write_scaled_features=write_scaled_features,
        target_column=target_column,
    )
