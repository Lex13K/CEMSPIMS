"""Dataset packaging: splits, scalers, manifest for model training."""

from mss.dataset.config import DatasetConfig, load_dataset_config
from mss.dataset.manifest import package_scaled_and_manifest, write_manifest
from mss.dataset.scaler import apply_scaler, fit_and_save_scaler, fit_node_feature_scaler
from mss.dataset.splits import build_split_assignment, write_splits_parquet

__all__ = [
    "DatasetConfig",
    "load_dataset_config",
    "build_split_assignment",
    "write_splits_parquet",
    "fit_node_feature_scaler",
    "apply_scaler",
    "fit_and_save_scaler",
    "write_manifest",
    "package_scaled_and_manifest",
]
