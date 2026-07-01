"""`[model.train]` TOML block and config fingerprint for resume vs config drift."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mss.config.fingerprints import (
    current_model_train_fingerprint,
    fingerprint_toml_table,
)


@dataclass(frozen=True)
class ModelTrainConfig:
    seed: int
    max_epochs: int
    lr: float
    early_stopping_patience: int
    hidden_channels: int
    num_gnn_layers: int
    model_type: str
    pooling: str
    dataloader_num_workers: int
    batch_size: int
    show_progress: bool
    verbose: bool
    device: str
    pin_memory: bool
    graph_cache_dir: str | None  # None if disabled
    parquet_date_pushdown: bool  # Parquet row-group filters on `date` when reading graphs
    loss: str  # mse_log | mae_log | smooth_l1_log | qlike_level | hybrid_mse_qlike
    loss_eps: float  # clamp for qlike_level (level space)
    smooth_l1_beta: float  # SmoothL1Loss beta for smooth_l1_log
    loss_alpha: float  # hybrid_mse_qlike only: weight on MSE (0,1)
    qlike_pred_transform: str  # exp | softplus — level f from pred for QLIKE branch
    qlike_level_floor: float | None  # minimum level f; None -> use loss_eps
    warm_start_checkpoint: str | None  # optional path to donor checkpoint.pt (best weights)
    fit_log_calibration: bool  # fit y_true_log ~ a + b * y_pred on val after training
    use_edge_weights: bool  # pass correlation magnitudes into message passing
    gat_num_heads: int  # GAT attention heads when model_type=gat


def model_train_table_from_parsed(data: dict[str, Any]) -> dict[str, Any]:
    """Return the `[model.train]` table as a plain dict (tomllib nests as model.train)."""
    return dict(data.get("model", {}).get("train", {}) or {})


def fingerprint_model_train_table(mt: dict[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of the training table only."""
    return fingerprint_toml_table(mt)


def load_model_train_config(config_path: Path) -> ModelTrainConfig:
    raw = config_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    mt = model_train_table_from_parsed(data)

    gcd = mt.get("graph_cache_dir")
    if gcd is None or str(gcd).strip() == "":
        graph_cache_dir = None
    else:
        graph_cache_dir = str(gcd).strip()

    from mss.model_train.loss_ids import validate_training_loss_id
    from mss.model_train.qlike_mapping import validate_qlike_pred_transform

    loss_raw = validate_training_loss_id(str(mt.get("loss", "mse_log")))
    loss_eps = float(mt.get("loss_eps", 1e-12))
    smooth_l1_beta = float(mt.get("smooth_l1_beta", 1.0))
    loss_alpha = float(mt.get("loss_alpha", 0.5))
    qlike_raw = str(mt.get("qlike_pred_transform", "exp")).strip().lower()
    qlike_pred_transform = validate_qlike_pred_transform(qlike_raw)
    qlf = mt.get("qlike_level_floor")
    qlike_level_floor = None if qlf is None or str(qlf).strip() == "" else float(qlf)
    wsc = mt.get("warm_start_checkpoint")
    warm_start_checkpoint = None if wsc is None or str(wsc).strip() == "" else str(wsc).strip()
    if loss_eps <= 0:
        raise ValueError("[model.train].loss_eps must be positive")
    if smooth_l1_beta <= 0:
        raise ValueError("[model.train].smooth_l1_beta must be positive")
    if loss_raw == "hybrid_mse_qlike":
        if not (0.0 < loss_alpha < 1.0):
            raise ValueError("[model.train].loss_alpha must be in (0, 1) when loss is hybrid_mse_qlike")
    elif loss_alpha != 0.5:
        pass  # ignore non-default when not hybrid
    if qlike_level_floor is not None and qlike_level_floor <= 0:
        raise ValueError("[model.train].qlike_level_floor must be positive when set")

    return ModelTrainConfig(
        seed=int(mt.get("seed", 42)),
        max_epochs=int(mt.get("max_epochs", 50)),
        lr=float(mt.get("lr", 0.001)),
        early_stopping_patience=int(mt.get("early_stopping_patience", 8)),
        hidden_channels=int(mt.get("hidden_channels", 64)),
        num_gnn_layers=int(mt.get("num_gnn_layers", 2)),
        model_type=str(mt.get("model_type", "graphsage")).lower(),
        pooling=str(mt.get("pooling", "mean")),
        dataloader_num_workers=int(mt.get("dataloader_num_workers", 0)),
        batch_size=int(mt.get("batch_size", 16)),
        show_progress=bool(mt.get("show_progress", True)),
        verbose=bool(mt.get("verbose", True)),
        device=str(mt.get("device", "auto")),
        pin_memory=bool(mt.get("pin_memory", False)),
        graph_cache_dir=graph_cache_dir,
        parquet_date_pushdown=bool(mt.get("parquet_date_pushdown", True)),
        loss=loss_raw,
        loss_eps=loss_eps,
        smooth_l1_beta=smooth_l1_beta,
        loss_alpha=loss_alpha,
        qlike_pred_transform=qlike_pred_transform,
        qlike_level_floor=qlike_level_floor,
        warm_start_checkpoint=warm_start_checkpoint,
        fit_log_calibration=bool(mt.get("fit_log_calibration", False)),
        use_edge_weights=bool(mt.get("use_edge_weights", False)),
        gat_num_heads=int(mt.get("gat_num_heads", 1)),
    )