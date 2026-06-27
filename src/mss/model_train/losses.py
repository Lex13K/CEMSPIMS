"""Training objectives for `model.train` (registry + PyTorch modules)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mss.model_train.config import ModelTrainConfig
from mss.model_train.loss_ids import validate_training_loss_id
from mss.model_train.qlike_mapping import validate_qlike_pred_transform

__all__ = [
    "QLikeLevelLoss",
    "HybridMseQlikeLoss",
    "build_training_loss",
    "validate_training_loss_id",
]


def _level_f_torch(
    pred: torch.Tensor,
    *,
    transform: str,
    level_floor: float,
    loss_eps: float,
) -> torch.Tensor:
    t = validate_qlike_pred_transform(transform)
    floor = float(level_floor)
    if t == "exp":
        return torch.clamp(torch.exp(pred), min=max(floor, loss_eps))
    return torch.clamp(F.softplus(pred) + floor, min=max(floor, loss_eps))


class QLikeLevelLoss(nn.Module):
    """QLIKE on level variance: mean(log(f) + y/f); f from pred via exp or softplus + floor."""

    def __init__(
        self,
        *,
        loss_eps: float,
        qlike_pred_transform: str,
        qlike_level_floor: float,
    ) -> None:
        super().__init__()
        self.loss_eps = loss_eps
        self.qlike_pred_transform = validate_qlike_pred_transform(qlike_pred_transform)
        self.qlike_level_floor = float(qlike_level_floor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(torch.exp(target), min=self.loss_eps)
        f = _level_f_torch(
            pred,
            transform=self.qlike_pred_transform,
            level_floor=self.qlike_level_floor,
            loss_eps=self.loss_eps,
        )
        return torch.mean(torch.log(f) + (y / f))


class HybridMseQlikeLoss(nn.Module):
    """``alpha * MSE(log) + (1-alpha) * QLIKE(level)``."""

    def __init__(
        self,
        *,
        alpha: float,
        loss_eps: float,
        qlike_pred_transform: str,
        qlike_level_floor: float,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("hybrid_mse_qlike requires 0 < loss_alpha < 1")
        self.alpha = float(alpha)
        self._qlike = QLikeLevelLoss(
            loss_eps=loss_eps,
            qlike_pred_transform=qlike_pred_transform,
            qlike_level_floor=qlike_level_floor,
        )
        self._mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self._mse(pred, target)
        q = self._qlike(pred, target)
        return self.alpha * mse + (1.0 - self.alpha) * q


def build_training_loss(mcfg: ModelTrainConfig) -> nn.Module:
    """Build the loss used for train/val and early stopping."""
    lid = validate_training_loss_id(mcfg.loss)
    floor = mcfg.qlike_level_floor if mcfg.qlike_level_floor is not None else mcfg.loss_eps
    if lid == "mse_log":
        return nn.MSELoss()
    if lid == "mae_log":
        return nn.L1Loss()
    if lid == "smooth_l1_log":
        return nn.SmoothL1Loss(beta=mcfg.smooth_l1_beta)
    if lid == "qlike_level":
        return QLikeLevelLoss(
            loss_eps=mcfg.loss_eps,
            qlike_pred_transform=mcfg.qlike_pred_transform,
            qlike_level_floor=floor,
        )
    if lid == "hybrid_mse_qlike":
        return HybridMseQlikeLoss(
            alpha=mcfg.loss_alpha,
            loss_eps=mcfg.loss_eps,
            qlike_pred_transform=mcfg.qlike_pred_transform,
            qlike_level_floor=floor,
        )
    raise AssertionError(f"unhandled loss id: {lid}")
