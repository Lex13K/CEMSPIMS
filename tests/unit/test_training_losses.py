"""Training loss modules (requires torch; skipped if not installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mss.evaluation.metrics import aggregate_test_metric_value, qlike_level
from mss.model_train.config import load_model_train_config
from mss.model_train.losses import HybridMseQlikeLoss, QLikeLevelLoss, build_training_loss


def _mt_cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "train.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_mse_log_matches_torch_mse(tmp_path: Path) -> None:
    p = _mt_cfg(tmp_path, "[model.train]\nloss = \"mse_log\"\n")
    mcfg = load_model_train_config(p)
    loss_fn = build_training_loss(mcfg)
    pred = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.1, 1.2, 1.8])
    l = loss_fn(pred, target)
    l.backward()
    assert pred.grad is not None
    exp = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(l, exp)


def test_qlike_torch_matches_numpy(tmp_path: Path) -> None:
    p = _mt_cfg(tmp_path, "[model.train]\nloss = \"qlike_level\"\nloss_eps = 1e-12\n")
    mcfg = load_model_train_config(p)
    loss_fn = build_training_loss(mcfg)
    assert isinstance(loss_fn, QLikeLevelLoss)
    pred = torch.tensor([-1.0, 0.0, 0.5], requires_grad=True)
    target = torch.tensor([-0.5, 0.1, 0.2])
    lt = loss_fn(pred, target)
    lt.backward()
    np_val = aggregate_test_metric_value(
        "qlike_level",
        target.detach().numpy(),
        pred.detach().numpy(),
        loss_eps=mcfg.loss_eps,
        smooth_l1_beta=mcfg.smooth_l1_beta,
        qlike_pred_transform=mcfg.qlike_pred_transform,
        qlike_level_floor=mcfg.qlike_level_floor,
    )
    assert abs(float(lt.item()) - np_val) < 1e-5


def test_qlike_level_module_matches_metrics_exp(tmp_path: Path) -> None:
    p = _mt_cfg(tmp_path, "[model.train]\nloss = \"qlike_level\"\n")
    mcfg = load_model_train_config(p)
    loss_fn = build_training_loss(mcfg)
    pred_np = np.array([-0.2, 0.0])
    tgt_np = np.array([-0.1, 0.05])
    pred = torch.tensor(pred_np, requires_grad=True)
    target = torch.tensor(tgt_np)
    lt = loss_fn(pred, target)
    ref = qlike_level(np.exp(tgt_np), np.exp(pred_np), eps=mcfg.loss_eps)
    assert abs(float(lt.item()) - ref) < 1e-5


def test_hybrid_mse_qlike_is_weighted_sum(tmp_path: Path) -> None:
    p = _mt_cfg(
        tmp_path,
        "[model.train]\nloss = \"hybrid_mse_qlike\"\nloss_alpha = 0.25\nloss_eps = 1e-12\n",
    )
    mcfg = load_model_train_config(p)
    loss_fn = build_training_loss(mcfg)
    assert isinstance(loss_fn, HybridMseQlikeLoss)
    pred = torch.tensor([-1.0, 0.0, 0.5], requires_grad=True)
    target = torch.tensor([-0.5, 0.1, 0.2])
    lt = loss_fn(pred, target)
    mse = torch.nn.functional.mse_loss(pred, target)
    q = QLikeLevelLoss(
        loss_eps=mcfg.loss_eps,
        qlike_pred_transform=mcfg.qlike_pred_transform,
        qlike_level_floor=mcfg.loss_eps,
    )(pred, target)
    exp = 0.25 * mse + 0.75 * q
    assert torch.allclose(lt, exp)


def test_smooth_l1_matches_numpy(tmp_path: Path) -> None:
    p = _mt_cfg(
        tmp_path,
        "[model.train]\nloss = \"smooth_l1_log\"\nsmooth_l1_beta = 0.5\n",
    )
    mcfg = load_model_train_config(p)
    loss_fn = build_training_loss(mcfg)
    pred = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.2, 0.8])
    lt = loss_fn(pred, target)
    np_val = aggregate_test_metric_value(
        "smooth_l1_log",
        target.numpy(),
        pred.detach().numpy(),
        smooth_l1_beta=0.5,
    )
    assert abs(float(lt.item()) - np_val) < 1e-6
