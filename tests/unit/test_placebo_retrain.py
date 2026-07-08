"""Unit tests for optional placebo_retrain arm wiring (no Torch/PyG required)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_resolved_config


def test_run_placebo_retrain_noop_when_disabled(tmp_path: Path) -> None:
    """When placebo_retrain=false, run_placebo_retrain returns before training."""
    pytest.importorskip("torch")
    from mss.model_train.train_loop import run_placebo_retrain

    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("[model.evaluate]\nplacebo_retrain = false\n", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())
    run_placebo_retrain(cfg, overwrite=False)


def test_step_model_train_invokes_placebo_retrain(tmp_path: Path) -> None:
    """Orchestrator calls run_placebo_retrain after run_model_train (mocked train_loop)."""
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text("[model.evaluate]\nplacebo_retrain = true\n", encoding="utf-8")
    cfg = make_resolved_config(tmp_path, source_config_path=cfg_toml.resolve())

    fake_train_loop = MagicMock()
    fake_train_loop.run_model_train = MagicMock()
    fake_train_loop.run_placebo_retrain = MagicMock()

    with patch.dict(sys.modules, {"mss.model_train.train_loop": fake_train_loop}):
        from mss.pipeline.orchestrator import _step_model_train

        _step_model_train(cfg, overwrite=False, atol=1e-6, rtol=1e-5)

    fake_train_loop.run_model_train.assert_called_once_with(cfg, overwrite=False)
    fake_train_loop.run_placebo_retrain.assert_called_once_with(cfg, overwrite=False)
