"""Model train TOML fingerprint and loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.model_train.config import (
    current_model_train_fingerprint,
    fingerprint_model_train_table,
    load_model_train_config,
)


def test_fingerprint_stable_under_key_order() -> None:
    d1 = {"lr": 0.1, "max_epochs": 1}
    d2 = {"max_epochs": 1, "lr": 0.1}
    assert fingerprint_model_train_table(d1) == fingerprint_model_train_table(d2)


def test_fingerprint_differs_when_lr_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[model.train]\nlr = 0.01\n", encoding="utf-8")
    b.write_text("[model.train]\nlr = 0.02\n", encoding="utf-8")
    assert current_model_train_fingerprint(a) != current_model_train_fingerprint(b)


def test_load_defaults_and_dataloader_workers(tmp_path: Path) -> None:
    p = tmp_path / "empty.toml"
    p.write_text("# no [model.train]\n", encoding="utf-8")
    cfg = load_model_train_config(p)
    assert cfg.seed == 42
    assert cfg.parquet_date_pushdown is True
    assert cfg.dataloader_num_workers == 0
    assert cfg.model_type == "graphsage"
    assert cfg.graph_cache_dir is None
    assert cfg.loss == "mse_log"
    assert cfg.loss_eps == 1e-12
    assert cfg.smooth_l1_beta == 1.0
    assert cfg.loss_alpha == 0.5
    assert cfg.qlike_pred_transform == "exp"
    assert cfg.qlike_level_floor is None
    assert cfg.warm_start_checkpoint is None
    assert cfg.fit_log_calibration is False


def test_fingerprint_differs_when_loss_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[model.train]\nloss = \"mse_log\"\n", encoding="utf-8")
    b.write_text("[model.train]\nloss = \"mae_log\"\n", encoding="utf-8")
    assert current_model_train_fingerprint(a) != current_model_train_fingerprint(b)


def test_hybrid_requires_alpha_in_open_unit_interval(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text(
        "[model.train]\nloss = \"hybrid_mse_qlike\"\nloss_alpha = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_alpha"):
        load_model_train_config(p)


def test_load_invalid_loss_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[model.train]\nloss = \"unknown\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_model_train_config(p)


def test_fingerprint_differs_when_seed_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[model.train]\nseed = 1\n", encoding="utf-8")
    b.write_text("[model.train]\nseed = 2\n", encoding="utf-8")
    assert current_model_train_fingerprint(a) != current_model_train_fingerprint(b)
