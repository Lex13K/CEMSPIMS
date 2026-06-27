"""Loss id validation (no torch)."""

from __future__ import annotations

import pytest

from mss.model_train.loss_ids import validate_training_loss_id


def test_validate_training_loss_id() -> None:
    assert validate_training_loss_id("MSE_LOG") == "mse_log"


def test_validate_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        validate_training_loss_id("nope")
