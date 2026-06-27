"""Training / test metric ids (no torch dependency)."""

from __future__ import annotations

ALLOWED_TRAINING_LOSS_IDS: frozenset[str] = frozenset(
    {"mse_log", "mae_log", "smooth_l1_log", "qlike_level", "hybrid_mse_qlike"}
)


def validate_training_loss_id(name: str) -> str:
    n = name.strip().lower()
    if n not in ALLOWED_TRAINING_LOSS_IDS:
        raise ValueError(
            f"Unknown [model.train].loss: {name!r}; expected one of {sorted(ALLOWED_TRAINING_LOSS_IDS)}"
        )
    return n
