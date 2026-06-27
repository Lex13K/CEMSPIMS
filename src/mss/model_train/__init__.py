"""PyG model training pipeline (`model.train`)."""

from __future__ import annotations

__all__ = ["run_model_train"]


def __getattr__(name: str):
    if name == "run_model_train":
        from mss.model_train.train_loop import run_model_train

        return run_model_train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
