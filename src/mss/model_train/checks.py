"""Training output completeness and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mss.io.config import ResolvedConfig


CHECKPOINT_FILENAME = "checkpoint.pt"
TRAINING_STATE_FILENAME = "training_state.json"
FINAL_METRICS_FILENAME = "final_metrics.json"


def model_train_dir(cfg: ResolvedConfig) -> Path:
    return cfg.run_interim_dir / "model_train"


def final_metrics_path(cfg: ResolvedConfig) -> Path:
    return model_train_dir(cfg) / FINAL_METRICS_FILENAME


def checkpoint_path(cfg: ResolvedConfig) -> Path:
    return model_train_dir(cfg) / CHECKPOINT_FILENAME


def training_state_path(cfg: ResolvedConfig) -> Path:
    return model_train_dir(cfg) / TRAINING_STATE_FILENAME


def check_final_metrics_complete(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"passed": False, "issues": []}
    if not path.is_file():
        out["issues"].append(f"missing {path.name}")
        return out
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        out["issues"].append(f"invalid JSON: {e}")
        return out
    if data.get("status") != "completed":
        out["issues"].append("status is not 'completed'")
    for k in ("best_epoch", "best_val_loss"):
        if k not in data:
            out["issues"].append(f"missing field: {k}")
    out["passed"] = len(out["issues"]) == 0
    return out


def model_train_step_semantic_only(cfg: ResolvedConfig) -> bool:
    chk = check_final_metrics_complete(final_metrics_path(cfg))
    return bool(chk.get("passed"))


def model_train_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    if not model_train_step_semantic_only(cfg):
        return False
    from mss.config.fingerprints import (
        dataset_fingerprint_matches,
        graph_fingerprint_matches,
        train_fingerprint_matches,
    )

    return (
        train_fingerprint_matches(cfg)
        and graph_fingerprint_matches(cfg)
        and dataset_fingerprint_matches(cfg)
    )


def can_resume_training(cfg: ResolvedConfig) -> bool:
    """True if checkpoint + training_state exist and final_metrics is not completed."""
    d = model_train_dir(cfg)
    ck = d / CHECKPOINT_FILENAME
    st = d / TRAINING_STATE_FILENAME
    fm = d / FINAL_METRICS_FILENAME
    if not ck.is_file() or not st.is_file():
        return False
    if not fm.is_file():
        return True
    chk = check_final_metrics_complete(fm)
    return not chk.get("passed")
