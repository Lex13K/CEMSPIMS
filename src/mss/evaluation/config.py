"""`[model.evaluate]` TOML configuration (minimal pipeline)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_DM_LOSSES = frozenset({"qlike", "mse_log"})


@dataclass(frozen=True)
class ModelEvaluateConfig:
    splits: tuple[str, ...]
    batch_size: int
    device: str
    verbose: bool
    summary_test_exclusion_years: tuple[int, ...]
    summary_train_crisis_excl_start: str
    summary_train_crisis_excl_end: str
    test_metric: str | None  # None: use [model.train].loss
    hypothesis_hac_max_lags: int
    hypothesis_dm_losses: tuple[str, ...]
    summary_test_stress_excl_start: str
    summary_test_stress_excl_end: str
    hypothesis_h4_include_h1: bool
    apply_log_calibration: bool  # affine y_pred <- a + b * y_pred if log_calibration.json exists
    enable_placebo_ablation: bool = True
    placebo_edge_seed: int = 7


def _int_tuple(raw: object, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for x in raw:
            out.append(int(x))
        return tuple(out) if out else default
    return default


def _crisis_bound(raw: object, *, default: str) -> str:
    if raw is None:
        return default
    s = str(raw).strip()
    return s if s else default


def _dm_losses_tuple(raw: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for x in raw:
            s = str(x).strip().lower()
            if s in _ALLOWED_DM_LOSSES:
                out.append(s)
        return tuple(out) if out else default
    return default


def load_model_evaluate_config(config_path: Path) -> ModelEvaluateConfig:
    from mss.model_train.loss_ids import validate_training_loss_id

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    me = dict(data.get("model", {}).get("evaluate", {}) or {})
    raw_splits = me.get("splits")
    if raw_splits is None:
        split_single = str(me.get("split", "test")).strip().lower()
        splits = (split_single,) if split_single else ("test",)
    else:
        splits = tuple(str(s).strip().lower() for s in list(raw_splits) if str(s).strip())
        if not splits:
            splits = ("train", "val", "test")

    raw_tm = me.get("test_metric")
    if raw_tm is None or str(raw_tm).strip() == "":
        test_metric: str | None = None
    else:
        test_metric = validate_training_loss_id(str(raw_tm).strip().lower())

    return ModelEvaluateConfig(
        splits=splits,
        batch_size=int(me.get("batch_size", 32)),
        device=str(me.get("device", "auto")).lower(),
        verbose=bool(me.get("verbose", True)),
        summary_test_exclusion_years=_int_tuple(
            me.get("summary_test_exclusion_years"), default=(2020,)
        ),
        summary_train_crisis_excl_start=_crisis_bound(
            me.get("summary_train_crisis_excl_start"), default="2008-09-01"
        ),
        summary_train_crisis_excl_end=_crisis_bound(
            me.get("summary_train_crisis_excl_end"), default="2009-03-31"
        ),
        test_metric=test_metric,
        hypothesis_hac_max_lags=int(me.get("hypothesis_hac_max_lags", 29)),
        hypothesis_dm_losses=_dm_losses_tuple(me.get("hypothesis_dm_losses"), default=("qlike", "mse_log")),
        summary_test_stress_excl_start=_crisis_bound(
            me.get("summary_test_stress_excl_start"), default="2008-09-01"
        ),
        summary_test_stress_excl_end=_crisis_bound(
            me.get("summary_test_stress_excl_end"), default="2009-03-31"
        ),
        hypothesis_h4_include_h1=bool(me.get("hypothesis_h4_include_h1", False)),
        apply_log_calibration=bool(me.get("apply_log_calibration", True)),
        enable_placebo_ablation=bool(me.get("enable_placebo_ablation", True)),
        placebo_edge_seed=int(me.get("placebo_edge_seed", 7)),
    )

