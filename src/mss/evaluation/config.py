"""`[model.evaluate]` TOML configuration (minimal pipeline)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_DM_LOSSES = frozenset({"qlike", "mse_log"})
_ALLOWED_PLACEBO_EDGE_MODES = frozenset({"permute_only", "zero_attr"})
# v2 benchmark suite. raw_vix is the primary market-implied benchmark; the others are secondary.
_ALLOWED_BENCHMARKS = frozenset({"raw_vix", "calibrated_vix", "vix_har", "har"})
_DEFAULT_SECONDARY_BENCHMARKS = ("calibrated_vix", "vix_har", "har")


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
    hypothesis_stress_preset: str
    stress_preset_active: str  # empty when explicit start/end override preset
    hypothesis_h4_include_h1: bool
    apply_log_calibration: bool  # affine y_pred <- a + b * y_pred if log_calibration.json exists
    apply_log_calibration_placebo: bool  # apply same affine map to placebo preds when true
    enable_placebo_ablation: bool = True
    placebo_edge_seed: int = 7
    placebo_edge_mode: str = "zero_attr"  # permute_only | zero_attr
    # When True, train a second model on placebo (zeroed/permuted) graphs and score it as the placebo
    # benchmark for H5, instead of running the real model on placebo graphs. Roughly doubles train cost.
    placebo_retrain: bool = False
    fit_vix_log_calibration: bool = False
    hypothesis_dm_losses_vs_placebo: tuple[str, ...] = ("qlike", "mse_log")
    # Secondary benchmarks (beyond the canonical raw_vix H2/H3) for generalized DM and
    # incremental_vs_benchmark inference. raw_vix stays the primary H2/H3 comparison.
    hypothesis_dm_benchmarks: tuple[str, ...] = _DEFAULT_SECONDARY_BENCHMARKS
    hypothesis_incremental_benchmarks: tuple[str, ...] = _DEFAULT_SECONDARY_BENCHMARKS


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


def _benchmarks_tuple(raw: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a benchmark name list; raw_vix is canonical (H2/H3) and excluded from the secondary set."""
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for x in raw:
            s = str(x).strip().lower()
            if s in _ALLOWED_BENCHMARKS and s != "raw_vix" and s not in out:
                out.append(s)
        return tuple(out)
    return default


def load_model_evaluate_config(config_path: Path) -> ModelEvaluateConfig:
    from mss.evaluation.stress_presets import DEFAULT_STRESS_PRESET_ID, resolve_stress_window
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

    preset_id = str(me.get("hypothesis_stress_preset", DEFAULT_STRESS_PRESET_ID)).strip()
    has_explicit_stress = (
        "summary_test_stress_excl_start" in me and "summary_test_stress_excl_end" in me
    )
    stress_start, stress_end, active_preset = resolve_stress_window(
        preset_id=preset_id,
        explicit_start=str(me["summary_test_stress_excl_start"]) if has_explicit_stress else None,
        explicit_end=str(me["summary_test_stress_excl_end"]) if has_explicit_stress else None,
    )

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
        summary_test_stress_excl_start=stress_start,
        summary_test_stress_excl_end=stress_end,
        hypothesis_stress_preset=preset_id,
        stress_preset_active=active_preset,
        hypothesis_h4_include_h1=bool(me.get("hypothesis_h4_include_h1", False)),
        apply_log_calibration=bool(me.get("apply_log_calibration", False)),
        apply_log_calibration_placebo=bool(me.get("apply_log_calibration_placebo", False)),
        enable_placebo_ablation=bool(me.get("enable_placebo_ablation", True)),
        placebo_edge_seed=int(me.get("placebo_edge_seed", 7)),
        placebo_edge_mode=_validate_placebo_edge_mode(me.get("placebo_edge_mode", "zero_attr")),
        placebo_retrain=bool(me.get("placebo_retrain", False)),
        fit_vix_log_calibration=bool(me.get("fit_vix_log_calibration", False)),
        hypothesis_dm_losses_vs_placebo=_dm_losses_tuple(
            me.get("hypothesis_dm_losses_vs_placebo"), default=("qlike", "mse_log")
        ),
        hypothesis_dm_benchmarks=_benchmarks_tuple(
            me.get("hypothesis_dm_benchmarks"), default=_DEFAULT_SECONDARY_BENCHMARKS
        ),
        hypothesis_incremental_benchmarks=_benchmarks_tuple(
            me.get("hypothesis_incremental_benchmarks"), default=_DEFAULT_SECONDARY_BENCHMARKS
        ),
    )


def _validate_placebo_edge_mode(raw: object) -> str:
    mode = str(raw).strip().lower() if raw is not None else "zero_attr"
    if mode not in _ALLOWED_PLACEBO_EDGE_MODES:
        raise ValueError(
            f"Unknown [model.evaluate].placebo_edge_mode: {raw!r}; "
            f"expected one of {sorted(_ALLOWED_PLACEBO_EDGE_MODES)}"
        )
    return mode

