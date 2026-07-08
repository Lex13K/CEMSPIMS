"""Config validation warnings (Phase 2)."""

from __future__ import annotations

import sys
import tomllib
from datetime import datetime
from pathlib import Path

from mss.dataset.config import load_dataset_config
from mss.evaluation.config import load_model_evaluate_config
from mss.graph.config import load_graph_config
from mss.model_train.config import load_model_train_config


def _parse_date(value: str) -> datetime | None:
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _has_rebalance_freq(config_path: Path) -> bool:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    universe = (data.get("graph") or {}).get("universe") or {}
    return "rebalance_freq" in universe


def _universe_consistency_warnings(config_path: Path) -> list[str]:
    warnings: list[str] = []
    try:
        gc = load_graph_config(config_path)
    except (OSError, ValueError) as e:
        warnings.append(f"could not load graph config for universe checks: {e}")
        return warnings
    if gc.universe_mode == "fixed_replace" and gc.rebalance_freq != "daily":
        warnings.append(
            "[graph.universe] fixed_replace uses a sticky cohort (attrition replacement only); "
            f"rebalance_freq={gc.rebalance_freq!r} applies only when universe_mode='monthly_rebalance'"
        )
    if gc.universe_mode == "monthly_rebalance" and gc.rebalance_freq == "daily":
        warnings.append(
            "monthly_rebalance with rebalance_freq='daily' re-ranks the full cross-section every "
            "feature date (very high turnover)"
        )
    return warnings


def _graph_edges_warnings(config_path: Path) -> list[str]:
    warnings: list[str] = []
    try:
        gc = load_graph_config(config_path)
    except (OSError, ValueError):
        return warnings
    if sys.platform == "win32" and gc.edges_n_jobs > 1:
        warnings.append(
            f"[graph.edges].n_jobs={gc.edges_n_jobs} on Windows may be IO-bound; "
            "consider n_jobs=1 if edge builds are slow"
        )
    return warnings


def _v2_target_feature_warnings(config_path: Path) -> list[str]:
    """v2 baseline consistency: target, node features vs feature_columns, universe robustness modes."""
    warnings: list[str] = []
    try:
        gc = load_graph_config(config_path)
        dc = load_dataset_config(config_path)
    except (OSError, ValueError):
        return warnings

    if dc.target_column != "log_rv_fwd_30cal":
        warnings.append(
            f"[graph.output].target_column={dc.target_column!r} is not the v2 default "
            "'log_rv_fwd_30cal' (forward 30-calendar-day realized vol); the trading-day target is "
            "retained for robustness only"
        )

    produced = set(gc.node_feature_column_names())
    requested = set(dc.feature_columns)
    missing = sorted(requested - produced)
    unused = sorted(produced - requested)
    if missing:
        warnings.append(
            f"[dataset].feature_columns request columns not produced by [graph.node_features]: {missing} "
            "(enable the matching include_* flags or the scaler/step will fail)"
        )
    if unused:
        warnings.append(
            f"[graph.node_features] produces columns not used in [dataset].feature_columns: {unused} "
            "(they will be built but not fed to the model)"
        )

    if gc.selection_rule == "dollar_volume":
        warnings.append(
            "[graph.universe].selection_rule='dollar_volume' is a robustness/alternative ranking; "
            "the canonical v2 universe is broad point-in-time market cap ('mcap')"
        )
    if gc.restrict_to_sp500:
        warnings.append(
            "[graph.universe].restrict_to_sp500=true is a robustness mode (S&P-500 membership filter); "
            "the canonical v2 universe is the broad point-in-time mcap cross-section"
        )
        if not gc.sp500_membership_path:
            warnings.append(
                "restrict_to_sp500=true but sp500_membership_path is empty; the universe step expects a "
                "normalized membership spans parquet at data/shared/interim/sp500_membership.parquet"
            )
    return warnings


def _placebo_mismatch_warnings(config_path: Path, *, configs_dir: Path) -> list[str]:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    me = (data.get("model") or {}).get("evaluate") or {}
    this_placebo = bool(me.get("enable_placebo_ablation", True))
    warnings: list[str] = []
    for sibling in sorted(configs_dir.glob("*.toml")):
        if sibling.resolve() == config_path.resolve():
            continue
        try:
            other = tomllib.loads(sibling.read_text(encoding="utf-8"))
        except OSError:
            continue
        other_me = (other.get("model") or {}).get("evaluate") or {}
        other_placebo = bool(other_me.get("enable_placebo_ablation", True))
        if other_placebo != this_placebo:
            warnings.append(
                f"placebo ablation differs from sibling config {sibling.name}: "
                f"this run enable_placebo_ablation={this_placebo}, "
                f"{sibling.name} has {other_placebo}"
            )
    return warnings


def collect_config_warnings(
    config_path: Path,
    *,
    configs_dir: Path | None = None,
) -> list[str]:
    """Return non-fatal warnings for known config footguns."""
    warnings: list[str] = []
    config_path = config_path.resolve()
    cdir = (configs_dir or config_path.parent).resolve()

    warnings.extend(_universe_consistency_warnings(config_path))
    warnings.extend(_graph_edges_warnings(config_path))
    warnings.extend(_v2_target_feature_warnings(config_path))

    try:
        mtrain = load_model_train_config(config_path)
        meval = load_model_evaluate_config(config_path)
    except (OSError, ValueError) as e:
        warnings.append(f"could not load model config for warning checks: {e}")
        return warnings

    if not mtrain.fit_log_calibration and meval.apply_log_calibration:
        warnings.append(
            "[model.train].fit_log_calibration is false but "
            "[model.evaluate].apply_log_calibration is true — "
            "eval may apply calibration that training did not fit"
        )

    if meval.apply_log_calibration_placebo and not mtrain.fit_log_calibration:
        warnings.append(
            "[model.evaluate].apply_log_calibration_placebo is true but "
            "[model.train].fit_log_calibration is false — "
            "placebo calibration requires a fitted log_calibration.json"
        )

    if meval.placebo_retrain:
        warnings.append(
            "[model.evaluate].placebo_retrain=true trains a second model on placebo graphs for the "
            "H5 arm; this roughly doubles training cost (intended for GPU/scale runs)"
        )

    if sys.platform == "win32" and mtrain.dataloader_num_workers > 0:
        warnings.append(
            f"[model.train].dataloader_num_workers={mtrain.dataloader_num_workers} on Windows; "
            "consider 0 to avoid multiprocessing issues"
        )

    try:
        dc = load_dataset_config(config_path)
    except OSError:
        dc = None

    if dc is not None:
        test_end = _parse_date(dc.test_end)
        stress_start = _parse_date(meval.summary_test_stress_excl_start)
        stress_end = _parse_date(meval.summary_test_stress_excl_end)
        if test_end is not None and stress_start is not None and stress_start > test_end:
            warnings.append(
                "summary_test_stress_excl_start is after [dataset].test_end — "
                "stress subsample may be empty"
            )
        if test_end is not None and stress_end is not None and stress_end > test_end:
            warnings.append(
                "summary_test_stress_excl_end is after [dataset].test_end — "
                "stress window extends beyond test holdout"
            )

    warnings.extend(_placebo_mismatch_warnings(config_path, configs_dir=cdir))
    return warnings
