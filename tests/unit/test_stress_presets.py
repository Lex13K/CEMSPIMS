"""H4 stress preset resolution (#20)."""

from __future__ import annotations

import pandas as pd

from mss.evaluation.config import ModelEvaluateConfig
from mss.evaluation.hypothesis_tests import build_formal_subsample_specs
from mss.evaluation.stress_presets import resolve_stress_window, preset_bounds


def test_preset_bounds_rate_shock() -> None:
    assert preset_bounds("rate_shock_2022") == ("2022-01-01", "2022-10-31")


def test_preset_bounds_covid() -> None:
    assert preset_bounds("covid_2020") == ("2020-02-01", "2020-05-31")


def test_explicit_override_wins() -> None:
    start, end, pid = resolve_stress_window(
        preset_id="rate_shock_2022",
        explicit_start="2021-06-01",
        explicit_end="2021-12-31",
    )
    assert (start, end) == ("2021-06-01", "2021-12-31")
    assert pid == ""


def test_preset_switch_changes_stress_subsample() -> None:
    dates = pd.date_range("2022-01-03", periods=20, freq="B")
    panel = pd.DataFrame(
        {
            "date": dates,
            "split": ["test"] * len(dates),
        }
    )
    ecfg_covid = ModelEvaluateConfig(
        splits=("test",),
        batch_size=32,
        device="auto",
        verbose=False,
        summary_test_exclusion_years=(2020,),
        summary_train_crisis_excl_start="2008-09-01",
        summary_train_crisis_excl_end="2009-03-31",
        test_metric=None,
        hypothesis_hac_max_lags=5,
        hypothesis_dm_losses=("qlike",),
        summary_test_stress_excl_start="2020-02-01",
        summary_test_stress_excl_end="2020-05-31",
        hypothesis_h4_include_h1=False,
        apply_log_calibration=False,
        apply_log_calibration_placebo=False,
        hypothesis_stress_preset="covid_2020",
        stress_preset_active="covid_2020",
    )
    ecfg_rate = ModelEvaluateConfig(
        **{**ecfg_covid.__dict__, "summary_test_stress_excl_start": "2022-01-01", "summary_test_stress_excl_end": "2022-10-31", "hypothesis_stress_preset": "rate_shock_2022", "stress_preset_active": "rate_shock_2022"}
    )
    n_covid = build_formal_subsample_specs(panel, ecfg_covid)["test_excl_stress"].n_obs
    n_rate = build_formal_subsample_specs(panel, ecfg_rate)["test_excl_stress"].n_obs
    assert n_covid > n_rate
