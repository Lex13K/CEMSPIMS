from __future__ import annotations

from pathlib import Path

from mss.evaluation.config import load_model_evaluate_config


def test_load_model_evaluate_config_defaults(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text("", encoding="utf-8")
    cfg = load_model_evaluate_config(p)
    assert cfg.splits == ("test",)
    assert cfg.batch_size == 32
    assert cfg.device == "auto"
    assert cfg.summary_test_exclusion_years == (2020,)
    assert cfg.summary_train_crisis_excl_start == "2008-09-01"
    assert cfg.summary_train_crisis_excl_end == "2009-03-31"
    assert cfg.test_metric is None
    assert cfg.hypothesis_hac_max_lags == 29
    assert cfg.hypothesis_dm_losses == ("qlike", "mse_log")
    assert cfg.hypothesis_h4_include_h1 is False
    assert cfg.summary_test_stress_excl_start == "2022-01-01"
    assert cfg.apply_log_calibration is False
    assert cfg.apply_log_calibration_placebo is False
    assert cfg.placebo_edge_mode == "zero_attr"
    assert cfg.fit_vix_log_calibration is False
    assert cfg.hypothesis_dm_losses_vs_placebo == ("qlike", "mse_log")
    assert cfg.placebo_retrain is False
    assert cfg.hypothesis_dm_benchmarks == ("calibrated_vix", "vix_har", "har")
    assert cfg.hypothesis_incremental_benchmarks == ("calibrated_vix", "vix_har", "har")


def test_load_model_evaluate_config_benchmarks_and_placebo_retrain(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(
        "[model.evaluate]\n"
        "placebo_retrain = true\n"
        'hypothesis_dm_benchmarks = ["raw_vix", "vix_har", "bogus"]\n'
        'hypothesis_incremental_benchmarks = ["har"]\n',
        encoding="utf-8",
    )
    cfg = load_model_evaluate_config(p)
    assert cfg.placebo_retrain is True
    # raw_vix is canonical and dropped from the secondary set; bogus is ignored.
    assert cfg.hypothesis_dm_benchmarks == ("vix_har",)
    assert cfg.hypothesis_incremental_benchmarks == ("har",)


def test_load_model_evaluate_config_explicit_splits(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(
        "[model.evaluate]\n"
        "splits = [\"train\", \"val\", \"test\"]\n"
        "batch_size = 4\n"
        "device = \"cpu\"\n",
        encoding="utf-8",
    )
    cfg = load_model_evaluate_config(p)
    assert cfg.splits == ("train", "val", "test")
    assert cfg.batch_size == 4
    assert cfg.device == "cpu"
    assert cfg.summary_test_exclusion_years == (2020,)
    assert cfg.summary_train_crisis_excl_start == "2008-09-01"
    assert cfg.summary_train_crisis_excl_end == "2009-03-31"
    assert cfg.test_metric is None


def test_load_model_evaluate_config_test_metric(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text('[model.evaluate]\ntest_metric = "qlike_level"\n', encoding="utf-8")
    cfg = load_model_evaluate_config(p)
    assert cfg.test_metric == "qlike_level"

