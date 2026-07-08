"""Unit tests for validate-config warnings."""

from __future__ import annotations

from pathlib import Path

from mss.config.validate import collect_config_warnings


def _write_default_like(configs_dir: Path, name: str = "default.toml") -> Path:
    p = configs_dir / name
    p.write_text(
        "[graph.universe]\n"
        "universe_mode = \"fixed_replace\"\n"
        "n_nodes = 500\n"
        "rebalance_freq = \"monthly\"\n\n"
        "[dataset]\n"
        "train_end = \"2014-12-31\"\n"
        "val_end = \"2018-12-31\"\n"
        "test_end = \"2024-12-31\"\n\n"
        "[model.train]\n"
        "dataloader_num_workers = 0\n\n"
        "[model.evaluate]\n"
        "summary_test_stress_excl_start = \"2022-01-01\"\n"
        "summary_test_stress_excl_end = \"2022-10-31\"\n"
        "enable_placebo_ablation = true\n"
        "apply_log_calibration = true\n",
        encoding="utf-8",
    )
    return p


def test_rebalance_freq_and_calibration_warnings(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = _write_default_like(configs_dir)
    warnings = collect_config_warnings(cfg, configs_dir=configs_dir)
    text = "\n".join(warnings)
    assert "rebalance_freq" in text
    assert "fit_log_calibration" in text


def test_placebo_sibling_mismatch(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    a = _write_default_like(configs_dir, "a.toml")
    b = configs_dir / "b.toml"
    b.write_text(
        "[model.evaluate]\nenable_placebo_ablation = false\n",
        encoding="utf-8",
    )
    warnings = collect_config_warnings(a, configs_dir=configs_dir)
    assert any("placebo" in w.lower() for w in warnings)


def test_v2_feature_column_mismatch_warning(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    p = configs_dir / "mismatch.toml"
    p.write_text(
        "[graph.output]\ntarget_column = \"log_rv_fwd_30cal\"\n\n"
        "[graph.node_features]\n"
        "rolling_mean = true\nrolling_vol = true\ninclude_skew = true\n\n"
        "[dataset]\nfeature_columns = [\"rolling_mean\", \"rolling_vol\"]\n",
        encoding="utf-8",
    )
    warnings = collect_config_warnings(p, configs_dir=configs_dir)
    text = "\n".join(warnings)
    # skew is produced but not requested -> unused warning.
    assert "skew" in text


def test_v2_robustness_universe_warnings(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    p = configs_dir / "robust.toml"
    p.write_text(
        "[graph.universe]\n"
        "selection_rule = \"dollar_volume\"\n"
        "restrict_to_sp500 = true\n\n"
        "[graph.output]\ntarget_column = \"log_rv_fwd_30cal\"\n\n"
        "[model.evaluate]\nplacebo_retrain = true\n",
        encoding="utf-8",
    )
    warnings = collect_config_warnings(p, configs_dir=configs_dir)
    text = "\n".join(warnings).lower()
    assert "dollar_volume" in text
    assert "restrict_to_sp500" in text
    assert "placebo_retrain" in text


def test_stress_window_beyond_test_end(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    p = configs_dir / "stress.toml"
    p.write_text(
        "[dataset]\ntest_end = \"2020-12-31\"\n\n"
        "[model.evaluate]\n"
        "summary_test_stress_excl_start = \"2022-01-01\"\n"
        "summary_test_stress_excl_end = \"2022-10-31\"\n",
        encoding="utf-8",
    )
    warnings = collect_config_warnings(p, configs_dir=configs_dir)
    assert any("stress" in w.lower() for w in warnings)
