"""Sanity check: thesis.export expected path set is stable and under thesis_exhibits/."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.thesis.export import expected_paths_for_thesis_export, thesis_root


def test_expected_paths_are_under_thesis_exhibits() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    paths = expected_paths_for_thesis_export(cfg)
    assert len(paths) >= 25
    tr = thesis_root(Path(cfg.processed_dir))
    for p in paths:
        assert tr in p.resolve().parents or p.resolve().parent == tr.parent
        # All outputs live inside .../processed/thesis_exhibits/...
        assert "thesis_exhibits" in str(p)


def test_thesis_table_filenames_present() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    names = [p.name for p in expected_paths_for_thesis_export(cfg)]
    assert "Table_A01_data_inputs_and_roles.csv" in names
    assert "Table_A02_wrds_crsp_raw_input_schema.csv" in names
    assert "Table_A03_required_target_and_benchmark_schema.csv" in names
    assert "Table_D03_log_scale_forecast_error_by_target_volatility_quintile_full_test_sample.csv" in names
    assert "Table_D04_log_scale_forecast_error_by_target_volatility_quintile_test_sample_excluding_2020.csv" in names
    assert "Table_D01_calibration_diagnostics_full_test_sample.csv" in names
    assert "Figure_F01_universe_diagnostics_plate.png" in names
    assert "Reproducibility_metadata.json" in names
    assert "thesis_table_index.md" in names
    assert "OLD_A1_full_hypothesis_output.csv" in names
    assert "Table_C02_full_test_formal_hypothesis_results.csv" in names
    assert "Table_C03_restricted_subsample_robustness_results.csv" in names
    assert "Table_A01_data_inputs_and_roles.tex" not in names
