"""Sanity check: thesis.export expected path set is stable and under tables/thesis/."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.processed.paths import tables_archive_dir, thesis_root
from mss.thesis.export import expected_paths_for_thesis_export


def test_expected_paths_are_under_tables_thesis() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    paths = expected_paths_for_thesis_export(cfg)
    assert len(paths) >= 20
    tr = thesis_root(Path(cfg.processed_dir))
    for p in paths:
        rel = p.resolve().relative_to(tr.parent.resolve())
        parts = rel.as_posix().split("/")
        assert parts[0] in ("thesis", "archive")


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
    assert "OLD_A1_full_hypothesis_output.csv" not in names
    assert "Table_C02_full_test_formal_hypothesis_results.csv" in names
    assert "Table_C03_restricted_subsample_robustness_results.csv" in names
    assert "Table_A01_data_inputs_and_roles.tex" not in names


def test_legacy_archive_paths_optional() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    archive = tables_archive_dir(Path(cfg.processed_dir))
    assert all(archive not in p.parents for p in expected_paths_for_thesis_export(cfg))
