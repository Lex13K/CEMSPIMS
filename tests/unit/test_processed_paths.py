"""Snapshot key relative paths from mss.processed.paths."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.processed import paths as p


def test_scoring_paths_under_processed() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    proc = Path(cfg.processed_dir)
    assert p.forecasts_path(cfg).parent == proc / "scoring"
    assert p.forecast_panel_path(cfg).name == "forecast_panel.parquet"
    assert p.test_loss_path(cfg).name == "test_loss.json"


def test_metrics_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    proc = Path(cfg.processed_dir)
    assert p.summary_table_path(cfg).parent == proc / "metrics" / "descriptive"
    assert p.hypothesis_tests_path(cfg).parent == proc / "metrics" / "formal"
    assert p.universe_churn_summary_csv_path(proc).parent == proc / "metrics" / "universe"


def test_figure_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    proc = Path(cfg.processed_dir)
    assert p.forecast_comparison_path(proc, "test").parent == proc / "figures" / "by_split" / "test"
    assert p.forecast_comparison_path(proc, "test").name == "forecast_vs_benchmarks.png"
    assert p.loss_figure_path_linear(proc).parent == proc / "figures" / "diagnostics"
    assert p.thesis_root(proc) == proc / "tables" / "thesis"
