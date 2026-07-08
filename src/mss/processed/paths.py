"""Canonical paths under data/<run_id>/processed/ (Phase 6 layout)."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import ResolvedConfig

# --- Filenames ---
FORECASTS_FILENAME = "forecasts.parquet"
FORECAST_PANEL_FILENAME = "forecast_panel.parquet"
TEST_LOSS_FILENAME = "test_loss.json"
SUMMARY_TABLE_FILENAME = "summary_table.csv"
HYPOTHESIS_TESTS_FILENAME = "hypothesis_tests.csv"
REGRESSION_MZ_GNN_FILENAME = "regression_mz_gnn.csv"
REGRESSION_INCREMENTAL_FILENAME = "regression_incremental.csv"
DIAGNOSTICS_SMOOTHING_FILENAME = "diagnostics_smoothing.csv"

FORECAST_VS_BENCHMARKS = "forecast_vs_benchmarks.png"
LOSS_FIG_LINEAR = "loss_over_epochs_linear.png"
LOSS_FIG_LOG = "loss_over_epochs_log.png"
SUMMARY_BARPLOT = "summary_barplot.png"
HYPOTHESIS_TABLE_PNG = "hypothesis_tests_table.png"

UNIVERSE_TURNOVER_TIMESERIES_PNG = "universe_turnover_timeseries.png"
UNIVERSE_RANKBUCKET_REPLACEMENT_HEATMAP_PNG = "universe_rankbucket_replacement_heatmap.png"
UNIVERSE_TENURE_DISTRIBUTION_PNG = "universe_tenure_distribution.png"

UNIVERSE_TURNOVER_TIMESERIES_CSV = "universe_turnover_timeseries.csv"
UNIVERSE_RANKBUCKET_REPLACEMENT_LONG_CSV = "universe_rankbucket_replacement_long.csv"
UNIVERSE_TENURE_DISTRIBUTION_CSV = "universe_tenure_distribution.csv"
UNIVERSE_CHURN_SUMMARY_CSV = "universe_churn_summary.csv"


def processed_root(cfg: ResolvedConfig | Path) -> Path:
    if isinstance(cfg, Path):
        return cfg
    return Path(cfg.processed_dir)


def scoring_dir(processed_dir: Path) -> Path:
    return processed_dir / "scoring"


def metrics_dir(processed_dir: Path) -> Path:
    return processed_dir / "metrics"


def metrics_descriptive_dir(processed_dir: Path) -> Path:
    return metrics_dir(processed_dir) / "descriptive"


def metrics_formal_dir(processed_dir: Path) -> Path:
    return metrics_dir(processed_dir) / "formal"


def metrics_universe_dir(processed_dir: Path) -> Path:
    return metrics_dir(processed_dir) / "universe"


def figures_dir(processed_dir: Path) -> Path:
    return processed_dir / "figures"


def figures_by_split_dir(processed_dir: Path, split: str) -> Path:
    return figures_dir(processed_dir) / "by_split" / split


def figures_diagnostics_dir(processed_dir: Path) -> Path:
    return figures_dir(processed_dir) / "diagnostics"


def figures_universe_churn_dir(processed_dir: Path) -> Path:
    return figures_diagnostics_dir(processed_dir) / "universe_churn"


def tables_dir(processed_dir: Path) -> Path:
    return processed_dir / "tables"


def tables_thesis_dir(processed_dir: Path) -> Path:
    return tables_dir(processed_dir) / "thesis"


def tables_archive_dir(processed_dir: Path) -> Path:
    return tables_dir(processed_dir) / "archive"


def comparisons_dir(processed_dir: Path) -> Path:
    return processed_dir / "comparisons"


def comparison_root(processed_dir: Path, comparison_id: str) -> Path:
    return comparisons_dir(processed_dir) / comparison_id


def comparison_tables_dir(processed_dir: Path, comparison_id: str) -> Path:
    return comparison_root(processed_dir, comparison_id) / "tables"


def comparison_figures_dir(processed_dir: Path, comparison_id: str) -> Path:
    return comparison_root(processed_dir, comparison_id) / "figures"


# --- Scoring (model.evaluate) ---
def forecasts_path(cfg: ResolvedConfig) -> Path:
    return scoring_dir(processed_root(cfg)) / FORECASTS_FILENAME


def forecast_panel_path(cfg: ResolvedConfig) -> Path:
    return scoring_dir(processed_root(cfg)) / FORECAST_PANEL_FILENAME


def test_loss_path(cfg: ResolvedConfig) -> Path:
    return scoring_dir(processed_root(cfg)) / TEST_LOSS_FILENAME


# --- Metrics descriptive ---
def summary_table_path(cfg: ResolvedConfig) -> Path:
    return metrics_descriptive_dir(processed_root(cfg)) / SUMMARY_TABLE_FILENAME


def diagnostics_smoothing_path(cfg: ResolvedConfig) -> Path:
    return metrics_descriptive_dir(processed_root(cfg)) / DIAGNOSTICS_SMOOTHING_FILENAME


# --- Metrics formal ---
def hypothesis_tests_path(cfg: ResolvedConfig) -> Path:
    return metrics_formal_dir(processed_root(cfg)) / HYPOTHESIS_TESTS_FILENAME


def regression_mz_gnn_path(cfg: ResolvedConfig) -> Path:
    return metrics_formal_dir(processed_root(cfg)) / REGRESSION_MZ_GNN_FILENAME


def regression_incremental_path(cfg: ResolvedConfig) -> Path:
    return metrics_formal_dir(processed_root(cfg)) / REGRESSION_INCREMENTAL_FILENAME


# --- Metrics universe (analysis.summarize) ---
def universe_turnover_timeseries_csv_path(processed_dir: Path) -> Path:
    return metrics_universe_dir(processed_dir) / UNIVERSE_TURNOVER_TIMESERIES_CSV


def universe_rankbucket_replacement_long_csv_path(processed_dir: Path) -> Path:
    return metrics_universe_dir(processed_dir) / UNIVERSE_RANKBUCKET_REPLACEMENT_LONG_CSV


def universe_tenure_distribution_csv_path(processed_dir: Path) -> Path:
    return metrics_universe_dir(processed_dir) / UNIVERSE_TENURE_DISTRIBUTION_CSV


def universe_churn_summary_csv_path(processed_dir: Path) -> Path:
    return metrics_universe_dir(processed_dir) / UNIVERSE_CHURN_SUMMARY_CSV


# --- Figures ---
def forecast_comparison_path(processed_dir: Path, split_suffix: str) -> Path:
    return figures_by_split_dir(processed_dir, split_suffix) / FORECAST_VS_BENCHMARKS


def loss_figure_path_linear(processed_dir: Path) -> Path:
    return figures_diagnostics_dir(processed_dir) / LOSS_FIG_LINEAR


def loss_figure_path_log(processed_dir: Path) -> Path:
    return figures_diagnostics_dir(processed_dir) / LOSS_FIG_LOG


def summary_barplot_path(processed_dir: Path) -> Path:
    return figures_diagnostics_dir(processed_dir) / SUMMARY_BARPLOT


def hypothesis_table_figure_path(processed_dir: Path) -> Path:
    return figures_diagnostics_dir(processed_dir) / HYPOTHESIS_TABLE_PNG


def universe_turnover_timeseries_path(processed_dir: Path) -> Path:
    return figures_universe_churn_dir(processed_dir) / UNIVERSE_TURNOVER_TIMESERIES_PNG


def universe_rankbucket_replacement_heatmap_path(processed_dir: Path) -> Path:
    return figures_universe_churn_dir(processed_dir) / UNIVERSE_RANKBUCKET_REPLACEMENT_HEATMAP_PNG


def universe_tenure_distribution_path(processed_dir: Path) -> Path:
    return figures_universe_churn_dir(processed_dir) / UNIVERSE_TENURE_DISTRIBUTION_PNG


def thesis_root(processed_dir: Path) -> Path:
    return tables_thesis_dir(processed_dir)
