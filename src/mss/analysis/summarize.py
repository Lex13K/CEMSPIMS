"""analysis.summarize pipeline: build a single loss figure."""

from __future__ import annotations

from mss.analysis.figures import (
    run_build_forecast_comparison_figures,
    run_build_hypothesis_table_figure,
    run_build_loss_figure,
    run_build_universe_churn_figures,
    run_build_summary_barplot,
)
from mss.io.config import ResolvedConfig


def run_summarize_loss_figure(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Summarize step: forecast comparison figures, then loss-over-epochs figure."""
    run_build_forecast_comparison_figures(cfg, overwrite=overwrite)
    run_build_loss_figure(cfg, overwrite=overwrite)
    run_build_summary_barplot(cfg, overwrite=overwrite)
    run_build_universe_churn_figures(cfg, overwrite=overwrite)
    run_build_hypothesis_table_figure(cfg, overwrite=overwrite)


def run_analysis_summarize(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Run all summarize steps in order."""
    run_summarize_loss_figure(cfg, overwrite=overwrite)
