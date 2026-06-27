"""Analysis TOML: minimal `[analysis.summarize]` options for loss figure."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisSummarizeConfig:
    """Minimal settings for summary loss figure."""

    dpi: int
    verbose: bool
    draw_hypothesis_table: bool


@dataclass(frozen=True)
class AnalysisReportConfig:
    """Settings for standalone `report.py` helpers (not used by the orchestrator)."""

    bootstrap_reps: int
    include_h1: bool
    verbose: bool


@dataclass(frozen=True)
class AnalysisFiguresConfig:
    """Settings for standalone figure helpers outside `analysis.summarize` (if used)."""

    dpi: int
    verbose: bool


def load_analysis_summarize_config(config_path: Path) -> AnalysisSummarizeConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    sm = dict(data.get("analysis", {}).get("summarize", {}) or {})
    return AnalysisSummarizeConfig(
        dpi=int(sm.get("dpi", 150)),
        verbose=bool(sm.get("verbose", True)),
        draw_hypothesis_table=bool(sm.get("draw_hypothesis_table", True)),
    )


def load_analysis_report_config(config_path: Path) -> AnalysisReportConfig:
    sm = load_analysis_summarize_config(config_path)
    return AnalysisReportConfig(
        bootstrap_reps=499,
        include_h1=False,
        verbose=sm.verbose,
    )


def load_analysis_figures_config(config_path: Path) -> AnalysisFiguresConfig:
    sm = load_analysis_summarize_config(config_path)
    return AnalysisFiguresConfig(dpi=sm.dpi, verbose=sm.verbose)

