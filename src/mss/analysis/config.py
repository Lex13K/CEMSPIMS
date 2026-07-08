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


def load_analysis_figures_config(config_path: Path) -> AnalysisFiguresConfig:
    sm = load_analysis_summarize_config(config_path)
    return AnalysisFiguresConfig(dpi=sm.dpi, verbose=sm.verbose)
