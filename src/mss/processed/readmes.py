"""Static README templates for processed/ subfolders."""

from __future__ import annotations

from pathlib import Path

_README_ROOT = """# processed/

Pipeline outputs for one run (`data/<run_id>/processed/`).

Phase 6 layout (hard cutover): scoring artifacts, metrics, figures, and thesis tables
live in subfolders below. Re-run `model.evaluate` onward after upgrading from the
legacy flat `summaries/` + root-level parquets layout.
"""

_README_SCORING = """# scoring/

Raw model evaluation outputs from `model.evaluate` score_splits / write_forecast_panel.

- `forecasts.parquet` — per-date predictions (all splits)
- `forecast_panel.parquet` — VIX-merged panel with log/level columns and per-date losses
- `test_loss.json` — scalar test loss (matches training metric unless overridden)
"""

_README_METRICS = """# metrics/

Aggregated metrics CSVs. Descriptive tables include train/val slices; formal inference
reads only `formal/` (test subsamples).
"""

_README_METRICS_FORMAL = """# metrics/formal/

Formal hypothesis tests (H1–H5). Producer: `model.evaluate` / `run_hypothesis_tests`.
Consumer: `thesis.export`, optional `analysis.summarize` hypothesis table figure.
"""

_README_FIGURES = """# figures/

Diagnostic and split-first forecast comparison figures from `analysis.summarize`.
"""

_README_TABLES = """# tables/

Thesis manuscript exports (`thesis/`) and optional archival copies (`archive/` when
`[thesis.export] legacy_archive = true`).
"""

_README_COMPARISONS = """# comparisons/

Cross-run comparison outputs from `analysis.compare_runs` (Phase 7).
Each subfolder `<comparison_id>/` holds Appendix E-style tables and figures
aggregating metrics from multiple completed runs. Writes are anchored under
the comparison anchor run's `processed/` directory.
"""

_TEMPLATES: dict[str, str] = {
    "README.md": _README_ROOT,
    "scoring/README.md": _README_SCORING,
    "metrics/README.md": _README_METRICS,
    "metrics/formal/README.md": _README_METRICS_FORMAL,
    "figures/README.md": _README_FIGURES,
    "tables/README.md": _README_TABLES,
    "comparisons/README.md": _README_COMPARISONS,
}


def ensure_processed_readmes(processed_dir: Path) -> None:
    """Write README templates if missing (idempotent)."""
    root = Path(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in _TEMPLATES.items():
        path = root / rel
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text.strip() + "\n", encoding="utf-8")
