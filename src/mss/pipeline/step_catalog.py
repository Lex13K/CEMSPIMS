"""Human-readable titles and descriptions for pipeline steps (app graph UI)."""

from __future__ import annotations

from dataclasses import dataclass

from mss.pipeline.order import CANONICAL_PIPELINE_ORDER


@dataclass(frozen=True)
class StepInfo:
    pipeline: str
    step_id: str
    title: str
    description: str


_CATALOG: dict[tuple[str, str], tuple[str, str]] = {
    ("data.prepare", "ingest"): (
        "Ingest raw CSVs",
        "Convert WRDS, VIX, and S&P CSV inputs into shared interim Parquet shards.",
    ),
    ("data.prepare", "returns_panel"): (
        "Build returns panel",
        "Assemble the long-format equity returns panel used for graphs and features.",
    ),
    ("data.prepare", "targets"): (
        "Build targets",
        "Compute forward 30-day realized vol targets and benchmark lags in shared targets.parquet.",
    ),
    ("graph.prepare", "feature_dates"): (
        "Feature dates",
        "Determine trading dates on which graphs are built for this run.",
    ),
    ("graph.prepare", "universe"): (
        "Universe selection",
        "Select top-N firms by market cap (fixed replace or rebalance mode).",
    ),
    ("graph.prepare", "node_features"): (
        "Node features",
        "Compute per-node feature vectors on each graph date.",
    ),
    ("graph.prepare", "edges"): (
        "Edges",
        "Build correlation-based edges and top-k neighbor lists per date.",
    ),
    ("dataset.package", "splits"): (
        "Train/val/test splits",
        "Assign chronological splits for model training and evaluation.",
    ),
    ("dataset.package", "scaler"): (
        "Train-only scaler",
        "Fit feature scaler on training dates only.",
    ),
    ("dataset.package", "manifest"): (
        "Dataset manifest",
        "Write dataset package manifest and config fingerprint.",
    ),
    ("model.cache_graphs", "materialize"): (
        "Cache PyG graphs",
        "Materialize PyTorch Geometric graph objects for training.",
    ),
    ("model.train", "train"): (
        "Train GNN",
        "Train GraphSAGE/GAT with early stopping on validation loss.",
    ),
    ("model.evaluate", "score_splits"): (
        "Score splits",
        "Run inference on train/val/test and write forecast parquets.",
    ),
    ("model.evaluate", "write_forecast_panel"): (
        "Forecast panel",
        "Merge model and benchmark forecasts into forecast_panel.parquet.",
    ),
    ("model.evaluate", "aggregate_test_loss"): (
        "Test loss",
        "Aggregate scalar test loss and error series.",
    ),
    ("model.evaluate", "write_summary_table"): (
        "Summary table",
        "Write held-out performance metrics for GNN, VIX, HAR, and placebo.",
    ),
    ("model.evaluate", "run_hypothesis_tests"): (
        "Hypothesis tests",
        "Run formal H1–H5 inference tests (MZ, DM, incremental).",
    ),
    ("analysis.summarize", "loss_figure"): (
        "Diagnostic figures",
        "Generate loss curves, forecast plots, and summary bar charts.",
    ),
}

_RAW_INFO = StepInfo(
    pipeline="data.prepare",
    step_id="raw",
    title="Raw inputs",
    description="WRDS, VIX, and S&P 500 CSV files under data/shared/raw/.",
)

_TERMINAL_INFO = StepInfo(
    pipeline="results",
    step_id="terminal",
    title="Run results",
    description="Processed metrics, figures, and exports for this run.",
)


def step_key(pipeline: str, step_id: str) -> str:
    return f"{pipeline}/{step_id}"


def get_step_info(pipeline: str, step_id: str) -> StepInfo:
    if step_id == "raw":
        return _RAW_INFO
    if step_id == "terminal":
        return _TERMINAL_INFO
    title, desc = _CATALOG.get(
        (pipeline, step_id),
        (f"{pipeline}: {step_id}", "Pipeline step."),
    )
    return StepInfo(pipeline=pipeline, step_id=step_id, title=title, description=desc)


def iter_catalog_steps() -> list[tuple[str, str]]:
    """(pipeline, step_id) in canonical order, excluding thesis.export and compare_runs."""
    from mss.pipeline.orchestrator import PIPELINES

    skip = {"thesis.export", "analysis.compare_runs"}
    out: list[tuple[str, str]] = []
    for pipeline in CANONICAL_PIPELINE_ORDER:
        if pipeline in skip:
            continue
        spec = PIPELINES.get(pipeline)
        if spec is None:
            continue
        for step_id, _ in spec.steps:
            out.append((pipeline, step_id))
    return out
