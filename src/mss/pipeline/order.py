"""Canonical pipeline ordering (shared by orchestrator, branch, CLI)."""

CANONICAL_PIPELINE_ORDER: tuple[str, ...] = (
    "data.prepare",
    "graph.prepare",
    "dataset.package",
    "model.cache_graphs",
    "model.train",
    "model.evaluate",
    "analysis.summarize",
    "thesis.export",
    "analysis.compare_runs",
)
