# Pipeline step catalog

from mss.pipeline.step_catalog import get_step_info, iter_catalog_steps


def test_catalog_has_ingest() -> None:
    info = get_step_info("data.prepare", "ingest")
    assert "Ingest" in info.title


def test_iter_catalog_excludes_compare() -> None:
    pipelines = {p for p, _ in iter_catalog_steps()}
    assert "analysis.compare_runs" not in pipelines
    assert "thesis.export" not in pipelines
