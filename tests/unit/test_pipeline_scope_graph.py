"""Pipeline scope on run manifest and experiment graph."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.run.experiment_graph import build_experiment_graph
from mss.run.manifest import RunManifest, save_run_manifest
from mss.run.step_journal import StepJournal, StepJournalEntry, save_step_journal


def test_pipeline_scope_limits_graph_nodes(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    data = tmp_path / "data"
    for rid in ("parent", "child"):
        run_dir = data / rid
        (run_dir / "interim").mkdir(parents=True)
        cfg_path = configs / f"{rid}.toml"
        cfg_path.write_text(
            "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
            "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n"
            "[graph.edges]\ntop_k = 10\n",
            encoding="utf-8",
        )

    child_cfg = load_resolved_config(configs / "child.toml", "child")
    man = RunManifest(
        run_id="child",
        parent_run_id="parent",
        branch_pipeline="graph.prepare",
        branch_step_id="feature_dates",
        created_at="2020-01-01T00:00:00Z",
        config_path="configs/child.toml",
        pipeline_scope=["graph.prepare"],
    )
    save_run_manifest(child_cfg.manifest_path, man)
    save_step_journal(
        child_cfg,
        StepJournal(
            entries={
                "graph.prepare/feature_dates": StepJournalEntry(
                    pipeline="graph.prepare",
                    step_id="feature_dates",
                    status="completed",
                    started_at="2020-01-01T00:00:00Z",
                    completed_at="2020-01-01T00:01:00Z",
                )
            }
        ),
    )

    g = build_experiment_graph(configs)
    child_nodes = [n for n in g["nodes"] if n["id"].startswith("run:child:")]
    pipelines = {n["pipeline"] for n in child_nodes}
    assert pipelines <= {"graph.prepare"}
    assert not any(n["id"] == "terminal:child" for n in g["nodes"])
