"""Branch graph layout: fork step and progressive child nodes."""

from __future__ import annotations

import json
from pathlib import Path

from mss.io.config import load_resolved_config
from mss.run.experiment_graph import build_experiment_graph
from mss.run.manifest import RunManifest, save_run_manifest
from mss.run.step_journal import StepJournal, StepJournalEntry, save_step_journal


def _write_pair(tmp_path: Path, configs: Path) -> None:
    for rid in ("parent", "child"):
        run_dir = tmp_path / "data" / rid
        (run_dir / "interim").mkdir(parents=True)
        (configs / f"{rid}.toml").write_text(
            "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
            "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n"
            "[graph.edges]\ntop_k = 10\n",
            encoding="utf-8",
        )


def test_branch_from_edges_forks_at_node_features(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_pair(tmp_path, configs)

    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)

    child_cfg = load_resolved_config(configs / "child.toml", "child")
    save_run_manifest(
        child_cfg.manifest_path,
        RunManifest(
            run_id="child",
            parent_run_id="parent",
            branch_pipeline="graph.prepare",
            branch_step_id="edges",
            created_at="2020-01-01T00:00:00Z",
            config_path="configs/child.toml",
            pipeline_scope=["graph.prepare"],
        ),
    )
    journal = StepJournal(
        entries={
            "graph.prepare/edges": StepJournalEntry(
                pipeline="graph.prepare",
                step_id="edges",
                status="running",
                started_at="2020-01-01T00:00:00Z",
            )
        }
    )
    save_step_journal(child_cfg, journal)

    g = build_experiment_graph(configs)
    branch_edges = [e for e in g["edges"] if e["kind"] == "branch"]
    assert len(branch_edges) == 1
    assert branch_edges[0]["source"] == "run:parent:graph.prepare:node_features"
    assert branch_edges[0]["target"] == "run:child:graph.prepare:edges"

    child_ids = {n["id"] for n in g["nodes"] if n["run_id"] == "child" or "run:child:" in n["id"]}
    assert "run:child:graph.prepare:edges" in child_ids
    assert "run:child:graph.prepare:feature_dates" not in child_ids

    child_edges = nodes_pos(g, "run:child:graph.prepare:edges")
    parent_edges = nodes_pos(g, "run:parent:graph.prepare:edges")
    parent_nf = nodes_pos(g, "run:parent:graph.prepare:node_features")
    assert child_edges["x"] > parent_nf["x"]
    assert abs(child_edges["y"] - parent_edges["y"]) < 1


def nodes_pos(g: dict, nid: str) -> dict:
    n = next(x for x in g["nodes"] if x["id"] == nid)
    return n["position"]


def test_sibling_branches_align_on_catalog_rows(tmp_path: Path, monkeypatch) -> None:
    """Branches at different steps share Y rows for the same catalog step."""
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid in ("default", "edges_a", "edges_b", "train_c"):
        run_dir = tmp_path / "data" / rid
        (run_dir / "interim").mkdir(parents=True)
        (configs / f"{rid}.toml").write_text(
            "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
            "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)

    def _branch(
        child_id: str,
        *,
        branch_step_id: str,
        branch_pipeline: str,
        journal_pipeline: str,
        journal_step: str,
        scope: list[str],
    ) -> None:
        cfg = load_resolved_config(configs / f"{child_id}.toml", child_id)
        save_run_manifest(
            cfg.manifest_path,
            RunManifest(
                run_id=child_id,
                parent_run_id="default",
                branch_pipeline=branch_pipeline,
                branch_step_id=branch_step_id,
                created_at="2020-01-01T00:00:00Z",
                config_path=f"configs/{child_id}.toml",
                pipeline_scope=scope,
            ),
        )
        save_step_journal(
            cfg,
            StepJournal(
                entries={
                    f"{journal_pipeline}/{journal_step}": StepJournalEntry(
                        pipeline=journal_pipeline,
                        step_id=journal_step,
                        status="running",
                        started_at="2020-01-01T00:00:00Z",
                    )
                }
            ),
        )

    _branch(
        "edges_a",
        branch_step_id="edges",
        branch_pipeline="graph.prepare",
        journal_pipeline="graph.prepare",
        journal_step="edges",
        scope=["graph.prepare", "dataset.package", "model.cache_graphs", "model.train"],
    )
    _branch(
        "edges_b",
        branch_step_id="edges",
        branch_pipeline="graph.prepare",
        journal_pipeline="graph.prepare",
        journal_step="edges",
        scope=["graph.prepare", "dataset.package", "model.cache_graphs", "model.train"],
    )
    _branch(
        "train_c",
        branch_step_id="train",
        branch_pipeline="model.train",
        journal_pipeline="model.train",
        journal_step="train",
        scope=["model.train", "model.evaluate", "analysis.summarize"],
    )

    g = build_experiment_graph(configs)
    default_train = nodes_pos(g, "run:default:model.train:train")
    train_c_train = nodes_pos(g, "run:train_c:model.train:train")

    default_edges = nodes_pos(g, "run:default:graph.prepare:edges")
    edges_a_edges = nodes_pos(g, "run:edges_a:graph.prepare:edges")
    edges_b_edges = nodes_pos(g, "run:edges_b:graph.prepare:edges")

    assert abs(edges_a_edges["y"] - default_edges["y"]) < 1
    assert abs(edges_b_edges["y"] - default_edges["y"]) < 1
    assert abs(train_c_train["y"] - default_train["y"]) < 1
