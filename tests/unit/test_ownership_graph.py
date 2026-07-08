"""Tests for branch graph visibility rules."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.run.experiment_graph import build_experiment_graph
from mss.run.manifest import RunManifest, ensure_run_manifest, save_run_manifest
from mss.run.ownership import include_owned_step_in_graph


def test_include_owned_step_shows_downstream_in_scope_without_journal() -> None:
    assert include_owned_step_in_graph(
        parent_run_id="default",
        branch_pipeline="model.train",
        branch_step_id="train",
        pipeline="model.evaluate",
        step_id="score_splits",
        owner="child",
        run_id="child",
        pipeline_scope=["model.train", "model.evaluate"],
        touched=False,
    )


def test_scoped_evaluate_nodes_visible_on_graph(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid in ("parent", "child"):
        run_dir = tmp_path / "data" / rid
        (run_dir / "interim").mkdir(parents=True)
        (configs / f"{rid}.toml").write_text(
            "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
            "interim = \"data/{run_id}/interim\"\nprocessed = \"data/{run_id}/processed\"\n",
            encoding="utf-8",
        )

    parent_cfg = load_resolved_config(configs / "parent.toml", "parent")
    ensure_run_manifest(parent_cfg)
    child_cfg = load_resolved_config(configs / "child.toml", "child")
    save_run_manifest(
        child_cfg.manifest_path,
        RunManifest(
            run_id="child",
            parent_run_id="parent",
            branch_pipeline="model.train",
            branch_step_id="train",
            created_at="2020-01-01T00:00:00Z",
            config_path="configs/child.toml",
            pipeline_scope=["model.train", "model.evaluate"],
        ),
    )

    g = build_experiment_graph(configs)
    child_eval = [
        n for n in g["nodes"] if n["id"].startswith("run:child:model.evaluate:")
    ]
    assert child_eval, "evaluate steps should appear once in pipeline scope"
