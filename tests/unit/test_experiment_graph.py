"""Tests for mss.run.experiment_graph."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.run.experiment_graph import build_experiment_graph
from tests.conftest import make_resolved_config, paths_toml


def _write_run_config(configs: Path, tmp: Path, run_id: str, *, parent: str | None = None, branch_at: str | None = None) -> None:
    proc = tmp / "data" / run_id / "processed"
    interim = tmp / "data" / run_id / "interim"
    proc.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    toml = configs / f"{run_id}.toml"
    toml.write_text(
        paths_toml(
            raw=tmp / "raw",
            shared_interim=tmp / "shared",
            run_interim=interim,
            processed=proc,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "parent_run_id": parent,
        "branch_pipeline": branch_at,
        "created_at": "2026-01-01T00:00:00Z",
        "config_path": f"configs/{run_id}.toml",
        "config_sha256": "",
        "shared_raw_dir": "data/shared/raw",
        "shared_interim_dir": "data/shared/interim",
        "prepare_artifact_hashes": {},
        "inherited_artifacts": [],
    }
    import json

    (tmp / "data" / run_id / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_experiment_graph_empty_configs(tmp_path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    g = build_experiment_graph(configs)
    assert g["nodes"] == []


def test_experiment_graph_has_raw_and_terminal(tmp_path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_run_config(configs, tmp_path, "default")
    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    g = build_experiment_graph(configs)
    ids = {n["id"] for n in g["nodes"]}
    assert "global:raw" in ids
    assert "terminal:default" in ids
    assert any(n["kind"] == "global_step" for n in g["nodes"])
    # Single root run: entire spine shares x=0 for straight vertical edges.
    xs = {n["position"]["x"] for n in g["nodes"]}
    assert xs == {0}


def test_experiment_graph_sequence_edges_chain(tmp_path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_run_config(configs, tmp_path, "default")
    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    g = build_experiment_graph(configs)
    seq = [e for e in g["edges"] if e["kind"] == "sequence"]
    assert len(seq) == len(g["nodes"]) - 1
    targets = {e["target"] for e in seq}
    sources = {e["source"] for e in seq}
    assert "global:raw" in sources
    assert "terminal:default" in targets


def test_experiment_graph_branch_edge(tmp_path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_run_config(configs, tmp_path, "default")
    _write_run_config(configs, tmp_path, "child", parent="default", branch_at="model.train")
    monkeypatch.setattr("mss.run.experiment_graph.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    child_cfg = load_resolved_config(configs / "child.toml", "child")
    from mss.run.manifest import RunManifest, load_run_manifest, save_run_manifest

    man = load_run_manifest(child_cfg.manifest_path)
    man = RunManifest(
        **{**man.to_dict(), "branch_step_id": "train"}
    )
    save_run_manifest(child_cfg.manifest_path, man)
    from mss.run.step_journal import StepJournal, StepJournalEntry, save_step_journal

    save_step_journal(
        child_cfg,
        StepJournal(
            entries={
                "model.train/train": StepJournalEntry(
                    pipeline="model.train",
                    step_id="train",
                    status="running",
                    started_at="2026-01-01T00:00:00Z",
                )
            }
        ),
    )
    g = build_experiment_graph(configs)
    branch_edges = [e for e in g["edges"] if e["kind"] == "branch"]
    assert len(branch_edges) >= 1
    assert branch_edges[0]["child_run_id"] == "child"
