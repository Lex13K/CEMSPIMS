"""Tests for branch_run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mss.io.config import load_resolved_config
from mss.run.branch import branch_run
from mss.run.manifest import ensure_run_manifest


@pytest.fixture
def branch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.branch.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "parent.toml").write_text(
        "[paths]\n"
        'raw = "data/shared/raw"\n'
        'shared_interim = "data/shared/interim"\n'
        'interim = "data/{run_id}/interim"\n'
        'processed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    shared = tmp_path / "data" / "shared" / "interim"
    shared.mkdir(parents=True)
    (shared / "returns_panel.parquet").write_bytes(b"x")
    (shared / "targets.parquet").write_bytes(b"y")
    parent_run = tmp_path / "data" / "parent"
    (parent_run / "interim" / "graphs").mkdir(parents=True)
    (parent_run / "interim" / "stage04_dates.parquet").write_bytes(b"z")
    (parent_run / "processed").mkdir(parents=True)
    parent_cfg = load_resolved_config(configs / "parent.toml", "parent")
    ensure_run_manifest(parent_cfg)
    return tmp_path


def test_branch_at_graph_prepare_no_crsp_under_child(branch_root: Path) -> None:
    configs = branch_root / "configs"
    branch_run("parent", "child", at_pipeline="graph.prepare", configs_dir=configs)
    child_data = branch_root / "data" / "child"
    assert (configs / "child.toml").is_file()
    assert (child_data / "run_manifest.json").is_file()
    assert not (child_data / "interim" / "crsp_parquet").exists()
    assert not any(child_data.rglob("crsp_parquet"))
    manifest = json.loads((child_data / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == "parent"
    assert manifest["branch_pipeline"] == "graph.prepare"


def test_branch_at_dataset_copies_graphs(branch_root: Path) -> None:
    configs = branch_root / "configs"
    graphs = branch_root / "data" / "parent" / "interim" / "graphs"
    (graphs / "universe.parquet").write_bytes(b"u")
    branch_run("parent", "child2", at_pipeline="dataset.package", configs_dir=configs)
    child_graphs = branch_root / "data" / "child2" / "interim" / "graphs"
    assert child_graphs.is_dir()
    assert (child_graphs / "universe.parquet").is_file()


def test_branch_at_model_train_skips_downstream_artifacts(branch_root: Path) -> None:
    configs = branch_root / "configs"
    parent_run = branch_root / "data" / "parent"
    (parent_run / "interim" / "graphs").mkdir(parents=True, exist_ok=True)
    (parent_run / "interim" / "dataset").mkdir(parents=True)
    (parent_run / "interim" / "dataset" / "manifest.json").write_text("{}", encoding="utf-8")
    (parent_run / "interim" / "model_train").mkdir(parents=True)
    (parent_run / "interim" / "model_train" / "checkpoint.pt").write_bytes(b"ckpt")
    (parent_run / "processed" / "forecasts.parquet").write_text("", encoding="utf-8")

    branch_run("parent", "child3", at_pipeline="model.train", configs_dir=configs)
    child_data = branch_root / "data" / "child3"
    assert (child_data / "interim" / "graphs").is_dir()
    assert (child_data / "interim" / "dataset" / "manifest.json").is_file()
    assert not (child_data / "interim" / "model_train").exists()
    assert not (child_data / "processed").exists()
    manifest = json.loads((child_data / "run_manifest.json").read_text(encoding="utf-8"))
    assert "processed" not in manifest.get("inherited_artifacts", [])
    assert "interim/model_train" not in manifest.get("inherited_artifacts", [])
