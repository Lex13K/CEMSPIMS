"""Tests for run manifest load/save."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.io.config import load_resolved_config
from mss.run.manifest import RunManifest, ensure_run_manifest, load_run_manifest, save_run_manifest


@pytest.fixture
def cfg_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = configs / "r1.toml"
    cfg.write_text(
        "[paths]\n"
        'raw = "data/shared/raw"\n'
        'shared_interim = "data/shared/interim"\n'
        'interim = "data/{run_id}/interim"\n'
        'processed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    return cfg


def test_ensure_run_manifest_creates_file(cfg_paths: Path) -> None:
    cfg = load_resolved_config(cfg_paths, "r1")
    manifest = ensure_run_manifest(cfg)
    assert manifest.run_id == "r1"
    assert cfg.manifest_path.is_file()
    loaded = load_run_manifest(cfg.manifest_path)
    assert loaded.config_sha256 == manifest.config_sha256


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    m = RunManifest(
        run_id="child",
        parent_run_id="parent",
        branch_pipeline="graph.prepare",
        created_at="2026-01-01T00:00:00+00:00",
        config_path="configs/child.toml",
        config_sha256="abc",
        shared_raw_dir="data/shared/raw",
        shared_interim_dir="data/shared/interim",
        inherited_artifacts=[],
    )
    save_run_manifest(path, m)
    loaded = load_run_manifest(path)
    assert loaded.parent_run_id == "parent"
    assert loaded.branch_pipeline == "graph.prepare"
