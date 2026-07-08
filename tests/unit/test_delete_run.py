"""Tests for delete_run."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.run.delete_run import delete_run
from mss.run.registry import discover_run_ids, is_materialized_run


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.delete_run.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid in ("default", "child"):
        (tmp_path / "data" / rid).mkdir(parents=True)
        (configs / f"{rid}.toml").write_text(
            "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
            'interim = "data/{run_id}/interim"\nprocessed = "data/{run_id}/processed"\n',
            encoding="utf-8",
        )
    return configs


def test_delete_run_removes_data_and_config(runs_root: Path) -> None:
    delete_run("child", configs_dir=runs_root)
    assert not is_materialized_run("child", runs_root)
    assert discover_run_ids(runs_root) == ["default"]


def test_delete_run_blocks_when_children_exist(runs_root: Path, tmp_path: Path) -> None:
    (tmp_path / "data" / "grandchild").mkdir()
    (runs_root / "grandchild.toml").write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        'interim = "data/{run_id}/interim"\nprocessed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    import json

    (tmp_path / "data" / "grandchild" / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "grandchild",
                "parent_run_id": "default",
                "created_at": "x",
                "config_path": "configs/grandchild.toml",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="child runs first"):
        delete_run("default", configs_dir=runs_root)
