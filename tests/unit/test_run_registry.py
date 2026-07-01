"""Tests for run registry and config diff."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.run.registry import diff_configs, discover_run_ids, format_run_tree, is_materialized_run, list_runs, summarize_config_diff


@pytest.fixture
def registry_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid in ("a", "b"):
        (tmp_path / "data" / rid).mkdir(parents=True)
    (configs / "a.toml").write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        'interim = "data/{run_id}/interim"\nprocessed = "data/{run_id}/processed"\n'
        "[graph.edges]\ntop_k = 10\n",
        encoding="utf-8",
    )
    (configs / "b.toml").write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        'interim = "data/{run_id}/interim"\nprocessed = "data/{run_id}/processed"\n'
        "[graph.edges]\ntop_k = 15\n",
        encoding="utf-8",
    )
    return configs


def test_list_runs(registry_root: Path) -> None:
    runs = list_runs(registry_root)
    assert {r.run_id for r in runs} == {"a", "b"}


def test_diff_config_summary(registry_root: Path) -> None:
    changes = summarize_config_diff("a", "b", configs_dir=registry_root)
    assert any("graph.edges.top_k" in c for c in changes)


def test_diff_config_unified(registry_root: Path) -> None:
    text = diff_configs("a", "b", configs_dir=registry_root)
    assert "top_k" in text


def test_run_tree(registry_root: Path) -> None:
    tree = format_run_tree(registry_root)
    assert "a" in tree and "b" in tree


def test_config_without_data_dir_not_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run.registry.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "ghost.toml").write_text(
        "[paths]\nraw = \"data/shared/raw\"\nshared_interim = \"data/shared/interim\"\n"
        'interim = "data/{run_id}/interim"\nprocessed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    assert discover_run_ids(configs) == []
    assert not is_materialized_run("ghost", configs)
