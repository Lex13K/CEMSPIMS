"""Unit tests for config section fingerprints."""

from __future__ import annotations

from pathlib import Path

from mss.config.fingerprints import (
    current_graph_fingerprint,
    fingerprint_config_section,
    fingerprint_toml_table,
    graph_fingerprint_path,
    read_fingerprint_sidecar,
    stored_fingerprint_matches,
    write_fingerprint_sidecar,
)
from tests.conftest import make_resolved_config


def test_fingerprint_toml_table_stable_under_key_order() -> None:
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert fingerprint_toml_table(d1) == fingerprint_toml_table(d2)


def test_fingerprint_config_section_graph(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[graph.universe]\nn_nodes = 500\n", encoding="utf-8")
    b.write_text("[graph.universe]\nn_nodes = 400\n", encoding="utf-8")
    assert fingerprint_config_section(a, "graph") != fingerprint_config_section(b, "graph")


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[graph]\nret_col = \"ret_used\"\n", encoding="utf-8")
    sidecar = tmp_path / "fp.json"
    fp = current_graph_fingerprint(cfg_path)
    write_fingerprint_sidecar(sidecar, section="graph", fingerprint=fp, config_path=cfg_path)
    assert read_fingerprint_sidecar(sidecar) == fp
    assert stored_fingerprint_matches(cfg_path, section="graph", sidecar_path=sidecar)


def test_stored_fingerprint_false_when_missing_sidecar(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[graph]\n", encoding="utf-8")
    assert not stored_fingerprint_matches(
        cfg_path, section="graph", sidecar_path=tmp_path / "missing.json"
    )


def test_graph_fingerprint_path_under_run(tmp_path: Path) -> None:
    cfg = make_resolved_config(tmp_path, run_interim_dir=tmp_path / "data" / "default" / "interim")
    assert graph_fingerprint_path(cfg) == (
        tmp_path / "data" / "default" / "interim" / "graphs" / "config_fingerprint.json"
    )
