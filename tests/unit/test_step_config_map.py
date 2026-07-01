"""Tests for step config map and config patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.app.config_patch import apply_structured_patch
from mss.app.step_config_map import fields_for_step, read_field_values


def test_fields_for_edges() -> None:
    fields = fields_for_step("graph.prepare", "edges")
    keys = {f.dot_key for f in fields}
    assert "graph.edges.top_k" in keys


def test_apply_structured_patch_updates_top_k(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[graph.edges]\ntop_k = 10\ndependence = \"spearman\"\n",
        encoding="utf-8",
    )
    apply_structured_patch(cfg, {"graph.edges.top_k": 11})
    text = cfg.read_text(encoding="utf-8")
    assert "top_k = 11" in text
    assert "dependence" in text


def test_read_field_values(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[graph.edges]\ntop_k = 15\n", encoding="utf-8")
    fields = fields_for_step("graph.prepare", "edges")
    vals = read_field_values(cfg, fields)
    top = next(v for v in vals if v["key"] == "top_k")
    assert top["value"] == 15
