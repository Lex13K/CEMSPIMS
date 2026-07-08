"""Tests for processed/comparisons path helpers."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import load_resolved_config
from mss.processed import paths as p


def test_comparison_paths_under_anchor_processed() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_resolved_config(root / "configs" / "default.toml", "default")
    proc = Path(cfg.processed_dir)
    cid = "v2_matrix"
    assert p.comparison_root(proc, cid) == proc / "comparisons" / cid
    assert p.comparison_tables_dir(proc, cid) == proc / "comparisons" / cid / "tables"
    assert p.comparison_figures_dir(proc, cid) == proc / "comparisons" / cid / "figures"
