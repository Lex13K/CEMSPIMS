"""Run-scoped path helpers (shared vs per-run layout)."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import ResolvedConfig


def graphs_dir(cfg: ResolvedConfig) -> Path:
    return cfg.run_interim_dir / "graphs"


def prepare_artifact_paths(cfg: ResolvedConfig) -> dict[str, Path]:
    """Key shared prepare outputs used in manifests."""
    base = cfg.shared_interim_dir
    return {
        "returns_panel.parquet": base / "returns_panel.parquet",
        "targets.parquet": base / "targets.parquet",
    }
