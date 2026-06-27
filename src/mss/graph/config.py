"""Graph pipeline parameters loaded from project TOML `[graph]`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphConfig:
    show_progress: bool
    verbose: bool
    debug_progress: bool
    window_length: int
    min_obs_frac: float
    ret_col: str
    align_feature_dates_with_targets: bool
    universe_mode: str
    n_nodes: int
    selection_rule: str
    rebalance_freq: str
    universe_n_jobs: int
    universe_progress_every: int
    dependence: str
    top_k: int
    symmetrize: bool
    edges_n_jobs: int
    edges_update_every: int
    rolling_mean: bool
    rolling_vol: bool
    node_features_n_jobs: int
    node_features_progress_every: int
    target_column: str


def load_graph_config(config_path: Path) -> GraphConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    g = data.get("graph") or {}
    rw = g.get("rolling_window") or {}
    u = g.get("universe") or {}
    e = g.get("edges") or {}
    nf = g.get("node_features") or {}
    out = g.get("output") or {}

    window_length = int(rw.get("length", 20))
    min_obs_frac = float(rw.get("min_obs_frac", 0.8))
    ret_col = str(g.get("ret_col", "ret_used"))
    align = bool(g.get("align_feature_dates_with_targets", True))
    show_progress = bool(g.get("show_progress", True))
    verbose = bool(g.get("verbose", True))
    debug_progress = bool(g.get("debug_progress", False))

    return GraphConfig(
        show_progress=show_progress,
        verbose=verbose,
        debug_progress=debug_progress,
        window_length=window_length,
        min_obs_frac=min_obs_frac,
        ret_col=ret_col,
        align_feature_dates_with_targets=align,
        universe_mode=str(u.get("universe_mode", "fixed_replace")),
        n_nodes=int(u.get("n_nodes", 500)),
        selection_rule=str(u.get("selection_rule", "mcap")),
        rebalance_freq=str(u.get("rebalance_freq", "monthly")),
        universe_n_jobs=int(u.get("n_jobs", 1)),
        universe_progress_every=int(u.get("progress_every", 50)),
        dependence=str(e.get("dependence", "spearman")),
        top_k=int(e.get("top_k", 10)),
        symmetrize=bool(e.get("symmetrize", True)),
        edges_n_jobs=int(e.get("n_jobs", 1)),
        edges_update_every=int(e.get("update_every", 100)),
        rolling_mean=bool(nf.get("rolling_mean", True)),
        rolling_vol=bool(nf.get("rolling_vol", True)),
        node_features_n_jobs=int(nf.get("n_jobs", 1)),
        node_features_progress_every=int(nf.get("progress_every", 50)),
        target_column=str(out.get("target_column", "log_rv_fwd_30")),
    )


def min_obs_from_config(gc: GraphConfig) -> int:
    return int(gc.window_length * gc.min_obs_frac)
