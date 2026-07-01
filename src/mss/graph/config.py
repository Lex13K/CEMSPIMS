"""Graph pipeline parameters loaded from project TOML `[graph]`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_VALID_RET_COLS = frozenset({"ret_used", "ret_delisted"})
_VALID_UNIVERSE_MODES = frozenset({"fixed_replace", "monthly_rebalance"})
_VALID_REBALANCE_FREQ = frozenset({"daily", "monthly", "quarterly"})
# mcap is the canonical broad point-in-time universe; dollar_volume is an optional robustness ranking.
_VALID_SELECTION_RULES = frozenset({"mcap", "dollar_volume"})


def validate_universe_settings(universe_mode: str, rebalance_freq: str) -> None:
    mode = str(universe_mode).strip().lower()
    freq = str(rebalance_freq).strip().lower()
    if mode not in _VALID_UNIVERSE_MODES:
        raise ValueError(
            f"[graph.universe].universe_mode must be one of {sorted(_VALID_UNIVERSE_MODES)}; got {universe_mode!r}"
        )
    if freq not in _VALID_REBALANCE_FREQ:
        raise ValueError(
            f"[graph.universe].rebalance_freq must be one of {sorted(_VALID_REBALANCE_FREQ)}; got {rebalance_freq!r}"
        )
    if mode == "monthly_rebalance" and freq not in _VALID_REBALANCE_FREQ:
        raise ValueError("monthly_rebalance requires a valid rebalance_freq")


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
    restrict_to_sp500: bool
    sp500_membership_path: str
    universe_n_jobs: int
    universe_progress_every: int
    dependence: str
    top_k: int
    symmetrize: bool
    edges_n_jobs: int
    edges_update_every: int
    rolling_mean: bool
    rolling_vol: bool
    include_log_dollar_volume: bool
    include_turnover: bool
    include_downside_semivol: bool
    include_skew: bool
    node_features_n_jobs: int
    node_features_progress_every: int
    target_column: str

    def node_feature_column_names(self) -> tuple[str, ...]:
        cols: list[str] = []
        if self.rolling_mean:
            cols.append("rolling_mean")
        if self.rolling_vol:
            cols.append("rolling_vol")
        if self.include_log_dollar_volume:
            cols.append("log_dollar_volume")
        if self.include_turnover:
            cols.append("turnover")
        if self.include_downside_semivol:
            cols.append("downside_semivol")
        if self.include_skew:
            cols.append("skew")
        return tuple(cols)


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
    if ret_col not in _VALID_RET_COLS:
        raise ValueError(
            f"[graph].ret_col must be one of {sorted(_VALID_RET_COLS)}; got {ret_col!r}"
        )
    align = bool(g.get("align_feature_dates_with_targets", True))
    show_progress = bool(g.get("show_progress", True))
    verbose = bool(g.get("verbose", True))
    debug_progress = bool(g.get("debug_progress", False))

    universe_mode = str(u.get("universe_mode", "fixed_replace"))
    rebalance_freq = str(u.get("rebalance_freq", "monthly"))
    validate_universe_settings(universe_mode, rebalance_freq)
    selection_rule = str(u.get("selection_rule", "mcap")).strip().lower()
    if selection_rule not in _VALID_SELECTION_RULES:
        raise ValueError(
            f"[graph.universe].selection_rule must be one of {sorted(_VALID_SELECTION_RULES)}; "
            f"got {u.get('selection_rule')!r}"
        )

    return GraphConfig(
        show_progress=show_progress,
        verbose=verbose,
        debug_progress=debug_progress,
        window_length=window_length,
        min_obs_frac=min_obs_frac,
        ret_col=ret_col,
        align_feature_dates_with_targets=align,
        universe_mode=universe_mode,
        n_nodes=int(u.get("n_nodes", 500)),
        selection_rule=selection_rule,
        rebalance_freq=rebalance_freq,
        restrict_to_sp500=bool(u.get("restrict_to_sp500", False)),
        sp500_membership_path=str(u.get("sp500_membership_path", "")).strip(),
        universe_n_jobs=int(u.get("n_jobs", 1)),
        universe_progress_every=int(u.get("progress_every", 50)),
        dependence=str(e.get("dependence", "spearman")),
        top_k=int(e.get("top_k", 10)),
        symmetrize=bool(e.get("symmetrize", True)),
        edges_n_jobs=int(e.get("n_jobs", 1)),
        edges_update_every=int(e.get("update_every", 100)),
        rolling_mean=bool(nf.get("rolling_mean", True)),
        rolling_vol=bool(nf.get("rolling_vol", True)),
        include_log_dollar_volume=bool(nf.get("include_log_dollar_volume", False)),
        include_turnover=bool(nf.get("include_turnover", False)),
        include_downside_semivol=bool(nf.get("include_downside_semivol", False)),
        include_skew=bool(nf.get("include_skew", False)),
        node_features_n_jobs=int(nf.get("n_jobs", 1)),
        node_features_progress_every=int(nf.get("progress_every", 50)),
        target_column=str(out.get("target_column", "log_rv_fwd_30cal")),
    )


def min_obs_from_config(gc: GraphConfig) -> int:
    return int(gc.window_length * gc.min_obs_frac)
