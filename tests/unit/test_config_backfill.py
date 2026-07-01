"""Backfill fingerprint sidecar tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.config.backfill import ensure_config_fingerprint_sidecars
from mss.config.fingerprints import graph_fingerprint_path, read_fingerprint_sidecar
from mss.pipeline import completeness as comp
from tests.conftest import make_resolved_config


def test_backfill_stamps_graph_sidecar_when_semantics_pass(tmp_path: Path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text(
        "[graph]\nret_col = \"ret_used\"\nalign_feature_dates_with_targets = false\n\n"
        "[graph.rolling_window]\nlength = 1\nmin_obs_frac = 1.0\n\n"
        "[graph.universe]\nn_nodes = 1\n",
        encoding="utf-8",
    )
    cfg = make_resolved_config(
        tmp_path,
        source_config_path=cfg_toml.resolve(),
        shared_interim_dir=tmp_path / "shared_interim",
        run_interim_dir=tmp_path / "interim",
    )
    shared = tmp_path / "shared_interim"
    interim = tmp_path / "interim"
    g = interim / "graphs"
    shared.mkdir(parents=True)
    interim.mkdir(parents=True)
    g.mkdir(parents=True)
    dates = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    df = pd.DataFrame(
        {
            "date": dates,
            "permno": [1, 1],
            "permco": [1, 1],
            "shrcd": [10, 10],
            "exchcd": [3, 3],
            "ret_used": [0.01, -0.02],
            "prc": [10.0, 9.0],
            "vol": [1, 1],
            "shrout": [1000, 1000],
            "dlstcd": [None, None],
            "dlret": [None, None],
            "dlretx": [None, None],
            "dlprc": [None, None],
            "dlpdt": [None, None],
            "ticker": ["A", "A"],
            "comnam": ["A", "A"],
            "ncusip": [None, None],
            "cusip": [None, None],
            "secstat": [None, None],
        }
    )
    df.to_parquet(shared / "returns_panel.parquet", index=False)
    pd.DataFrame({"date": dates}).to_parquet(interim / "stage04_dates.parquet", index=False)
    u = pd.DataFrame(
        {"date": dates, "permno": [1, 1], "rank": [1, 1], "mcap": [100.0, 100.0]}
    )
    u.to_parquet(g / "universe.parquet", index=False)
    nf = u.assign(rolling_mean=0.01, rolling_vol=0.02)
    nf.to_parquet(g / "node_features.parquet", index=False)
    e = pd.DataFrame(
        {"date": dates, "src": [1, 1], "dst": [2, 2], "weight": [0.5, 0.5]}
    )
    e.to_parquet(g / "edges.parquet", index=False)

    assert read_fingerprint_sidecar(graph_fingerprint_path(cfg)) is None
    assert comp.graph_prepare_semantic_only(cfg)
    ensure_config_fingerprint_sidecars(cfg)
    assert read_fingerprint_sidecar(graph_fingerprint_path(cfg)) is not None
    assert comp.universe_step_semantically_complete(cfg)
