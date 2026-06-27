"""Parquet pushdown vs full read produce identical `Data` (requires torch-geometric)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_geometric")
import torch

from mss.model_train.pyg_data import build_graph_for_date


def _write_minimal_graph_parquets(tmp: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    d1 = pd.Timestamp("2020-01-02")
    d2 = pd.Timestamp("2020-01-03")
    univ = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "permno": [1, 2, 1, 2],
        }
    )
    nf = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "permno": [1, 2, 1, 2],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.1, 0.2, 0.3, 0.4],
        }
    )
    edges = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "src": [1, 2, 1, 2],
            "dst": [2, 1, 2, 1],
            "weight": [0.5, 0.6, 0.7, 0.8],
        }
    )
    u_path = tmp / "universe.parquet"
    n_path = tmp / "node_features.parquet"
    e_path = tmp / "edges.parquet"
    univ.to_parquet(u_path, index=False)
    nf.to_parquet(n_path, index=False)
    edges.to_parquet(e_path, index=False)
    labels_df = pd.DataFrame(
        {
            "date": [d1, d2],
            "y_t": [0.55, 0.66],
        }
    )
    labels_df["date"] = pd.to_datetime(labels_df["date"])
    return u_path, n_path, e_path, labels_df


def _assert_data_equal(a, b) -> None:
    assert torch.allclose(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.allclose(a.edge_attr, b.edge_attr)
    assert torch.allclose(a.y, b.y)


def test_build_graph_pushdown_matches_no_pushdown(tmp_path: Path) -> None:
    u_path, n_path, e_path, labels_df = _write_minimal_graph_parquets(tmp_path)
    manifest = {
        "universe_path": str(u_path.resolve()),
        "node_features_path": str(n_path.resolve()),
        "edges_path": str(e_path.resolve()),
        "feature_columns": ["f1", "f2"],
    }
    date = pd.Timestamp("2020-01-02")
    off = build_graph_for_date(
        date, manifest, labels_df, use_parquet_pushdown=False
    )
    on = build_graph_for_date(date, manifest, labels_df, use_parquet_pushdown=True)
    _assert_data_equal(on, off)
