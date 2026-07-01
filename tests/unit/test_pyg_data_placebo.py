"""Placebo ablation: permuted topology with zeroed edge weights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_geometric")
import torch

from mss.model_train.pyg_data import build_graph_for_date


def _write_fixture(tmp: Path) -> tuple[dict, pd.DataFrame, pd.Timestamp]:
    d1 = pd.Timestamp("2020-01-02")
    permnos = [1, 2, 3]
    univ = pd.DataFrame({"date": [d1] * len(permnos), "permno": permnos})
    nf = pd.DataFrame(
        {
            "date": [d1] * len(permnos),
            "permno": permnos,
            "f1": [1.0, 2.0, 3.0],
            "f2": [0.1, 0.2, 0.3],
        }
    )
    edges = pd.DataFrame(
        {
            "date": [d1, d1, d1],
            "src": [1, 2, 1],
            "dst": [2, 3, 3],
            "weight": [0.75, 0.55, 0.35],
        }
    )
    u_path = tmp / "universe.parquet"
    n_path = tmp / "node_features.parquet"
    e_path = tmp / "edges.parquet"
    univ.to_parquet(u_path, index=False)
    nf.to_parquet(n_path, index=False)
    edges.to_parquet(e_path, index=False)
    labels_df = pd.DataFrame({"date": [d1], "y_t": [0.5]})
    labels_df["date"] = pd.to_datetime(labels_df["date"])
    manifest = {
        "universe_path": str(u_path.resolve()),
        "node_features_path": str(n_path.resolve()),
        "edges_path": str(e_path.resolve()),
        "feature_columns": ["f1", "f2"],
    }
    return manifest, labels_df, d1


def test_placebo_zero_attr_permuted_topology_zero_weights(tmp_path: Path) -> None:
    manifest, labels_df, date = _write_fixture(tmp_path)
    actual = build_graph_for_date(date, manifest, labels_df, edge_mode="actual")
    placebo = build_graph_for_date(
        date,
        manifest,
        labels_df,
        edge_mode="placebo_permute",
        edge_placebo_seed=7,
        placebo_edge_mode="zero_attr",
    )
    assert not torch.equal(actual.edge_index, placebo.edge_index)
    assert torch.allclose(placebo.edge_attr, torch.zeros_like(placebo.edge_attr))
    assert not torch.allclose(actual.edge_attr, placebo.edge_attr)


def test_placebo_permute_only_keeps_weights(tmp_path: Path) -> None:
    manifest, labels_df, date = _write_fixture(tmp_path)
    actual = build_graph_for_date(date, manifest, labels_df, edge_mode="actual")
    placebo = build_graph_for_date(
        date,
        manifest,
        labels_df,
        edge_mode="placebo_permute",
        edge_placebo_seed=7,
        placebo_edge_mode="permute_only",
    )
    assert not torch.equal(actual.edge_index, placebo.edge_index)
    assert torch.allclose(actual.edge_attr, placebo.edge_attr)
