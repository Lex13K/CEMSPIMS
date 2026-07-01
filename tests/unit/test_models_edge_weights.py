"""Edge-weight usage in GNN forward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_graphsage_use_edge_weights_changes_output() -> None:
    from mss.model_train.models import build_model

    model_on = build_model(
        "graphsage", 2, 8, 2, "mean", use_edge_weights=True
    )
    model_off = build_model(
        "graphsage", 2, 8, 2, "mean", use_edge_weights=False
    )
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[0.9], [0.9], [0.1], [0.1]], dtype=torch.float32)
    batch = torch.zeros(3, dtype=torch.long)
    model_on.eval()
    model_off.eval()
    with torch.no_grad():
        y_on = model_on(x, edge_index, edge_attr, batch)
        y_off = model_off(x, edge_index, edge_attr, batch)
    assert not torch.allclose(y_on, y_off)


def test_gat_respects_use_edge_weights_flag() -> None:
    from mss.model_train.models import build_model

    model_on = build_model("gat", 2, 4, 1, "mean", use_edge_weights=True, gat_num_heads=2)
    model_off = build_model("gat", 2, 4, 1, "mean", use_edge_weights=False, gat_num_heads=2)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[0.9], [0.9], [0.1], [0.1]], dtype=torch.float32)
    batch = torch.zeros(3, dtype=torch.long)
    model_on.eval()
    model_off.eval()
    with torch.no_grad():
        y_on = model_on(x, edge_index, edge_attr, batch)
        y_off = model_off(x, edge_index, edge_attr, batch)
    assert y_on.shape == y_off.shape
    assert not torch.allclose(y_on, y_off)
