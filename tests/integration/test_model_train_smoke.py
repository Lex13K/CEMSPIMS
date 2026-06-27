"""Optional smoke tests when `pip install -e ".[train]"` is used."""

from __future__ import annotations

import pytest

try:
    import torch_geometric  # noqa: F401
except ImportError:
    pytest.skip("train extras not installed", allow_module_level=True)

import torch


@pytest.mark.train
def test_require_train_stack_succeeds() -> None:
    from mss.model_train.train_loop import _require_train_stack

    _require_train_stack()


@pytest.mark.train
def test_build_model_graphsage_one_step() -> None:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from mss.model_train.models import build_model

    model = build_model("graphsage", in_channels=3, hidden_channels=8, num_layers=2, pooling="mean")
    data = Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3, 1),
    )
    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out.shape == torch.Size([1])
