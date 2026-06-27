"""PyG GNN regressor (GraphSAGE / GAT)."""

from __future__ import annotations

import torch.nn as nn


def _get_pyg_nn():
    import torch_geometric.nn as nn_pyg

    return nn_pyg


def build_model(
    gnn_type: str,
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    pooling: str,
) -> nn.Module:
    """GNN stack -> global pool -> MLP -> scalar output per graph."""
    nn_pyg = _get_pyg_nn()
    gt = gnn_type.lower()
    GNNConv = nn_pyg.SAGEConv if gt == "graphsage" else nn_pyg.GATConv
    if gt not in ("graphsage", "gat"):
        raise ValueError(f"Unknown model_type: {gnn_type!r} (use graphsage or gat)")

    class GNNRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = nn.ModuleList()
            in_ch = in_channels
            for _ in range(num_layers):
                if gt == "gat":
                    conv = GNNConv(in_ch, hidden_channels, edge_dim=1)
                else:
                    conv = GNNConv(in_ch, hidden_channels)
                self.convs.append(conv)
                in_ch = hidden_channels
            self.pooling = pooling
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, x, edge_index, edge_attr, batch):
            for conv in self.convs:
                if edge_attr is not None and hasattr(conv, "edge_dim") and conv.edge_dim is not None:
                    x = conv(x, edge_index, edge_attr).relu()
                else:
                    x = conv(x, edge_index).relu()
            x = nn_pyg.global_mean_pool(x, batch)
            return self.mlp(x).squeeze(-1)

    return GNNRegressor()
