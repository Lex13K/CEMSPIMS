"""PyG GNN regressor (GraphSAGE / GAT)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


def _get_pyg_nn():
    import torch_geometric.nn as nn_pyg

    return nn_pyg


class WeightedSAGEConv(nn.Module):
    """GraphSAGE with weighted neighbor mean (PyG 2.8+ SAGEConv has no edge_weight)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalize: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import MessagePassing
        from torch_geometric.nn.dense.linear import Linear
        from torch_geometric.utils import scatter

        class _Conv(MessagePassing):
            def __init__(self) -> None:
                super().__init__(aggr="add")
                self.do_normalize = normalize
                self.lin_l = Linear(in_channels, out_channels, bias=bias)
                self.lin_r = Linear(in_channels, out_channels, bias=False)

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
            ) -> torch.Tensor:
                out = self.propagate(edge_index, x=(x, x), edge_weight=edge_weight)
                _, col = edge_index
                weight_sum = scatter(
                    edge_weight,
                    col,
                    dim=0,
                    dim_size=x.size(0),
                    reduce="sum",
                )
                out = out / weight_sum.clamp(min=1e-8).unsqueeze(-1)
                out = self.lin_l(out) + self.lin_r(x)
                if self.do_normalize:
                    out = F.normalize(out, p=2.0, dim=-1)
                return out

            def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
                return edge_weight.view(-1, 1) * x_j

        self._conv = _Conv()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            raise ValueError("WeightedSAGEConv requires edge_weight")
        return self._conv(x, edge_index, edge_weight)


def build_model(
    gnn_type: str,
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    pooling: str,
    *,
    use_edge_weights: bool = False,
    gat_num_heads: int = 1,
) -> nn.Module:
    """GNN stack -> global pool -> MLP -> scalar output per graph."""
    nn_pyg = _get_pyg_nn()
    gt = gnn_type.lower()
    if gt not in ("graphsage", "gat"):
        raise ValueError(f"Unknown model_type: {gnn_type!r} (use graphsage or gat)")
    if gat_num_heads < 1:
        raise ValueError("gat_num_heads must be >= 1")

    class GNNRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gnn_type = gt
            self.use_edge_weights = use_edge_weights
            self.convs = nn.ModuleList()
            in_ch = in_channels
            for _ in range(num_layers):
                if gt == "gat":
                    edge_dim = 1 if use_edge_weights else None
                    conv = nn_pyg.GATConv(
                        in_ch,
                        hidden_channels,
                        heads=gat_num_heads,
                        edge_dim=edge_dim,
                    )
                    in_ch = hidden_channels * gat_num_heads
                else:
                    if use_edge_weights:
                        conv = WeightedSAGEConv(in_ch, hidden_channels)
                    else:
                        conv = nn_pyg.SAGEConv(in_ch, hidden_channels)
                    in_ch = hidden_channels
                self.convs.append(conv)
            self.pooling = pooling
            self.mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, x, edge_index, edge_attr, batch):
            for conv in self.convs:
                if self.gnn_type == "gat":
                    if self.use_edge_weights and edge_attr is not None:
                        x = conv(x, edge_index, edge_attr).relu()
                    else:
                        x = conv(x, edge_index).relu()
                elif self.use_edge_weights and edge_attr is not None:
                    w = edge_attr.squeeze(-1).abs()
                    x = conv(x, edge_index, w).relu()
                else:
                    x = conv(x, edge_index).relu()
            x = nn_pyg.global_mean_pool(x, batch)
            return self.mlp(x).squeeze(-1)

    return GNNRegressor()
