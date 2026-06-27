"""Build one PyG Data per feature date; Dataset over train/val splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mss.model_train.manifest_io import node_features_path_from_manifest
from mss.model_train.dates import split_filtered_dates_and_labels_df


def _date_seed(date: pd.Timestamp, base_seed: int) -> int:
    d = pd.Timestamp(date).normalize()
    return int((int(d.value) ^ int(base_seed)) & 0xFFFFFFFF)


def _placebo_permute_edge_index(
    edge_index: torch.Tensor,
    *,
    n_nodes: int,
    date: pd.Timestamp,
    seed: int,
) -> torch.Tensor:
    if edge_index.numel() == 0 or n_nodes <= 1:
        return edge_index
    rng = np.random.default_rng(_date_seed(date, seed))
    perm = torch.from_numpy(rng.permutation(n_nodes)).long()
    return perm[edge_index]


def _read_parquet_rows_for_normalized_date(
    path: Path,
    columns: list[str],
    date_norm: pd.Timestamp,
    *,
    use_pushdown: bool,
) -> pd.DataFrame:
    """
    Rows for one calendar date across universe / node_features / edges parquets.

    When ``use_pushdown`` is True, try ``filters=[("date", "==", date_norm)]`` first
    for smaller I/O; if that yields no rows (dtype/stats mismatch), fall back to a
    full column read and filter in pandas (same result as pushdown-disabled path).
    """
    if use_pushdown:
        filters = [("date", "==", date_norm)]
        try:
            df = pd.read_parquet(path, columns=columns, filters=filters)
        except (OSError, ValueError, TypeError, KeyError):
            df = pd.read_parquet(path, columns=columns)
        else:
            if len(df) == 0:
                df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_parquet(path, columns=columns)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[df["date"] == date_norm]


def _get_pyg_data():
    import torch_geometric.data as data_mod

    return data_mod


def build_graph_for_date(
    date: pd.Timestamp,
    manifest: dict[str, Any],
    labels_df: pd.DataFrame,
    *,
    use_parquet_pushdown: bool = True,
    edge_mode: str = "actual",
    edge_placebo_seed: int = 7,
) -> Any:
    """Single `torch_geometric.data.Data` for one feature date."""
    data_mod = _get_pyg_data()
    date = pd.to_datetime(date).normalize()
    universe_path = Path(manifest["universe_path"])
    nf_path = node_features_path_from_manifest(manifest)
    edges_path = Path(manifest["edges_path"])
    feature_columns = list(manifest["feature_columns"])

    univ = _read_parquet_rows_for_normalized_date(
        universe_path,
        ["date", "permno"],
        date,
        use_pushdown=use_parquet_pushdown,
    ).sort_values("permno")
    permnos = univ["permno"].tolist()
    if not permnos:
        raise ValueError(f"No universe rows for date {date}")
    permno_to_idx = {int(p): i for i, p in enumerate(permnos)}
    n_nodes = len(permnos)

    nf = _read_parquet_rows_for_normalized_date(
        nf_path,
        ["date", "permno"] + feature_columns,
        date,
        use_pushdown=use_parquet_pushdown,
    )
    nf = nf[nf["permno"].isin(permnos)]
    nf = nf.set_index("permno").reindex(permnos).reset_index()
    x = nf[feature_columns].values.astype(np.float32)
    if np.any(~np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_t = torch.from_numpy(x)

    edges = _read_parquet_rows_for_normalized_date(
        edges_path,
        ["date", "src", "dst", "weight"],
        date,
        use_pushdown=use_parquet_pushdown,
    )
    src_idx = edges["src"].map(permno_to_idx)
    dst_idx = edges["dst"].map(permno_to_idx)
    valid = src_idx.notna() & dst_idx.notna()
    edges = edges.loc[valid]
    src_idx = src_idx.loc[valid].astype(int)
    dst_idx = dst_idx.loc[valid].astype(int)
    edge_index = torch.from_numpy(np.stack([src_idx.values, dst_idx.values])).long()
    edge_attr = torch.from_numpy(edges["weight"].values.astype(np.float32)).unsqueeze(1)
    if edge_mode == "placebo_permute":
        edge_index = _placebo_permute_edge_index(
            edge_index,
            n_nodes=n_nodes,
            date=date,
            seed=edge_placebo_seed,
        )
    elif edge_mode != "actual":
        raise ValueError(f"Unknown edge_mode: {edge_mode!r}")

    row = labels_df[pd.to_datetime(labels_df["date"]).dt.normalize() == date]
    if row.empty or pd.isna(row["y_t"].iloc[0]):
        raise ValueError(f"Missing or NaN label for date {date}")
    y = torch.tensor([[float(row["y_t"].iloc[0])]], dtype=torch.float32)

    return data_mod.Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _cache_path_for_date(cache_dir: Path, date: pd.Timestamp) -> Path:
    return cache_dir / f"{pd.Timestamp(date).strftime('%Y-%m-%d')}.pt"


class GraphDateDataset(Dataset):
    """One graph per date in a split (train or val)."""

    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        *,
        cache_dir: Path | None = None,
        use_parquet_pushdown: bool = True,
        edge_mode: str = "actual",
        edge_placebo_seed: int = 7,
    ) -> None:
        self.manifest = manifest
        self.split = split
        self.cache_dir = cache_dir
        self.use_parquet_pushdown = use_parquet_pushdown
        self.edge_mode = edge_mode
        self.edge_placebo_seed = edge_placebo_seed
        self.dates, self.labels_df = split_filtered_dates_and_labels_df(
            manifest, split
        )

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Any:
        date = self.dates[idx]
        if self.cache_dir is not None and self.edge_mode == "actual":
            path = _cache_path_for_date(self.cache_dir, date)
            if path.is_file():
                return torch.load(path, map_location="cpu", weights_only=False)
        return build_graph_for_date(
            date,
            self.manifest,
            self.labels_df,
            use_parquet_pushdown=self.use_parquet_pushdown,
            edge_mode=self.edge_mode,
            edge_placebo_seed=self.edge_placebo_seed,
        )
