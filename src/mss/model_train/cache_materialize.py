"""Pre-materialize per-date PyG graphs for faster `model.train`.

This is the non-optional canonical "graph cache" pipeline step:
- it writes `interim/model_train/graph_cache/YYYY-MM-DD.pt`
- it uses the same manifest + split/label filtering logic as GraphDateDataset
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from mss.io.config import ResolvedConfig
from mss.model_train.config import load_model_train_config
from mss.model_train.dates import train_val_expected_dates_and_labels_df
from mss.model_train.checks import model_train_dir
from mss.model_train.manifest_io import load_dataset_manifest


def materialize_pyg_graph_cache(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    # Torch/PyG imports are intentionally inside the function so that
    # non-train environments can still import the rest of the repo.
    import torch

    from mss.model_train.pyg_data import build_graph_for_date

    interim = Path(cfg.interim_dir)
    manifest_path = interim / "dataset" / "manifest.json"
    manifest = load_dataset_manifest(manifest_path)

    mcfg = load_model_train_config(cfg.source_config_path)

    out_dir = model_train_dir(cfg)
    cache_dir = out_dir / "graph_cache"
    if overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_dates, labels_df = train_val_expected_dates_and_labels_df(manifest)
    if not expected_dates:
        raise ValueError(
            "No train/val dates with non-NaN labels found. "
            "Check [dataset] split boundaries and targets construction."
        )

    for date in tqdm(
        expected_dates,
        desc="Building PyG graph cache",
        unit="date",
        file=sys.stderr,
        disable=not mcfg.verbose,
    ):
        pt_path = cache_dir / f"{date.strftime('%Y-%m-%d')}.pt"
        if pt_path.exists() and not overwrite:
            try:
                if pt_path.stat().st_size > 0:
                    continue
            except OSError:
                # Treat unreadable/invalid cache as missing and rebuild.
                pass
        data = build_graph_for_date(
            date,
            manifest,
            labels_df,
            use_parquet_pushdown=mcfg.parquet_date_pushdown,
        )
        torch.save(data, pt_path)

