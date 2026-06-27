"""Unit tests for canonical PyG graph cache completion/materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mss.io.config import ResolvedConfig
from mss.pipeline.completeness import model_cache_graphs_step_semantically_complete
from mss.pipeline.artifacts import ensure_step_inputs_ready, step_is_complete


def _cfg(tmp: Path, *, run: str = "t") -> ResolvedConfig:
    cfg_toml = tmp / "cfg.toml"
    cfg_toml.write_text("", encoding="utf-8")
    return ResolvedConfig(
        run_id=run,
        project_root=tmp,
        raw_dir=tmp / "raw",
        interim_dir=tmp / "interim",
        processed_dir=tmp / "processed",
        source_config_path=cfg_toml.resolve(),
    )


def _write_minimal_manifest_and_inputs(tmp: Path, *, target_column: str) -> dict[str, Path]:
    interim = tmp / "interim"
    ds_dir = interim / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    splits_path = tmp / "splits.parquet"
    targets_path = tmp / "targets.parquet"
    universe_path = tmp / "universe.parquet"
    node_features_path = tmp / "node_features.parquet"
    edges_path = tmp / "edges.parquet"
    scaler_path = tmp / "scaler_params.json"

    d_train = pd.Timestamp("2020-01-02")
    d_val = pd.Timestamp("2020-01-03")
    splits = pd.DataFrame(
        {"date": [d_train, d_val], "split": ["train", "val"]}
    )
    splits.to_parquet(splits_path, index=False)

    targets = pd.DataFrame(
        {"date": [d_train, d_val], target_column: [0.1, 0.2]}
    )
    targets.to_parquet(targets_path, index=False)

    # Universe: both permnos exist on both dates.
    universe = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val],
            "permno": [1, 2, 1, 2],
        }
    )
    universe.to_parquet(universe_path, index=False)

    feature_columns = ["f1", "f2"]
    nf = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val],
            "permno": [1, 2, 1, 2],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.1, 0.2, 0.3, 0.4],
        }
    )
    nf.to_parquet(node_features_path, index=False)

    edges = pd.DataFrame(
        {
            "date": [d_train, d_train, d_val, d_val],
            "src": [1, 2, 1, 2],
            "dst": [2, 1, 2, 1],
            "weight": [0.5, 0.6, 0.7, 0.8],
        }
    )
    edges.to_parquet(edges_path, index=False)

    scaler_path.write_text("{}", encoding="utf-8")

    manifest = {
        "splits_path": splits_path.resolve().as_posix(),
        "node_features_path": node_features_path.resolve().as_posix(),
        "edges_path": edges_path.resolve().as_posix(),
        "universe_path": universe_path.resolve().as_posix(),
        "scaler_path": scaler_path.resolve().as_posix(),
        "targets_path": targets_path.resolve().as_posix(),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "scale_target": False,
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return {
        "interim": interim,
        "ds_dir": ds_dir,
        "splits_path": splits_path,
        "targets_path": targets_path,
        "universe_path": universe_path,
        "node_features_path": node_features_path,
        "edges_path": edges_path,
        "scaler_path": scaler_path,
        "d_train": d_train,
        "d_val": d_val,
        "feature_columns": feature_columns,
    }


def test_model_cache_graphs_step_semantically_complete(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    tmp_files = _write_minimal_manifest_and_inputs(tmp_path, target_column="log_rv_fwd_30")

    # Expected cache files are train+val dates with non-NaN labels (both dates here).
    cache_dir = tmp_path / "interim" / "model_train" / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "2020-01-02.pt").write_bytes(b"x")
    (cache_dir / "2020-01-03.pt").write_bytes(b"y")

    assert model_cache_graphs_step_semantically_complete(cfg)

    # Missing one file => incomplete
    (cache_dir / "2020-01-03.pt").write_bytes(b"")
    assert not model_cache_graphs_step_semantically_complete(cfg)


def test_ensure_model_cache_graphs_inputs_ready_requires_manifest(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="manifest.json"):
        ensure_step_inputs_ready("model.cache_graphs", "materialize", cfg)


def test_step_is_complete_model_cache_graphs_respects_final_files(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_minimal_manifest_and_inputs(tmp_path, target_column="log_rv_fwd_30")
    assert not step_is_complete("model.cache_graphs", "materialize", cfg)

    cache_dir = tmp_path / "interim" / "model_train" / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "2020-01-02.pt").write_bytes(b"x")
    (cache_dir / "2020-01-03.pt").write_bytes(b"y")
    assert step_is_complete("model.cache_graphs", "materialize", cfg)


@pytest.mark.train
def test_materialize_pyg_graph_cache_writes_pt_files(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from mss.model_train.cache_materialize import materialize_pyg_graph_cache

    cfg = _cfg(tmp_path)
    _write_minimal_manifest_and_inputs(tmp_path, target_column="log_rv_fwd_30")

    materialize_pyg_graph_cache(cfg, overwrite=True)

    cache_dir = tmp_path / "interim" / "model_train" / "graph_cache"
    pt1 = cache_dir / "2020-01-02.pt"
    pt2 = cache_dir / "2020-01-03.pt"
    assert pt1.is_file() and pt1.stat().st_size > 0
    assert pt2.is_file() and pt2.stat().st_size > 0

    obj = torch.load(pt1, map_location="cpu", weights_only=False)
    assert hasattr(obj, "x") and hasattr(obj, "edge_index") and hasattr(obj, "y")

