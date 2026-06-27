"""Unit tests for dataset split assignment and scaler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mss.dataset.config import load_dataset_config
from mss.dataset.scaler import fit_node_feature_scaler
from mss.dataset.splits import build_split_assignment


def test_build_split_assignment_chronological_boundaries() -> None:
    dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-06-01", "2021-01-04"]
    )
    df = build_split_assignment(
        dates,
        train_end="2020-01-02",
        val_end="2020-06-01",
        test_end="2021-12-31",
    )
    assert list(df["split"]) == ["train", "val", "val", "test"]


def test_build_split_assignment_drops_after_test_end() -> None:
    dates = pd.to_datetime(["2020-01-02", "2025-01-01"])
    df = build_split_assignment(
        dates,
        train_end="2019-12-31",
        val_end="2020-12-31",
        test_end="2020-12-31",
    )
    assert len(df) == 1
    assert df["split"].iloc[0] == "val"


def test_build_split_assignment_invalid_order_raises() -> None:
    with pytest.raises(ValueError, match="chronological"):
        build_split_assignment(
            pd.to_datetime(["2020-01-02"]),
            train_end="2020-12-31",
            val_end="2019-01-01",
            test_end="2021-01-01",
        )


def test_fit_node_feature_scaler_train_only(tmp_path: Path) -> None:
    nf = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")] * 2 + [pd.Timestamp("2020-01-03")] * 2,
            "permno": [1, 2, 1, 2],
            "rolling_mean": [1.0, 2.0, 3.0, 4.0],
            "rolling_vol": [0.5, 1.5, 2.5, 3.5],
        }
    )
    train_dates = {pd.Timestamp("2020-01-02")}
    p = fit_node_feature_scaler(
        nf, train_dates, ("rolling_mean", "rolling_vol"), scaler_type="standard"
    )
    assert p["params"]["rolling_mean"]["mean"] == pytest.approx(1.5)
    assert p["params"]["rolling_vol"]["mean"] == pytest.approx(1.0)


def test_load_dataset_config_defaults_target_from_graph(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[graph.output]\ntarget_column = \"custom_tgt\"\n\n"
        "[dataset]\ntrain_end = \"2020-01-01\"\n",
        encoding="utf-8",
    )
    dc = load_dataset_config(cfg)
    assert dc.target_column == "custom_tgt"
