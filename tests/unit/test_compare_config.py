"""Tests for mss.analysis.compare_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.analysis.compare_config import (
    build_compare_run_specs,
    config_path_for_run,
    load_compare_runs_config,
    preset_label_overrides,
    resolve_preset,
)
from tests.conftest import make_resolved_config, paths_toml


def test_load_compare_runs_config_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[paths]\n", encoding="utf-8")
    cr = load_compare_runs_config(cfg)
    assert cr.comparison_id == ""
    assert cr.peer_runs == ()
    assert cr.run_labels == ()


def test_load_compare_runs_config_with_section(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        """
[analysis.compare_runs]
comparison_id = "v2_ablations"
peer_runs = ["default_train_scale", "scale_gpu"]

[[analysis.compare_runs.run_labels]]
run_id = "default_train_scale"
label = "512x6 high lr"
column_key = "train_scale"
""".strip(),
        encoding="utf-8",
    )
    cr = load_compare_runs_config(cfg)
    assert cr.comparison_id == "v2_ablations"
    assert cr.peer_runs == ("default_train_scale", "scale_gpu")
    assert len(cr.run_labels) == 1
    assert cr.run_labels[0].label == "512x6 high lr"


def test_resolve_preset_v2_ablations() -> None:
    anchor, peers = resolve_preset("v2_ablations")
    assert anchor == "default"
    assert peers == ("default_train_scale", "scale_gpu")


def test_preset_label_overrides_v2_ablations() -> None:
    overrides = preset_label_overrides("v2_ablations")
    by_id = {o.run_id: o for o in overrides}
    assert by_id["default"].column_key == "default"
    assert by_id["default_train_scale"].label == "512x6 high lr"


def test_config_path_for_run_direct(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "ablation.toml").write_text("[paths]\n", encoding="utf-8")
    p = config_path_for_run(configs, "ablation")
    assert p == (configs / "ablation.toml").resolve()


def test_build_compare_run_specs_anchor_first(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid in ("anchor", "peer"):
        proc = tmp_path / "data" / rid / "processed"
        proc.mkdir(parents=True)
        toml = configs / f"{rid}.toml"
        toml.write_text(
            paths_toml(
                raw=tmp_path / "raw",
                shared_interim=tmp_path / "shared",
                run_interim=tmp_path / "data" / rid / "interim",
                processed=proc,
            ),
            encoding="utf-8",
        )
    specs = build_compare_run_specs("anchor", ("peer",), configs_dir=configs)
    assert [s.run_id for s in specs] == ["anchor", "peer"]
    assert specs[0].primary_variation == "anchor (reference)"
    assert specs[0].column_key == "anchor"
    assert specs[1].label == "peer"


def test_resolve_preset_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown compare preset"):
        resolve_preset("nope")
