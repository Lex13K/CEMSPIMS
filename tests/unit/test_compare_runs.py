"""Tests for mss.analysis.compare_runs table builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mss.analysis.compare_config import CompareRunSpec, build_compare_run_specs
from mss.analysis.compare_runs import (
    MODEL_ORDER,
    SAMPLES_SUMMARY,
    build_table_e01,
    build_table_e02,
    collect_e3_rows,
)
from mss.processed.paths import hypothesis_tests_path, summary_table_path
from tests.conftest import make_resolved_config, paths_toml


def _write_summary_csv(spec: CompareRunSpec, *, placebo: bool) -> None:
    rows: list[str] = []
    models = list(MODEL_ORDER) if placebo else [m for m in MODEL_ORDER if m != "placebo"]
    for sample in SAMPLES_SUMMARY:
        for model in models:
            rows.append(f"{model},{sample},0.0100,0.0200,1.0,1.1,1.2000,10")
    p = summary_table_path(spec.cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "model,sample,mse_log,mae_log,mse,mae,qlike,n_samples\n" + "\n".join(rows),
        encoding="utf-8",
    )


def tomllib_load(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def _base_hypothesis_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "hypothesis_id": "H1",
        "inference_procedure": "mz_gnn",
        "statistic_type": "hac_t",
        "null_hypothesis": "n",
        "alternative": "a",
        "test_name": "t",
        "sample": "test",
        "coefficient_tested": "",
        "test_scope": "s",
        "joint_hypothesis": "none",
        "tail": "two",
        "alternative_direction": "d",
        "better_model": "gnn",
        "loss_name": "",
        "loss_definition": "",
        "statistic": 1.5,
        "p_value_primary": 0.04,
        "p_value_two_sided": 0.08,
        "p_value_one_sided_upper": 0.04,
        "hac_max_lags": 29,
        "n_obs": 100,
        "rows_removed_vs_full_test": 0,
        "effective_sample_start": "2020-01-02",
        "effective_sample_end": "2020-12-31",
        "stress_rows_removed": 0,
        "year_rows_removed": 0,
        "subsample_operational": True,
        "notes": "",
    }
    row.update(overrides)
    return row


def _minimal_hypothesis_csv(path: Path, *, include_h5: bool) -> None:
    rows: list[dict[str, object]] = []
    rows.append(_base_hypothesis_row())
    for loss in ("qlike", "mse_log"):
        rows.append(
            _base_hypothesis_row(
                hypothesis_id="H2",
                inference_procedure="dm_loss_diff",
                loss_name=loss,
                statistic=2.0,
                p_value_primary=0.03,
                p_value_two_sided=0.06,
            )
        )
    rows.append(
        _base_hypothesis_row(
            hypothesis_id="H3",
            inference_procedure="incremental_gnn_vix",
            statistic=1.0,
            p_value_primary=0.05,
            p_value_two_sided=0.10,
        )
    )
    if include_h5:
        for loss in ("qlike", "mse_log"):
            rows.append(
                _base_hypothesis_row(
                    hypothesis_id="H5",
                    inference_procedure="dm_loss_diff",
                    loss_name=loss,
                    statistic=1.1,
                    p_value_primary=0.02,
                    p_value_two_sided=0.04,
                )
            )
    for sample_key in ("test_excl_2020", "test_excl_stress", "test_excl_union"):
        for proc, loss in (
            ("dm_loss_diff", "qlike"),
            ("dm_loss_diff", "mse_log"),
            ("incremental_gnn_vix", None),
        ):
            kw: dict[str, object] = {
                "hypothesis_id": "H4",
                "sample": sample_key,
                "inference_procedure": proc,
                "statistic": 0.5,
                "p_value_primary": 0.20,
                "p_value_two_sided": 0.40,
            }
            if proc == "dm_loss_diff":
                kw["loss_name"] = loss
            else:
                kw["loss_name"] = ""
            rows.append(_base_hypothesis_row(**kw))
    pd.DataFrame(rows).to_csv(path, index=False)


def _setup_run_configs(tmp_path: Path) -> tuple[Path, tuple[CompareRunSpec, ...]]:
    configs = tmp_path / "configs"
    configs.mkdir()
    for rid, placebo in (("anchor", True), ("peer", False)):
        proc = tmp_path / "data" / rid / "processed"
        interim = tmp_path / "data" / rid / "interim"
        toml = configs / f"{rid}.toml"
        eval_block = "enable_placebo_ablation = true" if placebo else "enable_placebo_ablation = false"
        toml.write_text(
            paths_toml(
                raw=tmp_path / "raw",
                shared_interim=tmp_path / "shared",
                run_interim=interim,
                processed=proc,
            )
            + f"\n[graph.rolling_window]\nlength = 20\n\n"
            f"[model.evaluate]\n{eval_block}\n",
            encoding="utf-8",
        )
    specs = build_compare_run_specs("anchor", ("peer",), configs_dir=configs)
    for spec in specs:
        placebo = spec.run_id == "anchor"
        _write_summary_csv(spec, placebo=placebo)
    return configs, specs


def test_build_table_e01_dynamic_columns(tmp_path: Path) -> None:
    configs, specs = _setup_run_configs(tmp_path)
    cfgs = [tomllib_load(s.config_path) for s in specs]
    df = build_table_e01(specs, cfgs)
    assert list(df.columns) == ["component", "setting", "anchor", "peer"]
    assert len(df) > 0


def test_build_table_e02_omits_placebo_when_disabled(tmp_path: Path) -> None:
    _, specs = _setup_run_configs(tmp_path)
    placebo_enabled = {s.run_id: True for s in specs}
    placebo_enabled["peer"] = False
    df = build_table_e02(specs, placebo_enabled)
    peer_rows = df[df["configuration"] == "peer"]
    assert "placebo" not in set(peer_rows["model"])
    anchor_rows = df[df["configuration"] == "anchor"]
    assert "placebo" in set(anchor_rows["model"])


def test_collect_e3_rows_h5_optional(tmp_path: Path) -> None:
    cfg_path = tmp_path / "anchor.toml"
    proc = tmp_path / "processed"
    cfg_path.write_text(
        paths_toml(
            raw=tmp_path / "raw",
            shared_interim=tmp_path / "shared",
            run_interim=tmp_path / "interim",
            processed=proc,
        ),
        encoding="utf-8",
    )
    cfg = make_resolved_config(
        tmp_path,
        run_id="anchor",
        source_config_path=cfg_path,
        processed_dir=proc,
    )
    hpath = hypothesis_tests_path(cfg)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    _minimal_hypothesis_csv(hpath, include_h5=True)
    df = pd.read_csv(hpath)
    with_h5 = collect_e3_rows(df, include_h5=True)
    without_h5 = collect_e3_rows(df, include_h5=False)
    h5_with = [r for r in with_h5 if r["hypothesis"] == "H5"]
    h5_without = [r for r in without_h5 if r["hypothesis"] == "H5"]
    assert len(h5_with) == 2
    assert h5_without == []
