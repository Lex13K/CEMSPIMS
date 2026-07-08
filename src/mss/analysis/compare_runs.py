"""Cross-run comparison tables and figures (Appendix E style, Phase 7)."""

from __future__ import annotations

import json
import math
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mss.analysis.compare_config import (
    CompareRunSpec,
    CompareRunsConfig,
    build_compare_run_specs,
    load_compare_runs_config,
)
from mss.io.config import ResolvedConfig
from mss.io.paths import project_root
from mss.processed.paths import (
    comparison_figures_dir,
    comparison_root,
    comparison_tables_dir,
    comparisons_dir,
    diagnostics_smoothing_path,
    hypothesis_tests_path,
    summary_table_path,
)
from mss.processed.readmes import ensure_processed_readmes

SAMPLES_SUMMARY = ("full_test", "excl_2020")
MODEL_ORDER = ("gnn", "vix", "har", "placebo")

SAMPLE_LABEL_E3 = {
    "test": "Full test",
    "test_excl_2020": "Test excluding 2020",
    "test_excl_stress": "Test excluding stress window",
    "test_excl_union": "Test excluding 2020 and stress window",
}

VIX_QUINTILE_RTOL = 1e-9
VIX_QUINTILE_ATOL = 1e-12
NUM_ROUND = 4

TABLE_E01 = "Table_E01_alternative_configuration_comparison.csv"
TABLE_E02 = "Table_E02_held_out_summary_performance_across_configurations.csv"
TABLE_E03 = "Table_E03_formal_hypothesis_results_across_configurations.csv"
TABLE_E04 = "Table_E04_calibration_and_smoothing_diagnostics_across_configurations.csv"
TABLE_E05 = "Table_E05_full_test_mse_log_by_target_volatility_quintile.csv"
FIGURE_E01 = "Figure_E01_full_test_mse_log_by_target_volatility_quintile_across_configurations.png"

DIAG_KEYS = ("forecast_variance_log", "corr_with_target_log", "mz_intercept", "mz_slope")


def fmt_metric_fixed4(x: float) -> str:
    return f"{float(x):.{NUM_ROUND}f}"


def fmt_p_dm_primary(p: float) -> str:
    p = float(p)
    if p > 0 and p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.{NUM_ROUND}f}"


def fmt_statistic(x: float) -> str:
    return f"{float(x):.{NUM_ROUND}f}"


def fmt_p_value(p: float) -> str:
    p = float(p)
    if p <= 0 or not math.isfinite(p):
        return str(p)
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.{NUM_ROUND}f}"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _window_label_from_config(config_path: Path) -> str:
    data = _load_toml(config_path)
    length = (data.get("graph") or {}).get("rolling_window", {}).get("length")
    if length is not None:
        return f"{int(length)}d"
    return "na"


def _placebo_enabled(config_path: Path) -> bool:
    data = _load_toml(config_path)
    return bool(
        (data.get("model") or {}).get("evaluate", {}).get("enable_placebo_ablation", True)
    )


def require_compare_inputs(specs: tuple[CompareRunSpec, ...]) -> None:
    missing: list[str] = []
    for spec in specs:
        for path_fn in (summary_table_path, diagnostics_smoothing_path, hypothesis_tests_path):
            p = path_fn(spec.cfg)
            if not p.is_file():
                missing.append(f"{spec.run_id}: {p.relative_to(spec.cfg.processed_dir.parent)}")
    if missing:
        raise FileNotFoundError(
            "compare_runs requires model.evaluate outputs on each run:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def dependence_label(raw: str) -> str:
    s = str(raw).lower().strip()
    if s == "spearman":
        return "Spearman rank correlation"
    if s == "pearson":
        return "Pearson correlation"
    return raw


def model_type_label(raw: str) -> str:
    s = str(raw).lower().strip()
    if s == "graphsage":
        return "GraphSAGE"
    if s == "gat":
        return "GAT"
    return raw


def yes_no(b: bool | None) -> str:
    if b is None:
        return "Not specified"
    return "Yes" if b else "No"


def build_table_e01(
    specs: tuple[CompareRunSpec, ...],
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    col_keys = [s.column_key for s in specs]
    rows: list[dict[str, str]] = []

    def add(component: str, setting: str, values: tuple[str, ...]) -> None:
        row: dict[str, str] = {"component": component, "setting": setting}
        for key, val in zip(col_keys, values, strict=True):
            row[key] = val
        rows.append(row)

    def mt(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("model", {}).get("train", {})

    def ge(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("model", {}).get("evaluate", {})

    def gw(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("graph", {}).get("rolling_window", {})

    def ged(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("graph", {}).get("edges", {})

    fit_cal = [yes_no(bool(mt(c).get("fit_log_calibration", False))) for c in configs]

    add(
        "Graph construction",
        "Rolling window length (trading days)",
        tuple(str(gw(c).get("length", "")) for c in configs),
    )
    add(
        "Graph construction",
        "Minimum observation fraction",
        tuple(str(gw(c).get("min_obs_frac", "")) for c in configs),
    )
    add(
        "Graph construction",
        "Edge dependence measure",
        tuple(dependence_label(str(ged(c).get("dependence", ""))) for c in configs),
    )
    add(
        "Graph construction",
        "Top-k neighbors per node",
        tuple(str(ged(c).get("top_k", "")) for c in configs),
    )
    add(
        "GNN model",
        "Architecture family",
        tuple(model_type_label(str(mt(c).get("model_type", ""))) for c in configs),
    )
    add(
        "GNN model",
        "Number of message-passing layers",
        tuple(str(mt(c).get("num_gnn_layers", "")) for c in configs),
    )
    add(
        "GNN model",
        "Hidden dimension",
        tuple(str(mt(c).get("hidden_channels", "")) for c in configs),
    )
    add(
        "GNN training",
        "Learning rate",
        tuple(str(mt(c).get("lr", "")) for c in configs),
    )
    add(
        "GNN training",
        "Maximum epochs",
        tuple(str(mt(c).get("max_epochs", "")) for c in configs),
    )
    add(
        "GNN training",
        "Early stopping patience (epochs)",
        tuple(str(mt(c).get("early_stopping_patience", "")) for c in configs),
    )
    add(
        "GNN training",
        "Post-hoc log calibration of forecasts",
        tuple(fit_cal),
    )
    add(
        "Evaluation",
        "Placebo edge ablation enabled",
        tuple(yes_no(bool(ge(c).get("enable_placebo_ablation"))) for c in configs),
    )
    return pd.DataFrame(rows)


def build_table_e02(
    specs: tuple[CompareRunSpec, ...],
    placebo_enabled: dict[str, bool],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for spec in specs:
        rid = spec.run_id
        st = pd.read_csv(summary_table_path(spec.cfg))
        sub = st[st["sample"].isin(SAMPLES_SUMMARY)].copy()
        models = list(MODEL_ORDER)
        if not placebo_enabled[rid]:
            models = [m for m in models if m != "placebo"]
        wl = _window_label_from_config(spec.config_path)
        for sample in SAMPLES_SUMMARY:
            for model in models:
                r = sub[(sub["sample"] == sample) & (sub["model"] == model)]
                if len(r) != 1:
                    raise RuntimeError(
                        f"{rid}: expected one row for sample={sample!r} model={model!r}, got {len(r)}"
                    )
                row = r.iloc[0]
                rows_out.append(
                    {
                        "configuration": spec.label,
                        "window_label": wl,
                        "primary_variation": spec.primary_variation,
                        "sample": sample,
                        "model": model,
                        "mse_log": fmt_metric_fixed4(row["mse_log"]),
                        "mae_log": fmt_metric_fixed4(row["mae_log"]),
                        "qlike": fmt_metric_fixed4(row["qlike"]),
                        "n_samples": int(row["n_samples"]),
                    }
                )

    cfg_order = {spec.label: i for i, spec in enumerate(specs)}
    sample_order = {"full_test": 0, "excl_2020": 1}
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = pd.DataFrame(rows_out)
    df["_c"] = df["configuration"].map(cfg_order)
    df["_s"] = df["sample"].map(sample_order)
    df["_m"] = df["model"].map(model_rank)
    df = df.sort_values(["_c", "_s", "_m"]).drop(columns=["_c", "_s", "_m"])
    return df.reset_index(drop=True)


def inference_mz_h1(stat: float, p_two: float) -> str:
    if p_two >= 0.10:
        return "no clear evidence"
    if p_two >= 0.05:
        return "borderline evidence"
    if stat > 0:
        return "very strong evidence in favor" if p_two < 0.01 else "evidence in favor"
    return "evidence against"


def inference_dm(stat: float, p_primary: float, p_two: float) -> str:
    if stat > 0:
        if p_primary >= 0.10:
            return "no clear evidence"
        if p_primary >= 0.05:
            return "borderline evidence in favor"
        if p_primary >= 0.01:
            return "evidence in favor"
        return "very strong evidence in favor"
    if stat < 0:
        if p_two >= 0.10:
            return "no clear evidence"
        if p_two >= 0.05:
            return "borderline evidence against"
        if p_two >= 0.01:
            return "evidence against"
        return "very strong evidence against"
    return "no clear evidence"


def inference_incremental(stat: float, p_two: float) -> str:
    if p_two >= 0.10:
        return "no clear evidence"
    if p_two >= 0.05:
        if stat > 0:
            return "borderline evidence"
        return "borderline evidence against (wrong direction)"
    if stat > 0:
        return "very strong evidence in favor" if p_two < 0.01 else "evidence in favor"
    return "evidence against (wrong direction)"


def row_e3_for_hypothesis(row: pd.Series) -> dict[str, Any]:
    hid = str(row["hypothesis_id"])
    proc = str(row["inference_procedure"])
    stat = float(row["statistic"])
    p_two = float(row["p_value_two_sided"])
    if math.isnan(p_two):
        p_two = float(row["p_value_primary"])
    p_primary = float(row["p_value_primary"])

    if proc == "mz_gnn":
        p_str = fmt_p_value(p_two)
        inference = inference_mz_h1(stat, p_two)
        procedure = "Mincer–Zarnowitz"
        loss_or_term = "GNN forecast slope (Mincer–Zarnowitz)"
    elif proc == "dm_loss_diff":
        loss_name = str(row["loss_name"])
        loss_or_term = "QLIKE" if loss_name == "qlike" else "MSE(log)"
        p_str = fmt_p_dm_primary(p_primary)
        inference = inference_dm(stat, p_primary, p_two)
        procedure = "Diebold–Mariano"
    elif proc == "incremental_gnn_vix":
        p_str = fmt_p_value(p_two)
        inference = inference_incremental(stat, p_two)
        procedure = "Incremental regression"
        loss_or_term = "GNN coefficient conditional on VIX"
    else:
        raise ValueError(f"Unknown inference_procedure: {proc}")

    sample_key = str(row["sample"])
    sample_label = SAMPLE_LABEL_E3.get(sample_key, sample_key)

    return {
        "hypothesis": hid,
        "sample": sample_label,
        "procedure": procedure,
        "loss_or_term": loss_or_term,
        "statistic": fmt_statistic(stat),
        "p_value": p_str,
        "inference": inference,
    }


def collect_e3_rows(df: pd.DataFrame, *, include_h5: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    h1 = df[(df["hypothesis_id"].astype(str) == "H1") & (df["sample"].astype(str) == "test")]
    if len(h1) != 1:
        raise RuntimeError(f"Expected exactly one H1 row on test, got {len(h1)}")
    out.append(row_e3_for_hypothesis(h1.iloc[0]))

    for loss in ("qlike", "mse_log"):
        h2 = df[
            (df["hypothesis_id"].astype(str) == "H2")
            & (df["sample"].astype(str) == "test")
            & (df["loss_name"].astype(str) == loss)
        ]
        if len(h2) != 1:
            raise RuntimeError(f"Expected one H2 row for loss={loss}, got {len(h2)}")
        out.append(row_e3_for_hypothesis(h2.iloc[0]))

    h3 = df[(df["hypothesis_id"].astype(str) == "H3") & (df["sample"].astype(str) == "test")]
    if len(h3) != 1:
        raise RuntimeError(f"Expected exactly one H3 row on test, got {len(h3)}")
    out.append(row_e3_for_hypothesis(h3.iloc[0]))

    if include_h5:
        for loss in ("qlike", "mse_log"):
            h5 = df[
                (df["hypothesis_id"].astype(str) == "H5")
                & (df["sample"].astype(str) == "test")
                & (df["loss_name"].astype(str) == loss)
            ]
            if len(h5) == 1:
                out.append(row_e3_for_hypothesis(h5.iloc[0]))

    for sample_key in ("test_excl_2020", "test_excl_stress", "test_excl_union"):
        for proc, loss in (
            ("dm_loss_diff", "qlike"),
            ("dm_loss_diff", "mse_log"),
            ("incremental_gnn_vix", None),
        ):
            if proc == "dm_loss_diff":
                h4 = df[
                    (df["hypothesis_id"].astype(str) == "H4")
                    & (df["sample"].astype(str) == sample_key)
                    & (df["inference_procedure"].astype(str) == proc)
                    & (df["loss_name"].astype(str) == loss)
                ]
            else:
                h4 = df[
                    (df["hypothesis_id"].astype(str) == "H4")
                    & (df["sample"].astype(str) == sample_key)
                    & (df["inference_procedure"].astype(str) == proc)
                ]
            if len(h4) != 1:
                raise RuntimeError(
                    f"Expected one H4 row for sample={sample_key} proc={proc} loss={loss}, got {len(h4)}"
                )
            out.append(row_e3_for_hypothesis(h4.iloc[0]))

    return out


def build_table_e03(
    specs: tuple[CompareRunSpec, ...],
    placebo_enabled: dict[str, bool],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for spec in specs:
        df = pd.read_csv(hypothesis_tests_path(spec.cfg))
        rows = collect_e3_rows(df, include_h5=placebo_enabled[spec.run_id])
        part = pd.DataFrame(rows)
        wl = _window_label_from_config(spec.config_path)
        part.insert(0, "primary_variation", spec.primary_variation)
        part.insert(0, "window_label", wl)
        part.insert(0, "configuration", spec.label)
        blocks.append(part)
    return pd.concat(blocks, ignore_index=True)


def build_table_e04(
    specs: tuple[CompareRunSpec, ...],
    placebo_enabled: dict[str, bool],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for spec in specs:
        rid = spec.run_id
        diag = pd.read_csv(diagnostics_smoothing_path(spec.cfg))
        sub = diag[diag["sample"].isin(SAMPLES_SUMMARY)]
        sub = sub[sub["diagnostic"].isin(DIAG_KEYS)]
        models = list(MODEL_ORDER)
        if not placebo_enabled[rid]:
            models = [m for m in models if m != "placebo"]
        wl = _window_label_from_config(spec.config_path)
        for sample in SAMPLES_SUMMARY:
            for model in models:
                r = sub[(sub["sample"] == sample) & (sub["model"] == model)]
                if len(r) != len(DIAG_KEYS):
                    raise RuntimeError(f"{rid}: incomplete diagnostics for {sample}/{model}")
                wide: dict[str, Any] = {
                    "configuration": spec.label,
                    "window_label": wl,
                    "primary_variation": spec.primary_variation,
                    "sample": sample,
                    "model": model,
                }
                for d in DIAG_KEYS:
                    cell = r[r["diagnostic"] == d]
                    if len(cell) != 1:
                        raise RuntimeError(f"{rid}: expected one {d} for {sample}/{model}")
                    wide[d] = fmt_metric_fixed4(float(cell.iloc[0]["value"]))
                rows_out.append(wide)

    cfg_order = {spec.label: i for i, spec in enumerate(specs)}
    sample_order = {"full_test": 0, "excl_2020": 1}
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = pd.DataFrame(rows_out)
    df["_c"] = df["configuration"].map(cfg_order)
    df["_s"] = df["sample"].map(sample_order)
    df["_m"] = df["model"].map(model_rank)
    df = df.sort_values(["_c", "_s", "_m"]).drop(columns=["_c", "_s", "_m"])
    return df.reset_index(drop=True)


def read_vol_quintile_gnn_vix(cfg: ResolvedConfig) -> tuple[dict[str, float], dict[str, float]]:
    diag = pd.read_csv(diagnostics_smoothing_path(cfg))
    ft = diag[
        (diag["sample"] == "full_test")
        & (diag["diagnostic"] == "mse_log_by_target_vol_quantile")
    ]
    gnn: dict[str, float] = {}
    vix: dict[str, float] = {}
    for _, row in ft.iterrows():
        q = str(row["detail"])
        v = float(row["value"])
        if row["model"] == "gnn":
            gnn[q] = v
        elif row["model"] == "vix":
            vix[q] = v
    for d in (gnn, vix):
        if set(d.keys()) != {"q1", "q2", "q3", "q4", "q5"}:
            raise RuntimeError(f"{cfg.run_id}: missing quintile keys in diagnostics")
    return gnn, vix


def assert_vix_quintiles_agree(
    series_by_run: dict[str, dict[str, float]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    runs = list(series_by_run.keys())
    ref = series_by_run[runs[0]]
    for rid in runs[1:]:
        for q in ("q1", "q2", "q3", "q4", "q5"):
            a, b = ref[q], series_by_run[rid][q]
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                raise RuntimeError(
                    "VIX benchmark full-test MSE(log) by target-volatility quintile differs "
                    f"between runs: {runs[0]!r} vs {rid!r} at {q}: {a!r} vs {b!r}. "
                    "Refusing to collapse to a single VIX column."
                )
    return ref


def build_table_e05(specs: tuple[CompareRunSpec, ...]) -> pd.DataFrame:
    gnn_by_run: dict[str, dict[str, float]] = {}
    vix_by_run: dict[str, dict[str, float]] = {}
    for spec in specs:
        gnn, vix = read_vol_quintile_gnn_vix(spec.cfg)
        gnn_by_run[spec.run_id] = gnn
        vix_by_run[spec.run_id] = vix

    vix_canon = assert_vix_quintiles_agree(vix_by_run, rtol=VIX_QUINTILE_RTOL, atol=VIX_QUINTILE_ATOL)

    quintiles = ["q1", "q2", "q3", "q4", "q5"]
    rows = []
    for q in quintiles:
        row: dict[str, Any] = {"target_volatility_quintile": q}
        for spec in specs:
            col = f"{spec.column_key}_gnn"
            row[col] = fmt_metric_fixed4(gnn_by_run[spec.run_id][q])
        row["vix"] = fmt_metric_fixed4(vix_canon[q])
        rows.append(row)
    return pd.DataFrame(rows)


def write_figure_e01(table_e05: pd.DataFrame, specs: tuple[CompareRunSpec, ...], out_path: Path) -> None:
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(q_labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, spec in enumerate(specs):
        col = f"{spec.column_key}_gnn"
        y = table_e05[col].astype(float).values
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            label=f"{spec.label} (GNN)",
            linewidth=1.8,
        )
    if "vix" in table_e05.columns:
        ax.plot(
            x,
            table_e05["vix"].astype(float).values,
            marker="x",
            label="VIX benchmark",
            linewidth=1.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels)
    ax.set_xlabel("Target realized volatility quintile (full test sample)")
    ax.set_ylabel("Mean squared error (log target)")
    ax.set_title("Full-test MSE(log) by target-volatility quintile across configurations")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def write_comparison_note(
    out_path: Path,
    *,
    specs: tuple[CompareRunSpec, ...],
    placebo_enabled: dict[str, bool],
    anchor_run_id: str,
) -> None:
    lines = [
        "# Cross-run comparison (Appendix E style)",
        "",
        f"Anchor run: `{anchor_run_id}`. Compared runs: "
        + ", ".join(f"`{s.run_id}`" for s in specs)
        + ".",
        "",
        "## Per-run variation vs anchor",
        "",
        "| run_id | label | primary_variation | placebo_ablation |",
        "|---|---|---|---|",
    ]
    for spec in specs:
        pe = "yes" if placebo_enabled[spec.run_id] else "no (placebo rows omitted in E.2/E.4)"
        lines.append(
            f"| {spec.run_id} | {spec.label} | {spec.primary_variation} | {pe} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| File | Description |",
            "|---|---|",
            f"| `tables/{TABLE_E01}` | Side-by-side configuration settings |",
            f"| `tables/{TABLE_E02}` | Held-out summary metrics |",
            f"| `tables/{TABLE_E03}` | Formal hypothesis results with inference labels |",
            f"| `tables/{TABLE_E04}` | Calibration / smoothing diagnostics |",
            f"| `tables/{TABLE_E05}` | Quintile MSE(log) source for figure |",
            f"| `figures/{FIGURE_E01}` | Quintile line chart |",
            "",
            "Runs without placebo ablation omit placebo rows in Tables E.2 and E.4.",
            "The VIX quintile column in Table E.5 is checked for agreement across runs before collapsing.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_paths_for_compare(anchor_cfg: ResolvedConfig, comparison_id: str) -> list[Path]:
    root = comparison_root(Path(anchor_cfg.processed_dir), comparison_id)
    tables = comparison_tables_dir(Path(anchor_cfg.processed_dir), comparison_id)
    figures = comparison_figures_dir(Path(anchor_cfg.processed_dir), comparison_id)
    return [
        root / "README.md",
        root / "comparison_manifest.json",
        root / "comparison_note.md",
        tables / TABLE_E01,
        tables / TABLE_E02,
        tables / TABLE_E03,
        tables / TABLE_E04,
        tables / TABLE_E05,
        figures / FIGURE_E01,
    ]


def run_compare_runs(
    anchor_cfg: ResolvedConfig,
    specs: tuple[CompareRunSpec, ...],
    comparison_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Build comparison package under anchor processed/comparisons/<comparison_id>/."""
    if not comparison_id.strip():
        raise ValueError("comparison_id must be non-empty")
    if len(specs) < 2:
        raise ValueError("compare_runs requires at least two runs (anchor + one peer)")

    require_compare_inputs(specs)
    processed = Path(anchor_cfg.processed_dir)
    ensure_processed_readmes(processed)

    out_root = comparison_root(processed, comparison_id)
    out_tables = comparison_tables_dir(processed, comparison_id)
    out_figures = comparison_figures_dir(processed, comparison_id)

    manifest_path = out_root / "comparison_manifest.json"
    if not overwrite and manifest_path.is_file():
        return out_root

    configs = [_load_toml(s.config_path) for s in specs]
    placebo_enabled = {s.run_id: _placebo_enabled(s.config_path) for s in specs}

    out_root.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    e01 = build_table_e01(specs, configs)
    e01.to_csv(out_tables / TABLE_E01, index=False)
    build_table_e02(specs, placebo_enabled).to_csv(out_tables / TABLE_E02, index=False)
    build_table_e03(specs, placebo_enabled).to_csv(out_tables / TABLE_E03, index=False)
    build_table_e04(specs, placebo_enabled).to_csv(out_tables / TABLE_E04, index=False)
    e05 = build_table_e05(specs)
    e05.to_csv(out_tables / TABLE_E05, index=False)
    write_figure_e01(e05, specs, out_figures / FIGURE_E01)

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_comparison_note(
        out_root / "comparison_note.md",
        specs=specs,
        placebo_enabled=placebo_enabled,
        anchor_run_id=anchor_cfg.run_id,
    )

    readme = (
        f"# comparisons/{comparison_id}/\n\n"
        "Cross-run Appendix E-style tables and figures. "
        "Read `comparison_note.md` for per-run variation annotations. "
        "Re-run is read-only on source runs.\n"
    )
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    payload = {
        "comparison_id": comparison_id,
        "anchor_run_id": anchor_cfg.run_id,
        "created_at": ts,
        "git_commit": _git_head(),
        "runs": [
            {
                "run_id": s.run_id,
                "label": s.label,
                "column_key": s.column_key,
                "primary_variation": s.primary_variation,
                "config_path": str(s.config_path.relative_to(anchor_cfg.project_root)).replace("\\", "/"),
            }
            for s in specs
        ],
        "output_paths": [str(p.relative_to(processed)).replace("\\", "/") for p in expected_paths_for_compare(anchor_cfg, comparison_id)],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_root


def run_compare_runs_from_anchor_config(
    anchor_cfg: ResolvedConfig,
    *,
    comparison_id: str | None = None,
    peer_runs: tuple[str, ...] | None = None,
    overwrite: bool = False,
    configs_dir: Path | None = None,
) -> Path:
    """Resolve peers from TOML and/or arguments, then build comparison."""
    cr = load_compare_runs_config(anchor_cfg.source_config_path)
    cid = (comparison_id or cr.comparison_id or "").strip()
    if not cid:
        raise ValueError(
            "comparison_id required (CLI --comparison-id or [analysis.compare_runs].comparison_id)"
        )
    peers = peer_runs if peer_runs is not None else cr.peer_runs
    if not peers:
        raise ValueError("peer_runs required (CLI --runs or [analysis.compare_runs].peer_runs)")
    specs = build_compare_run_specs(
        anchor_cfg.run_id,
        peers,
        configs_dir=configs_dir,
        label_overrides=cr.run_labels,
    )
    return run_compare_runs(anchor_cfg, specs, cid, overwrite=overwrite)
