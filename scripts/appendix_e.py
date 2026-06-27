"""
Build Appendix E (alternative configurations) export under data/appendixe/ (gitignored).
Run after completing default, middle1, and yearwindow pipelines.

Run from repo root:
  python scripts/appendix_e.py
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo layout and run specs (update paths here if configs move)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "run_id": "default",
        "config_path": REPO_ROOT / "configs" / "default.toml",
        "configuration": "20-day canonical",
        "window_label": "20d",
    },
    {
        "run_id": "middle1",
        "config_path": REPO_ROOT / "configs" / "middle1.toml",
        "configuration": "63-day alternative",
        "window_label": "63d",
    },
    {
        "run_id": "yearwindow",
        "config_path": REPO_ROOT / "configs" / "yearwindow.toml",
        "configuration": "252-day alternative",
        "window_label": "252d",
    },
)

# Keys used in Table E.01 column headers (first run defines names)
E01_COLS = ("default_20d", "middle1_63d", "yearwindow_252d")

OUT_BASE = REPO_ROOT / "data" / "appendixe"
OUT_TABLES = OUT_BASE / "tables"
OUT_FIGURES = OUT_BASE / "figures"

SAMPLES_SUMMARY = ("full_test", "excl_2020")
MODEL_ORDER = ("gnn", "vix", "har", "placebo")

# Formal-test sample key -> thesis-facing label (Table E.3)
SAMPLE_LABEL_E3 = {
    "test": "Full test",
    "test_excl_2020": "Test excluding 2020",
    "test_excl_stress": "Test excluding stress window",
    "test_excl_union": "Test excluding 2020 and stress window",
}

VIX_QUINTILE_RTOL = 1e-9
VIX_QUINTILE_ATOL = 1e-12
NUM_ROUND = 4


def fmt_metric_fixed4(x: float) -> str:
    """Fixed 4-decimal strings for thesis-facing tables (E.2, E.4, E.5)."""
    return f"{float(x):.{NUM_ROUND}f}"


def fmt_p_dm_primary(p: float) -> str:
    """One-sided Diebold–Mariano primary p-value for Table E.3 (4 decimals; scientific if tiny)."""
    p = float(p)
    if p > 0 and p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.{NUM_ROUND}f}"


def fmt_statistic(x: float) -> str:
    return f"{float(x):.{NUM_ROUND}f}"


def fmt_p_value(p: float) -> str:
    """Compact p-value string for thesis tables."""
    p = float(p)
    if p <= 0 or not math.isfinite(p):
        return str(p)
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.{NUM_ROUND}f}"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def summaries_dir(run_id: str) -> Path:
    return REPO_ROOT / "data" / run_id / "processed" / "summaries"


def require_files(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        msg = "Missing required input file(s):\n" + "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(msg)


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


# ---------------------------------------------------------------------------
# Table E.1 — configuration comparison
# ---------------------------------------------------------------------------


def build_table_e01(configs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(component: str, setting: str, values: tuple[str, str, str]) -> None:
        rows.append(
            {
                "component": component,
                "setting": setting,
                E01_COLS[0]: values[0],
                E01_COLS[1]: values[1],
                E01_COLS[2]: values[2],
            }
        )

    def mt(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("model", {}).get("train", {})

    def ge(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("model", {}).get("evaluate", {})

    def gw(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("graph", {}).get("rolling_window", {})

    def ged(c: dict[str, Any]) -> dict[str, Any]:
        return c.get("graph", {}).get("edges", {})

    fit_cal = []
    for c in configs:
        mtrain = mt(c)
        fit_cal.append(yes_no(bool(mtrain.get("fit_log_calibration", False))))

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
        (fit_cal[0], fit_cal[1], fit_cal[2]),
    )
    add(
        "Evaluation",
        "Placebo edge ablation enabled",
        tuple(yes_no(bool(ge(c).get("enable_placebo_ablation"))) for c in configs),
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table E.2 — summary performance
# ---------------------------------------------------------------------------


def build_table_e02(
    run_specs: tuple[dict[str, Any], ...],
    placebo_enabled: dict[str, bool],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for spec in run_specs:
        rid = spec["run_id"]
        st = pd.read_csv(summaries_dir(rid) / "summary_table.csv")
        sub = st[st["sample"].isin(SAMPLES_SUMMARY)].copy()
        models = list(MODEL_ORDER)
        if not placebo_enabled[rid]:
            models = [m for m in models if m != "placebo"]
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
                        "configuration": spec["configuration"],
                        "window_label": spec["window_label"],
                        "sample": sample,
                        "model": model,
                        "mse_log": fmt_metric_fixed4(row["mse_log"]),
                        "mae_log": fmt_metric_fixed4(row["mae_log"]),
                        "qlike": fmt_metric_fixed4(row["qlike"]),
                        "n_samples": int(row["n_samples"]),
                    }
                )

    # Stable ordering
    cfg_order = {spec["configuration"]: i for i, spec in enumerate(run_specs)}
    sample_order = {"full_test": 0, "excl_2020": 1}
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = pd.DataFrame(rows_out)
    df["_c"] = df["configuration"].map(cfg_order)
    df["_s"] = df["sample"].map(sample_order)
    df["_m"] = df["model"].map(model_rank)
    df = df.sort_values(["_c", "_s", "_m"]).drop(columns=["_c", "_s", "_m"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table E.3 — hypothesis tests + inference labels
# ---------------------------------------------------------------------------


def inference_mz_h1(stat: float, p_two: float) -> str:
    """Mincer–Zarnowitz slope on GNN forecast (two-sided). Positive slope supports predictive content."""
    if p_two >= 0.10:
        return "no clear evidence"
    if p_two >= 0.05:
        return "borderline evidence"
    # p_two < 0.05: significant
    if stat > 0:
        return "very strong evidence in favor" if p_two < 0.01 else "evidence in favor"
    return "evidence against"


def inference_dm(stat: float, p_primary: float, p_two: float) -> str:
    """
    Diebold–Mariano: d = VIX loss − GNN loss; one-sided upper p_primary tests E[d] > 0.

    - stat > 0: tier on p_primary (formal one-sided test for GNN lower loss).
    - stat < 0: do not use one-sided p for “against”; tier on p_two + sign (VIX better / negative mean diff).
    """
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
    """Incremental GNN coefficient conditional on VIX (two-sided). Positive coef supports incremental information."""
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
    sample_label = SAMPLE_LABEL_E3[sample_key]

    return {
        "hypothesis": hid,
        "sample": sample_label,
        "procedure": procedure,
        "loss_or_term": loss_or_term,
        "statistic": fmt_statistic(stat),
        "p_value": p_str,
        "inference": inference,
    }


def collect_e3_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Select H1–H4 rows per spec."""
    out: list[dict[str, Any]] = []
    # H1
    h1 = df[(df["hypothesis_id"].astype(str) == "H1") & (df["sample"].astype(str) == "test")]
    if len(h1) != 1:
        raise RuntimeError(f"Expected exactly one H1 row on test, got {len(h1)}")
    out.append(row_e3_for_hypothesis(h1.iloc[0]))

    # H2 — qlike then mse_log
    for loss in ("qlike", "mse_log"):
        h2 = df[
            (df["hypothesis_id"].astype(str) == "H2")
            & (df["sample"].astype(str) == "test")
            & (df["loss_name"].astype(str) == loss)
        ]
        if len(h2) != 1:
            raise RuntimeError(f"Expected one H2 row for loss={loss}, got {len(h2)}")
        out.append(row_e3_for_hypothesis(h2.iloc[0]))

    # H3
    h3 = df[(df["hypothesis_id"].astype(str) == "H3") & (df["sample"].astype(str) == "test")]
    if len(h3) != 1:
        raise RuntimeError(f"Expected exactly one H3 row on test, got {len(h3)}")
    out.append(row_e3_for_hypothesis(h3.iloc[0]))

    # H4 subsamples
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


def build_table_e3(run_specs: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for spec in run_specs:
        df = pd.read_csv(summaries_dir(spec["run_id"]) / "hypothesis_tests.csv")
        rows = collect_e3_rows(df)
        part = pd.DataFrame(rows)
        part.insert(0, "window_label", spec["window_label"])
        part.insert(0, "configuration", spec["configuration"])
        blocks.append(part)
    result = pd.concat(blocks, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Table E.4 — diagnostics
# ---------------------------------------------------------------------------

DIAG_KEYS = ("forecast_variance_log", "corr_with_target_log", "mz_intercept", "mz_slope")


def build_table_e04(
    run_specs: tuple[dict[str, Any], ...],
    placebo_enabled: dict[str, bool],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for spec in run_specs:
        rid = spec["run_id"]
        diag = pd.read_csv(summaries_dir(rid) / "diagnostics_smoothing.csv")
        sub = diag[diag["sample"].isin(SAMPLES_SUMMARY)]
        sub = sub[sub["diagnostic"].isin(DIAG_KEYS)]
        models = list(MODEL_ORDER)
        if not placebo_enabled[rid]:
            models = [m for m in models if m != "placebo"]
        for sample in SAMPLES_SUMMARY:
            for model in models:
                r = sub[(sub["sample"] == sample) & (sub["model"] == model)]
                if len(r) != len(DIAG_KEYS):
                    raise RuntimeError(f"{rid}: incomplete diagnostics for {sample}/{model}")
                wide: dict[str, Any] = {
                    "configuration": spec["configuration"],
                    "window_label": spec["window_label"],
                    "sample": sample,
                    "model": model,
                }
                for d in DIAG_KEYS:
                    cell = r[r["diagnostic"] == d]
                    if len(cell) != 1:
                        raise RuntimeError(f"{rid}: expected one {d} for {sample}/{model}")
                    wide[d] = fmt_metric_fixed4(float(cell.iloc[0]["value"]))
                rows_out.append(wide)

    cfg_order = {spec["configuration"]: i for i, spec in enumerate(run_specs)}
    sample_order = {"full_test": 0, "excl_2020": 1}
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = pd.DataFrame(rows_out)
    df["_c"] = df["configuration"].map(cfg_order)
    df["_s"] = df["sample"].map(sample_order)
    df["_m"] = df["model"].map(model_rank)
    df = df.sort_values(["_c", "_s", "_m"]).drop(columns=["_c", "_s", "_m"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table E.5 + VIX equality check
# ---------------------------------------------------------------------------


def read_vol_quintile_gnn_vix(
    run_id: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (gnn_q -> value, vix_q -> value) for full_test."""
    diag = pd.read_csv(summaries_dir(run_id) / "diagnostics_smoothing.csv")
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
            raise RuntimeError(f"{run_id}: missing quintile keys in diagnostics")
    return gnn, vix


def assert_vix_quintiles_agree(
    series_by_run: dict[str, dict[str, float]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    """Return canonical VIX quintile dict after checking all runs match."""
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


def build_table_e05(run_specs: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    gnn_by_run: dict[str, dict[str, float]] = {}
    vix_by_run: dict[str, dict[str, float]] = {}
    for spec in run_specs:
        rid = spec["run_id"]
        gnn, vix = read_vol_quintile_gnn_vix(rid)
        gnn_by_run[rid] = gnn
        vix_by_run[rid] = vix

    vix_canon = assert_vix_quintiles_agree(vix_by_run, rtol=VIX_QUINTILE_RTOL, atol=VIX_QUINTILE_ATOL)

    quintiles = ["q1", "q2", "q3", "q4", "q5"]
    rows = []
    for q in quintiles:
        rows.append(
            {
                "target_volatility_quintile": q,
                "default_20d_gnn": fmt_metric_fixed4(gnn_by_run["default"][q]),
                "middle1_63d_gnn": fmt_metric_fixed4(gnn_by_run["middle1"][q]),
                "yearwindow_252d_gnn": fmt_metric_fixed4(gnn_by_run["yearwindow"][q]),
                "vix": fmt_metric_fixed4(vix_canon[q]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure E.1
# ---------------------------------------------------------------------------


def write_figure_e01(table_e05: pd.DataFrame, out_path: Path) -> None:
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(q_labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    cols = [
        ("default_20d_gnn", "20-day canonical (GNN)"),
        ("middle1_63d_gnn", "63-day alternative (GNN)"),
        ("yearwindow_252d_gnn", "252-day alternative (GNN)"),
        ("vix", "VIX benchmark"),
    ]
    markers = ["o", "s", "^", "D"]
    for i, (col, label) in enumerate(cols):
        y = table_e05[col].astype(float).values
        ax.plot(x, y, marker=markers[i], label=label, linewidth=1.8)
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


# ---------------------------------------------------------------------------
# Markdown note
# ---------------------------------------------------------------------------


def write_appendix_note(out_path: Path) -> None:
    text = """# Appendix E — Alternative tuned specifications

This folder contains a compact, comparative export for sensitivity to alternative
pipeline configurations. It is **not** a duplicate of the full formal-inference
appendix for each run.

## Contents

| Artifact | Description |
|----------|-------------|
| `tables/Table_E01_alternative_configuration_comparison.csv` | Side-by-side comparison of graph, model, training, and evaluation settings that differ across the canonical 20-day run and two alternative tuned specifications. |
| `tables/Table_E02_held_out_summary_performance_across_configurations.csv` | Held-out metrics (`full_test`, excluding 2020) for GNN, VIX, HAR, and placebo (where applicable). |
| `tables/Table_E03_formal_hypothesis_results_across_configurations.csv` | Formal tests (H1–H4) with thesis-style inference labels. |
| `tables/Table_E04_calibration_and_smoothing_diagnostics_across_configurations.csv` | Forecast variance, correlation with target, and Mincer–Zarnowitz intercept/slope diagnostics. |
| `tables/Table_E05_full_test_mse_log_by_target_volatility_quintile.csv` | Source data for the quintile comparison figure (GNN under each configuration plus the VIX benchmark). |
| `figures/Figure_E01_full_test_mse_log_by_target_volatility_quintile_across_configurations.png` | Line chart of MSE(log) by target-volatility quintile. |

The **20-day default** run remains the **canonical** specification for the thesis (see Appendix B.1). The **63-day** configuration is an alternative tuned specification: it is broadly competitive on held-out metrics but **does not improve** on the canonical run in a way that would motivate replacing it. The **252-day** alternative tuned specification **materially underperforms** the canonical model on the summaries reported here.

## Placebo rows (252-day run)

For the 252-day configuration, placebo edge ablation was **disabled** in the evaluation config (`enable_placebo_ablation = false`). The processed outputs therefore duplicate placebo metrics with the GNN when ablation is off. **Tables E.2 and E.4 omit placebo rows entirely for that configuration** so the appendix does not suggest a distinct placebo benchmark where none was estimated.

## VIX quintile column

The VIX benchmark quintile series was read from each run’s diagnostics and **checked for numerical agreement** across runs before collapsing to a single `vix` column in Table E.5.
"""
    out_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    config_paths = [spec["config_path"] for spec in RUN_SPECS]
    summary_paths: list[Path] = []
    for spec in RUN_SPECS:
        d = summaries_dir(spec["run_id"])
        summary_paths.extend(
            [
                d / "summary_table.csv",
                d / "hypothesis_tests.csv",
                d / "diagnostics_smoothing.csv",
            ]
        )
    require_files(config_paths + summary_paths)

    configs = [_load_toml(p) for p in config_paths]
    placebo_enabled = {
        spec["run_id"]: bool(cfg.get("model", {}).get("evaluate", {}).get("enable_placebo_ablation", False))
        for spec, cfg in zip(RUN_SPECS, configs, strict=True)
    }

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    e01 = build_table_e01(configs)
    e01.to_csv(OUT_TABLES / "Table_E01_alternative_configuration_comparison.csv", index=False)

    e02 = build_table_e02(RUN_SPECS, placebo_enabled)
    e02.to_csv(OUT_TABLES / "Table_E02_held_out_summary_performance_across_configurations.csv", index=False)

    e03 = build_table_e3(RUN_SPECS)
    e03.to_csv(OUT_TABLES / "Table_E03_formal_hypothesis_results_across_configurations.csv", index=False)

    e04 = build_table_e04(RUN_SPECS, placebo_enabled)
    e04.to_csv(OUT_TABLES / "Table_E04_calibration_and_smoothing_diagnostics_across_configurations.csv", index=False)

    e05 = build_table_e05(RUN_SPECS)
    e05.to_csv(OUT_TABLES / "Table_E05_full_test_mse_log_by_target_volatility_quintile.csv", index=False)

    write_figure_e01(e05, OUT_FIGURES / "Figure_E01_full_test_mse_log_by_target_volatility_quintile_across_configurations.png")

    write_appendix_note(OUT_BASE / "Appendix_E_alternative_configurations.md")

    print(f"Wrote Appendix E package under {OUT_BASE}")


if __name__ == "__main__":
    main()
