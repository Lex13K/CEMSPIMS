"""Export thesis-ready exhibits from processed evaluation artifacts.

Appendix **table** outputs are **CSV only**, with thesis-facing names like ``Table_A01_...``,
``Table_B01_...``, ``Table_C01_...``, ``Table_D01_...``. Raw/pipeline duplicates are prefixed
``OLD_``. Main-chapter tables (``main/tables/M*.csv``) may still emit ``.tex`` for LaTeX workflows.

Figure outputs: ``Figure_F01_*`` under ``appendix/figures/``. Reproducibility metadata under ``manifest/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mss.dataset.config import load_dataset_config
from mss.evaluation.config import load_model_evaluate_config
from mss.graph.config import load_graph_config
from mss.io.config import ResolvedConfig
from mss.model_train.config import load_model_train_config


MODEL_ORDER = ("gnn", "vix", "har", "placebo")
MODEL_LABEL = {"gnn": "GNN", "vix": "VIX", "har": "HAR", "placebo": "Placebo"}
PROC_LABEL = {
    "mz_gnn": "MZ",
    "dm_loss_diff": "DM loss diff",
    "incremental_gnn_vix": "Incremental (GNN|VIX)",
}
MODEL_COLOR = {"gnn": "tab:blue", "vix": "tab:red", "har": "tab:green", "placebo": "tab:purple"}
QUINTILE_ORDER = ("q1", "q2", "q3", "q4", "q5")
QUINTILE_ROW_LABEL = {
    "q1": "Q1 (lowest)",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4",
    "q5": "Q5 (highest)",
}


@dataclass(frozen=True)
class ExhibitRecord:
    exhibit_id: str
    title: str
    section: str  # main | appendix
    output_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    filter_logic: str
    output_formats: tuple[str, ...]
    build_timestamp_utc: str
    thesis_label: str = ""  # e.g. "Table T01", "Figure F01"; empty for legacy M*/A* rows


def thesis_root(processed_dir: Path) -> Path:
    return processed_dir / "thesis_exhibits"


def _main_tables_dir(processed_dir: Path) -> Path:
    return thesis_root(processed_dir) / "main" / "tables"


def _main_figures_dir(processed_dir: Path) -> Path:
    return thesis_root(processed_dir) / "main" / "figures"


def _appendix_tables_dir(processed_dir: Path) -> Path:
    return thesis_root(processed_dir) / "appendix" / "tables"


def _appendix_figures_dir(processed_dir: Path) -> Path:
    return thesis_root(processed_dir) / "appendix" / "figures"


def _manifest_dir(processed_dir: Path) -> Path:
    return thesis_root(processed_dir) / "manifest"


def _fmt_num(x: Any, *, decimals: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return f"{v:.{decimals}f}"


def _fmt_p(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if v < 1e-4:
        return f"{v:.1e}"
    return f"{v:.4f}"


def _fmt_p_thesis_display(x: Any) -> str:
    """Thesis appendix C.2/C.3: no scientific notation; cap at '< 0.001'."""
    try:
        p = float(x)
    except (TypeError, ValueError):
        return ""
    if p != p:  # NaN
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def _validate_exact_set(observed: set[str], expected: set[str], *, label: str) -> None:
    if observed != expected:
        extra = sorted(observed - expected)
        miss = sorted(expected - observed)
        raise ValueError(f"{label}: expected={sorted(expected)} missing={miss} extra={extra}")


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_latex(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=False, na_rep="", longtable=False)
    out_path.write_text(tex, encoding="utf-8")


def _t03_exclusion_years_phrase(years: tuple[int, ...]) -> str:
    ys = sorted({int(y) for y in years})
    if not ys:
        return "the configured excluded calendar years"
    if len(ys) == 1:
        return f"calendar year {ys[0]}"
    if len(ys) == 2:
        return f"calendar years {ys[0]} and {ys[1]}"
    return "calendar years " + ", ".join(str(y) for y in ys[:-1]) + f", and {ys[-1]}"


def _t03_drop_years_phrase(years: tuple[int, ...]) -> str:
    """Agreement for 'dropping that year' vs 'those years'."""
    ys = {int(y) for y in years}
    if len(ys) <= 1:
        return "dropping that calendar year"
    return "dropping those calendar years"


# --- Thesis display tables for formal hypotheses (Tables C.2 / C.3); raw copy is OLD_Table_T05_... ---

T05_SAMPLE_DISPLAY = {
    "test": "Full test",
    "test_excl_2020": "Test excluding 2020",
    "test_excl_stress": "Test excluding stress window",
    "test_excl_union": "Test excluding 2020 and stress window",
}

T05_PROC_THESIS = {
    "mz_gnn": "Mincer–Zarnowitz",
    "dm_loss_diff": "Diebold–Mariano",
    "incremental_gnn_vix": "Incremental regression",
}

T05_DISPLAY_COLUMNS = (
    "Hypothesis",
    "Sample",
    "Procedure",
    "Loss / coefficient",
    "Statistic",
    "Primary p-value",
    "Inference",
)


def _t05_display_sample(sample: str) -> str:
    return T05_SAMPLE_DISPLAY.get(str(sample).strip(), str(sample))


def _t05_display_procedure(inference_procedure: str) -> str:
    return T05_PROC_THESIS.get(str(inference_procedure).strip(), str(inference_procedure))


def _t05_loss_or_coefficient(row: pd.Series) -> str:
    proc = str(row.get("inference_procedure", ""))
    loss = str(row.get("loss_name", "") or "").strip()
    if proc == "mz_gnn":
        return "GNN slope"
    if proc == "incremental_gnn_vix":
        return "GNN slope conditional on VIX"
    if proc == "dm_loss_diff":
        if loss == "mse_log":
            return "MSE(log)"
        if loss == "qlike":
            return "QLIKE"
        return loss if loss else "—"
    return "—"


def _t05_statistic_cell(row: pd.Series) -> str:
    try:
        v = float(row["statistic"])
    except (TypeError, ValueError):
        return ""
    if not (v == v):  # NaN
        return ""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _t05_primary_p_cell(row: pd.Series) -> str:
    return _fmt_p_thesis_display(row.get("p_value_primary"))


def _t05_inference_cell(row: pd.Series) -> str:
    """Academic phrasing for thesis Tables C.2/C.3."""
    proc = str(row.get("inference_procedure", ""))
    try:
        p = float(row.get("p_value_primary"))
    except (TypeError, ValueError):
        return ""
    if p != p:  # NaN
        return ""
    try:
        stat = float(row.get("statistic", 0.0))
    except (TypeError, ValueError):
        stat = 0.0
    loss = str(row.get("loss_name", "") or "").strip()

    if proc == "mz_gnn":
        if p < 0.001:
            return "Very strong evidence in favor"
        if p < 0.05:
            return "Evidence in favor at the 5% level"
        if p < 0.10:
            return "Borderline evidence"
        return "No clear evidence in favor"
    if proc == "incremental_gnn_vix":
        if p < 0.001:
            return "Very strong evidence in favor"
        if p < 0.05:
            return "Evidence of incremental information at the 5% level"
        if p < 0.10:
            return "Borderline evidence"
        return "No clear evidence in favor"
    if proc == "dm_loss_diff":
        if p < 0.01:
            return "Very strong evidence in favor"
        if p < 0.05:
            if loss == "mse_log":
                return "Evidence in favor on the log-scale loss criterion"
            if loss == "qlike":
                return "Evidence in favor on the QLIKE loss criterion"
            return "Evidence in favor on this loss criterion"
        if p < 0.10:
            return "Borderline evidence"
        if loss == "qlike" and stat < 0:
            return "No evidence in favor on the primary level-loss criterion"
        return "No clear evidence in favor"
    return ""


def _display_formal_hypothesis_frame(hyp: pd.DataFrame) -> pd.DataFrame:
    required = (
        "hypothesis_id",
        "inference_procedure",
        "sample",
        "statistic",
        "p_value_primary",
    )
    _require_columns(hyp, required, name="hypothesis display")
    rows: list[dict[str, str]] = []
    for _, row in hyp.iterrows():
        rows.append(
            {
                "Hypothesis": str(row["hypothesis_id"]),
                "Sample": _t05_display_sample(str(row["sample"])),
                "Procedure": _t05_display_procedure(str(row["inference_procedure"])),
                "Loss / coefficient": _t05_loss_or_coefficient(row),
                "Statistic": _t05_statistic_cell(row),
                "Primary p-value": _t05_primary_p_cell(row),
                "Inference": _t05_inference_cell(row),
            }
        )
    return pd.DataFrame(rows, columns=list(T05_DISPLAY_COLUMNS))


def _t05_c02_candidates(hyp: pd.DataFrame) -> pd.DataFrame:
    sub = hyp.loc[
        (hyp["hypothesis_id"].astype(str).isin(("H1", "H2", "H3")))
        & (hyp["sample"].astype(str) == "test")
    ].copy()
    # Order: H1, H2 (QLIKE then MSE(log)), H3
    pieces: list[pd.DataFrame] = []
    h1 = sub.loc[sub["hypothesis_id"].astype(str) == "H1"]
    pieces.append(h1)
    h2 = sub.loc[sub["hypothesis_id"].astype(str) == "H2"].copy()
    if not h2.empty:
        h2["_loss_order"] = h2["loss_name"].astype(str).map({"qlike": 0, "mse_log": 1}).fillna(99)
        h2 = h2.sort_values("_loss_order").drop(columns=["_loss_order"])
    pieces.append(h2)
    h3 = sub.loc[sub["hypothesis_id"].astype(str) == "H3"]
    pieces.append(h3)
    out = pd.concat(pieces, axis=0)
    return out


def _t05_c03_candidates(hyp: pd.DataFrame) -> pd.DataFrame:
    sub = hyp.loc[hyp["sample"].astype(str) != "test"].copy()
    sample_rank = {
        "test_excl_2020": 0,
        "test_excl_stress": 1,
        "test_excl_union": 2,
    }
    sub["_srank"] = sub["sample"].astype(str).map(sample_rank).fillna(99)
    sub["_loss_order"] = sub["loss_name"].astype(str).map({"qlike": 0, "mse_log": 1}).fillna(99)
    proc_rank = {"dm_loss_diff": 0, "incremental_gnn_vix": 1, "mz_gnn": 2}
    sub["_prank"] = sub["inference_procedure"].astype(str).map(proc_rank).fillna(99)
    sub = sub.sort_values(["_srank", "_prank", "_loss_order"]).drop(
        columns=["_srank", "_loss_order", "_prank"]
    )
    return sub


def _display_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        out["model"] = out["model"].astype(str).map(lambda x: MODEL_LABEL.get(x, x))
    if "inference_procedure" in out.columns:
        out["inference_procedure"] = out["inference_procedure"].astype(str).map(
            lambda x: PROC_LABEL.get(x, x)
        )
    return out


# Wide calibration tables (Appendix D.1 / D.2); model row order: GNN, VIX, HAR, Placebo
_THESIS_CALIB_MODEL_ORDER = ("gnn", "vix", "har", "placebo")
_THESIS_CALIB_VALUE_COLUMNS = (
    ("forecast_variance_log", "Forecast variance in log space"),
    ("corr_with_target_log", "Correlation with realized log-volatility"),
    ("mz_intercept", "MZ intercept"),
    ("mz_slope", "MZ slope"),
)


def _diagnostics_calibration_long_subset(diag: pd.DataFrame) -> pd.DataFrame:
    """Long-format rows used for OLD A6/T09 and thesis-wide D01/D02 calibration tables."""
    samples = {"full_test", "excl_2020"}
    diagnostics = {"forecast_variance_log", "corr_with_target_log", "mz_intercept", "mz_slope"}
    sub = diag.loc[
        (diag["sample"].astype(str).isin(samples))
        & (diag["diagnostic"].astype(str).isin(diagnostics))
        & (diag["model"].astype(str).isin(MODEL_ORDER))
    ].copy()
    expected_rows = len(samples) * len(diagnostics) * len(MODEL_ORDER)
    if len(sub) != expected_rows:
        raise ValueError(f"calibration diagnostics: expected {expected_rows} rows, got {len(sub)}")
    if sub.duplicated(subset=["sample", "diagnostic", "model"]).any():
        raise ValueError("calibration diagnostics: duplicate rows for (sample, diagnostic, model)")
    return sub


def _calibration_wide_display_for_sample(diag: pd.DataFrame, sample_key: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for m in _THESIS_CALIB_MODEL_ORDER:
        row: dict[str, str] = {"Model": MODEL_LABEL.get(m, m.upper())}
        for dkey, dcol in _THESIS_CALIB_VALUE_COLUMNS:
            sub = diag.loc[
                (diag["sample"].astype(str) == sample_key)
                & (diag["diagnostic"].astype(str) == dkey)
                & (diag["model"].astype(str) == m)
            ]
            if sub.empty:
                raise ValueError(f"calibration wide: missing row for {sample_key=}, {dkey=}, {m=}")
            val = float(sub.iloc[0]["value"])
            row[dcol] = f"{val:.4f}"
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=["Model"] + [h for _, h in _THESIS_CALIB_VALUE_COLUMNS],
    )


def _b01_yes_no(val: bool) -> str:
    return "Yes" if val else "No"


def _b01_training_loss_label(loss_id: str) -> str:
    lid = str(loss_id).strip().lower()
    return {
        "mse_log": "Mean squared error (log target)",
        "mae_log": "Mean absolute error (log target)",
        "smooth_l1_log": "Smooth L1 (log target)",
        "qlike_level": "QLIKE (level)",
        "hybrid_mse_qlike": "Hybrid MSE and QLIKE",
    }.get(lid, lid.replace("_", " ").title())


def _b01_formal_benchmark_losses(losses: tuple[str, ...]) -> str:
    parts: list[str] = []
    for x in losses:
        s = str(x).strip().lower()
        if s == "qlike":
            parts.append("QLIKE")
        elif s == "mse_log":
            parts.append("MSE (log target)")
        else:
            parts.append(s.replace("_", " ").upper())
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _b01_model_type_label(model_type: str) -> str:
    t = str(model_type).strip().lower()
    return {"graphsage": "GraphSAGE", "gat": "Graph attention (GAT)"}.get(t, t.replace("_", " ").title())


def _b01_gnn_depth_setting(model_type: str) -> str:
    t = str(model_type).strip().lower()
    if t == "graphsage":
        return "Number of GraphSAGE layers"
    if t == "gat":
        return "Number of GAT layers"
    return "Number of message-passing layers"


def _b01_pooling_label(pooling: str) -> str:
    p = str(pooling).strip().lower()
    return {"mean": "Mean", "max": "Max", "sum": "Sum", "add": "Add"}.get(p, p.title())


def _b01_scaler_label(scaler_type: str) -> str:
    st = str(scaler_type).strip().lower()
    if st == "standard":
        return "Train-only standardization (zero mean and unit variance)"
    return st.replace("_", " ").title()


def _b01_univ_mode_label(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "fixed_replace":
        return "Fixed replace (seed cohort then attrition replacement)"
    if m == "monthly_rebalance":
        return "Monthly rebalance"
    return str(mode).replace("_", " ").title()


def _b01_selection_rule(rule: str) -> str:
    r = str(rule).strip().lower()
    if r == "mcap":
        return "Market capitalization"
    return str(rule).replace("_", " ").title()


def _b01_dependence_label(dep: str) -> str:
    d = str(dep).strip().lower()
    if d == "spearman":
        return "Spearman rank correlation"
    return "Pearson correlation"


def _b01_node_features_phrase(gc) -> str:
    parts: list[str] = []
    if gc.rolling_mean:
        parts.append("rolling mean return")
    if gc.rolling_vol:
        parts.append("rolling volatility of return")
    if len(parts) == 2:
        return "Rolling mean return; rolling volatility of return"
    if len(parts) == 1:
        return parts[0].capitalize()
    return "—"


def _b01_eval_splits_phrase(splits: tuple[str, ...]) -> str:
    label = {"train": "Training", "val": "Validation", "test": "Test"}
    return "; ".join(label.get(str(s).strip().lower(), str(s).title()) for s in splits)


def _b01_h4_years_phrase(years: tuple[int, ...]) -> str:
    ys = [str(int(y)) for y in years]
    if not ys:
        return "—"
    if len(ys) == 1:
        return ys[0]
    return ", ".join(ys)


def _b01_rebalance_label(freq: str) -> str:
    return str(freq).strip().title()


def _build_appendix_a_input_schema_tables(
    processed_dir: Path, ts: str
) -> tuple[ExhibitRecord, ExhibitRecord, ExhibitRecord]:
    """Appendix A.1–A.3: raw-input overview and minimum schemas (thesis-facing; static).

    Grounded in ``docs/data_contracts.md`` ingest/returns_panel contracts and README
    raw-file layout (``WRDS.csv``, ``snp500_volatility.csv``, ``VIX.csv``).
    """
    tdir = _appendix_tables_dir(processed_dir)

    a01 = pd.DataFrame(
        [
            {
                "input_file": "WRDS.csv",
                "source": "CRSP daily security-level data via WRDS (or equivalent vendor extract).",
                "role_in_pipeline": (
                    "Equity panel input: daily stock returns and identifiers for the cross-section, "
                    "universe construction, rolling window features, and correlation-based graph edges."
                ),
                "frequency": "Daily (U.S. equity trading days).",
                "notes": "Security-by-date panel with CRSP-style identifiers and return fields; not used for the VIX benchmark.",
            },
            {
                "input_file": "snp500_volatility.csv",
                "source": "Public index series for the S&P 500 cash index (vendor of record as obtained by the researcher).",
                "role_in_pipeline": "Target construction input: daily index returns for forward-looking realized volatility on the aggregate market.",
                "frequency": "Daily.",
                "notes": "Calendar dates must align with the evaluation spine so targets and benchmarks are comparable.",
            },
            {
                "input_file": "VIX.csv",
                "source": "CBOE VIX (e.g., FRED or CBOE release), or equivalent.",
                "role_in_pipeline": (
                    "Benchmark input only: observed market-implied volatility joined for out-of-sample "
                    "comparison; excluded from model features by design."
                ),
                "frequency": "Daily.",
                "notes": "Used in evaluation and reporting, not as a predictor in the graph model.",
            },
        ],
        columns=["input_file", "source", "role_in_pipeline", "frequency", "notes"],
    )

    a02 = pd.DataFrame(
        [
            {
                "field_name": "date",
                "required": "Yes",
                "description": "Trading calendar date for each security observation.",
                "used_for": "Panel identity; time alignment across inputs.",
            },
            {
                "field_name": "permno",
                "required": "Yes",
                "description": "CRSP permanent security identifier.",
                "used_for": "Panel identity; cross-sectional linking.",
            },
            {
                "field_name": "permco",
                "required": "Yes",
                "description": "CRSP permanent company identifier.",
                "used_for": "Firm-level identifier consistency.",
            },
            {
                "field_name": "shrcd",
                "required": "Yes",
                "description": "CRSP share code classifying the security type.",
                "used_for": "Universe maintenance (ordinary common shares).",
            },
            {
                "field_name": "exchcd",
                "required": "Yes",
                "description": "CRSP exchange code for the listing venue.",
                "used_for": "Universe maintenance and listing filters.",
            },
            {
                "field_name": "ret",
                "required": "Yes",
                "description": "Daily total return including distributions.",
                "used_for": "Return construction; dividend-inclusive comparison series.",
            },
            {
                "field_name": "retx",
                "required": "Yes",
                "description": "Daily return excluding distributions.",
                "used_for": (
                    "Return construction; principal return series for rolling features and dependence in the canonical pipeline."
                ),
            },
            {
                "field_name": "prc",
                "required": "Yes",
                "description": "CRSP price field (closing or bid–ask composite per vendor convention).",
                "used_for": "Market-capitalization ranking with shares outstanding.",
            },
            {
                "field_name": "vol",
                "required": "Yes",
                "description": "Daily trading volume.",
                "used_for": "Auxiliary trading-activity information in the equity panel.",
            },
            {
                "field_name": "shrout",
                "required": "Yes",
                "description": "Shares outstanding (CRSP units).",
                "used_for": "Market-capitalization ranking for universe selection.",
            },
            {
                "field_name": "dlstcd",
                "required": "Yes",
                "description": "CRSP delisting code.",
                "used_for": "Delisting and survivorship coding.",
            },
            {
                "field_name": "dlret",
                "required": "Yes",
                "description": "Delisting-period return.",
                "used_for": "Delisting return adjustment.",
            },
            {
                "field_name": "dlretx",
                "required": "Yes",
                "description": "Delisting return excluding distributions.",
                "used_for": "Delisting return adjustment.",
            },
            {
                "field_name": "dlprc",
                "required": "Yes",
                "description": "Delisting price.",
                "used_for": "Delisting fields supporting return reconstruction.",
            },
            {
                "field_name": "dlpdt",
                "required": "Yes",
                "description": "Last quoted price date before delisting.",
                "used_for": "Delisting calendar alignment.",
            },
            {
                "field_name": "ticker",
                "required": "Yes",
                "description": "Listed ticker symbol.",
                "used_for": "Security identification (auxiliary).",
            },
            {
                "field_name": "comnam",
                "required": "Yes",
                "description": "Company name.",
                "used_for": "Security identification (auxiliary).",
            },
            {
                "field_name": "cusip",
                "required": "Yes",
                "description": "CUSIP identifier.",
                "used_for": "Security identification (auxiliary).",
            },
            {
                "field_name": "ncusip",
                "required": "Yes",
                "description": "Numeric CUSIP.",
                "used_for": "Security identification (auxiliary).",
            },
            {
                "field_name": "secstat",
                "required": "Yes",
                "description": "Security trading status or header flag from the vendor file.",
                "used_for": "Security metadata retained from the extract.",
            },
        ],
        columns=["field_name", "required", "description", "used_for"],
    )

    a03 = pd.DataFrame(
        [
            {
                "input_file": "snp500_volatility.csv",
                "field_name": "date",
                "required": "Yes",
                "description": "Trading or observation date for the index series (may appear under standard vendor date headers).",
                "used_for": "Target construction; calendar merge to the modeling spine.",
            },
            {
                "input_file": "snp500_volatility.csv",
                "field_name": "sprtrn",
                "required": "Yes",
                "description": "Daily S&P 500 index total return.",
                "used_for": "Target construction; forward realized variance of the index.",
            },
            {
                "input_file": "VIX.csv",
                "field_name": "date",
                "required": "Yes",
                "description": "Observation date for the VIX quote (may appear under standard vendor date headers).",
                "used_for": "Benchmark alignment; calendar merge to forecasts and targets.",
            },
            {
                "input_file": "VIX.csv",
                "field_name": "vix",
                "required": "Yes",
                "description": "CBOE VIX closing level (vendors may label the series VIXCLS).",
                "used_for": "Benchmark input; out-of-sample comparison only.",
            },
        ],
        columns=["input_file", "field_name", "required", "description", "used_for"],
    )

    p01 = tdir / "Table_A01_data_inputs_and_roles.csv"
    p02 = tdir / "Table_A02_wrds_crsp_raw_input_schema.csv"
    p03 = tdir / "Table_A03_required_target_and_benchmark_schema.csv"
    _write_csv(a01, p01)
    _write_csv(a02, p02)
    _write_csv(a03, p03)

    doc_ref = "docs/data_contracts.md (ingest, returns_panel, targets)"
    r1 = ExhibitRecord(
        exhibit_id="A01",
        title="Table A.1. Data inputs and their roles in the canonical thesis pipeline.",
        section="appendix",
        output_paths=(str(p01),),
        source_paths=(),
        filter_logic=f"Static overview; {doc_ref}",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table A.1",
    )
    r2 = ExhibitRecord(
        exhibit_id="A02",
        title="Table A.2. Minimum required WRDS/CRSP raw-input schema for reproducing the empirical pipeline.",
        section="appendix",
        output_paths=(str(p02),),
        source_paths=(),
        filter_logic=f"Static schema; {doc_ref}",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table A.2",
    )
    r3 = ExhibitRecord(
        exhibit_id="A03",
        title=(
            "Table A.3. Minimum required target and benchmark raw-input schema for reproducing "
            "the empirical pipeline."
        ),
        section="appendix",
        output_paths=(str(p03),),
        source_paths=(),
        filter_logic=f"Static schema; {doc_ref}",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table A.3",
    )
    return r1, r2, r3


def _build_t02(cfg: ResolvedConfig, processed_dir: Path, ts: str) -> ExhibitRecord:
    cp = cfg.source_config_path
    gc = load_graph_config(cp)
    ds = load_dataset_config(cp)
    mt = load_model_train_config(cp)
    me = load_model_evaluate_config(cp)

    rows: list[tuple[str, str, str]] = [
        # Graph construction
        (
            "Graph construction",
            "Historical window length",
            f"{gc.window_length} trading days",
        ),
        ("Graph construction", "Minimum observation fraction", str(gc.min_obs_frac)),
        ("Graph construction", "Universe mode", _b01_univ_mode_label(gc.universe_mode)),
        ("Graph construction", "Number of stocks", str(gc.n_nodes)),
        ("Graph construction", "Selection rule", _b01_selection_rule(gc.selection_rule)),
        ("Graph construction", "Rebalance frequency", _b01_rebalance_label(gc.rebalance_freq)),
        ("Graph construction", "Dependence measure", _b01_dependence_label(gc.dependence)),
        ("Graph construction", "Top-k neighbors", str(gc.top_k)),
        ("Graph construction", "Graph symmetrization", _b01_yes_no(gc.symmetrize)),
        ("Graph construction", "Node features", _b01_node_features_phrase(gc)),
        ("Graph construction", "Forecast target", gc.target_column),
        # Dataset and splits
        ("Dataset and splits", "Training sample end", ds.train_end),
        ("Dataset and splits", "Validation sample end", ds.val_end),
        ("Dataset and splits", "Test sample end", ds.test_end),
        ("Dataset and splits", "Feature scaler", _b01_scaler_label(ds.scaler_type)),
        # Model architecture
        ("Model architecture", "Model type", _b01_model_type_label(mt.model_type)),
        ("Model architecture", _b01_gnn_depth_setting(mt.model_type), str(mt.num_gnn_layers)),
        ("Model architecture", "Hidden dimension", str(mt.hidden_channels)),
        ("Model architecture", "Pooling method", _b01_pooling_label(mt.pooling)),
        # Training
        ("Training", "Random seed", str(mt.seed)),
        ("Training", "Maximum epochs", str(mt.max_epochs)),
        ("Training", "Learning rate", f"{mt.lr:.10f}".rstrip("0").rstrip(".") if mt.lr < 1e-2 else str(mt.lr)),
        ("Training", "Early stopping patience", str(mt.early_stopping_patience)),
        ("Training", "Batch size", str(mt.batch_size)),
        ("Training", "Loss function", _b01_training_loss_label(mt.loss)),
        # Evaluation and robustness
        ("Evaluation and robustness", "Evaluation splits", _b01_eval_splits_phrase(me.splits)),
        ("Evaluation and robustness", "H4 exclusion year(s)", _b01_h4_years_phrase(me.summary_test_exclusion_years)),
        (
            "Evaluation and robustness",
            "H4 stress window",
            f"{me.summary_test_stress_excl_start}–{me.summary_test_stress_excl_end}",
        ),
        (
            "Evaluation and robustness",
            "Loss functions used in formal benchmark comparison",
            _b01_formal_benchmark_losses(me.hypothesis_dm_losses),
        ),
        ("Evaluation and robustness", "Placebo ablation enabled", _b01_yes_no(me.enable_placebo_ablation)),
        ("Evaluation and robustness", "Placebo edge seed", str(me.placebo_edge_seed)),
    ]

    df = pd.DataFrame(rows, columns=["component", "setting", "canonical_value"])
    out_csv = _appendix_tables_dir(processed_dir) / "Table_B01_canonical_configuration.csv"
    _write_csv(df, out_csv)
    return ExhibitRecord(
        exhibit_id="B01",
        title="Table B.1. Canonical empirical specification (configuration summary).",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(cp),),
        filter_logic="Reader-facing summary from [graph], [dataset], [model.train], [model.evaluate]",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table B.1",
    )


def _build_t03(cfg: ResolvedConfig, processed_dir: Path, ts: str) -> ExhibitRecord:
    me = load_model_evaluate_config(cfg.source_config_path)
    ds = load_dataset_config(cfg.source_config_path)

    years_phrase = _t03_exclusion_years_phrase(me.summary_test_exclusion_years)
    drop_phrase = _t03_drop_years_phrase(me.summary_test_exclusion_years)
    stress_start = str(me.summary_test_stress_excl_start).strip()
    stress_end = str(me.summary_test_stress_excl_end).strip()

    row_tuples: list[tuple[str, str, str]] = [
        (
            "Full test",
            "Baseline hold-out for formal tests",
            (
                "All trading days assigned to the test split in the canonical split definition. "
                f"The configured end date for the test period is {ds.test_end}."
            ),
        ),
        (
            "Test excluding 2020",
            "H4 robustness (calendar-year exclusion)",
            (
                f"Test days excluding {years_phrase}. "
                f"Inference on this subsample assesses sensitivity to {drop_phrase}."
            ),
        ),
        (
            "Test excluding stress window",
            "H4 robustness (stress-window exclusion)",
            (
                f"Test days outside the inclusive stress window from {stress_start} through {stress_end}. "
                "Inference on this subsample assesses sensitivity to excluding that episode."
            ),
        ),
        (
            "Test excluding 2020 and stress window",
            "H4 robustness (combined exclusions)",
            (
                f"Test days remaining after excluding both {years_phrase} and the stress window "
                f"from {stress_start} through {stress_end}. "
                "This applies both restrictions together on the hold-out test split."
            ),
        ),
    ]

    df = pd.DataFrame(
        row_tuples,
        columns=["Subsample", "Role in formal evaluation", "Definition"],
    )
    out_csv = (
        _appendix_tables_dir(processed_dir)
        / "Table_C01_formal_test_subsamples_used_in_robustness_analysis.csv"
    )

    _write_csv(df, out_csv)
    return ExhibitRecord(
        exhibit_id="C01",
        title="Table C.1. Formal test subsamples used in robustness analysis (H4)",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(cfg.source_config_path),),
        filter_logic="reader-facing subsample definitions from ModelEvaluateConfig + dataset.test_end",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table C.1",
    )


def _build_t04(processed_dir: Path, ts: str) -> ExhibitRecord:
    cols = (
        "hypothesis",
        "empirical_question",
        "procedure",
        "key_comparison_or_statistic",
        "evaluation_sample",
    )
    rows = [
        (
            "H1",
            "Whether the GNN forecast contains predictive information about future "
            "30-trading-day realized market volatility.",
            "Mincer–Zarnowitz predictive regression",
            "Significance of the GNN slope coefficient",
            "full test sample",
        ),
        (
            "H2",
            "Whether the GNN achieves lower out-of-sample forecast loss than the VIX benchmark.",
            "Diebold–Mariano-type loss comparison",
            "QLIKE and MSE (log target) loss differential",
            "full test sample",
        ),
        (
            "H3",
            "Whether the GNN adds predictive information beyond the VIX benchmark.",
            "Incremental predictive regression",
            "Significance of the GNN coefficient conditional on VIX",
            "full test sample",
        ),
        (
            "H4",
            "Whether the GNN retains its relative forecasting advantage once pre-specified "
            "extreme periods are excluded.",
            "Restricted-subsample Diebold–Mariano and incremental regressions",
            "Same H2 and H3 comparisons repeated on restricted subsamples",
            "Test excluding 2020; test excluding stress window; test excluding both.",
        ),
    ]
    df = pd.DataFrame(rows, columns=list(cols))
    out_csv = _appendix_tables_dir(processed_dir) / "Table_B02_hypothesis_to_procedure_artifact_map.csv"
    _write_csv(df, out_csv)
    return ExhibitRecord(
        exhibit_id="B02",
        title="Table B.2. Empirical hypotheses and evaluation procedures.",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(),
        filter_logic="Static thesis-facing map (no artifact paths).",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table B.2",
    )


def _quintile_log_forecast_error_frame(diag: pd.DataFrame, sample_key: str) -> pd.DataFrame:
    """MSE in log space by target-volatility quintile; thesis columns and row labels."""
    _require_columns(
        diag,
        ("sample", "diagnostic", "model", "value", "detail"),
        name="quintile diagnostics_smoothing.csv",
    )
    sub = diag.loc[
        (diag["sample"].astype(str) == sample_key)
        & (diag["diagnostic"].astype(str) == "mse_log_by_target_vol_quantile")
    ].copy()
    _validate_exact_set(
        set(sub["model"].astype(str)),
        set(MODEL_ORDER),
        label=f"quintile model set ({sample_key})",
    )
    _validate_exact_set(
        set(sub["detail"].astype(str)),
        set(QUINTILE_ORDER),
        label=f"quintile detail labels ({sample_key})",
    )
    if len(sub) != 20:
        raise ValueError(f"quintile table {sample_key}: expected 20 rows, got {len(sub)}")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    wide = sub.pivot_table(index="detail", columns="model", values="value", aggfunc="first")
    wide = wide.reindex(list(QUINTILE_ORDER))
    wide = wide[list(MODEL_ORDER)]
    wide = wide.rename(columns={m: MODEL_LABEL[m] for m in MODEL_ORDER})
    wide = wide.reset_index()
    wide["Target-volatility quintile"] = wide["detail"].astype(str).map(
        lambda x: QUINTILE_ROW_LABEL.get(x, str(x))
    )
    wide = wide.drop(columns=["detail"])
    value_cols = [MODEL_LABEL[m] for m in MODEL_ORDER]
    wide = wide[["Target-volatility quintile"] + value_cols]
    for c in value_cols:
        wide[c] = wide[c].map(lambda v: f"{float(v):.4f}")
    return wide


def _build_d03_quintile_table(diag: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    wide = _quintile_log_forecast_error_frame(diag, "full_test")
    out_csv = (
        _appendix_tables_dir(processed_dir) /
        "Table_D03_log_scale_forecast_error_by_target_volatility_quintile_full_test_sample.csv"
    )
    _write_csv(wide, out_csv)
    return ExhibitRecord(
        exhibit_id="D03",
        title="Table D.3. Log-scale forecast error by target-volatility quintile on the full test sample.",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(processed_dir / "summaries" / "diagnostics_smoothing.csv"),),
        filter_logic="MSE (log) by target-volatility quintile; full test",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table D.3",
    )


def _build_d04_quintile_table(diag: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    wide = _quintile_log_forecast_error_frame(diag, "excl_2020")
    out_csv = (
        _appendix_tables_dir(processed_dir) /
        "Table_D04_log_scale_forecast_error_by_target_volatility_quintile_test_sample_excluding_2020.csv"
    )
    _write_csv(wide, out_csv)
    return ExhibitRecord(
        exhibit_id="D04",
        title="Table D.4. Log-scale forecast error by target-volatility quintile on the test sample excluding 2020.",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(processed_dir / "summaries" / "diagnostics_smoothing.csv"),),
        filter_logic="MSE (log) by target-volatility quintile; test excluding configured years",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table D.4",
    )


def _write_reproducibility_metadata(cfg: ResolvedConfig, ts: str) -> ExhibitRecord:
    proc = Path(cfg.processed_dir)
    meta = {
        "schema": "cemspims.reproducibility_metadata.v1",
        "build_timestamp_utc": ts,
        "run_id": cfg.run_id,
        "project_root": str(cfg.project_root.resolve()),
        "config_path": str(cfg.source_config_path.resolve()),
        "interim_dir": str(cfg.interim_dir.resolve()),
        "processed_dir": str(cfg.processed_dir.resolve()),
        "pip_install_command": 'pip install -e ".[dev,train]"',
        "thesis_export_command": (
            f"python -m mss.cli run --run {cfg.run_id} "
            "--pipeline thesis.export --overwrite thesis.export"
        ),
    }
    out = _manifest_dir(proc) / "Reproducibility_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ExhibitRecord(
        exhibit_id="REPRO0",
        title="Reproducibility metadata (machine-readable)",
        section="appendix",
        output_paths=(str(out),),
        source_paths=(str(cfg.source_config_path),),
        filter_logic="paths and commands only; expand narrative in thesis manually",
        output_formats=("json",),
        build_timestamp_utc=ts,
        thesis_label="",
    )


def _write_thesis_table_index(processed_dir: Path, ts: str) -> None:
    lines = [
        "# Thesis export numbering index",
        "",
        f"Built: `{ts}` (UTC). Appendix **tables** are **CSV only** under `appendix/tables/`. "
        "Thesis-facing names use `Table_A01_…`, `Table_B01_…`, `Table_C01_…`, `Table_D01_…`. "
        "Raw/pipeline duplicates use the `OLD_` prefix. Main-chapter tables may still include `.tex` under `main/tables/`. "
        "Cite **Figure F01** for the thesis figure filename.",
        "",
        "| thesis_label | exhibit_id | appendix/tables CSV (thesis-facing) |",
        "|---|---|---|",
        "| Table A.1 | A01 | Table_A01_data_inputs_and_roles.csv |",
        "| Table A.2 | A02 | Table_A02_wrds_crsp_raw_input_schema.csv |",
        "| Table A.3 | A03 | Table_A03_required_target_and_benchmark_schema.csv |",
        "| Table B.1 | B01 | Table_B01_canonical_configuration.csv |",
        "| Table B.2 | B02 | Table_B02_hypothesis_to_procedure_artifact_map.csv |",
        "| Table C.1 | C01 | Table_C01_formal_test_subsamples_used_in_robustness_analysis.csv |",
        "| Table C.2 | C02 | Table_C02_full_test_formal_hypothesis_results.csv |",
        "| Table C.3 | C03 | Table_C03_restricted_subsample_robustness_results.csv |",
        "| Table D.1 | D01 | Table_D01_calibration_diagnostics_full_test_sample.csv |",
        "| Table D.2 | D02 | Table_D02_calibration_diagnostics_test_sample_excluding_2020.csv |",
        "| Table D.3 | D03 | Table_D03_log_scale_forecast_error_by_target_volatility_quintile_full_test_sample.csv |",
        "| Table D.4 | D04 | Table_D04_log_scale_forecast_error_by_target_volatility_quintile_test_sample_excluding_2020.csv |",
        "",
        "| Role | exhibit_id | appendix/tables CSV (OLD / archival) |",
        "|---|---|---|",
        "| OLD copy | A1 | OLD_A1_full_hypothesis_output.csv |",
        "| OLD copy | T05 | OLD_Table_T05_full_formal_hypothesis_output.csv |",
        "| OLD copy | A2 | OLD_A2_incremental_regression.csv |",
        "| OLD copy | T06 | OLD_Table_T06_incremental_regression_gnn_conditional_on_vix.csv |",
        "| OLD copy | A3 | OLD_A3_mz_regression.csv |",
        "| OLD copy | T07 | OLD_Table_T07_mincer_zarnowitz_regression_gnn.csv |",
        "| OLD copy | A5 | OLD_A5_universe_churn_summary_full.csv, OLD_A5_universe_churn_summary_compact.csv |",
        "| OLD copy | T08 | OLD_Table_T08_universe_churn_summary_full.csv, OLD_Table_T08_universe_churn_summary_compact.csv |",
        "| OLD copy | A6 | OLD_A6_compact_calibration_stats.csv |",
        "| OLD copy | T09 | OLD_Table_T09_compact_calibration_statistics.csv |",
        "",
        "| thesis_label | exhibit_id | path under thesis_exhibits/ |",
        "|---|---|---|",
        "| Figure F01 | F01 | appendix/figures/Figure_F01_universe_diagnostics_plate.* |",
        "| (metadata) | REPRO0 | manifest/Reproducibility_metadata.json |",
        "",
    ]
    p = _manifest_dir(processed_dir) / "thesis_table_index.md"
    p.write_text("\n".join(lines), encoding="utf-8")


def _build_m1(summary: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    _require_columns(
        summary,
        ("model", "sample", "mse_log", "mae_log", "qlike", "n_samples"),
        name="M1 summary_table.csv",
    )
    sub = summary.loc[summary["sample"].astype(str) == "full_test"].copy()
    obs_models = set(sub["model"].astype(str).tolist())
    _validate_exact_set(obs_models, set(MODEL_ORDER), label="M1 model set")
    if sub.duplicated(subset=["model"]).any():
        raise ValueError("M1: duplicate rows for model in full_test sample")
    if len(sub) != 4:
        raise ValueError(f"M1: expected 4 rows, got {len(sub)}")
    sub = sub.set_index("model").loc[list(MODEL_ORDER)].reset_index()
    for c in ("mse_log", "mae_log", "qlike"):
        if pd.to_numeric(sub[c], errors="coerce").isna().any():
            raise ValueError(f"M1: non-numeric values in {c}")

    out_csv = _main_tables_dir(processed_dir) / "M1_oos_fulltest_accuracy.csv"
    out_tex = _main_tables_dir(processed_dir) / "M1_oos_fulltest_accuracy.tex"
    _write_csv(sub[["model", "mse_log", "mae_log", "qlike", "n_samples"]], out_csv)

    disp = _display_labels(sub[["model", "mse_log", "mae_log", "qlike", "n_samples"]]).copy()
    disp["mse_log"] = disp["mse_log"].map(_fmt_num)
    disp["mae_log"] = disp["mae_log"].map(_fmt_num)
    disp["qlike"] = disp["qlike"].map(_fmt_num)
    disp["n_samples"] = disp["n_samples"].astype(int).astype(str)
    _write_latex(disp, out_tex)

    return ExhibitRecord(
        exhibit_id="M1",
        title="Out-of-Sample Forecast Accuracy on Full Test Sample",
        section="main",
        output_paths=(str(out_csv), str(out_tex)),
        source_paths=(str(processed_dir / "summaries" / "summary_table.csv"),),
        filter_logic="sample == full_test; model in {gnn,vix,har,placebo}",
        output_formats=("csv", "tex"),
        build_timestamp_utc=ts,
    )


def _build_m2(processed_dir: Path, ts: str) -> ExhibitRecord:
    src = processed_dir / "figures" / "target_vs_model_vix" / "target_vs_model_vix_test.png"
    if (not src.is_file()) or src.stat().st_size <= 0:
        raise FileNotFoundError(f"M2 source figure missing/empty: {src}")
    # Decode test to ensure file is readable.
    _ = plt.imread(src)
    out = _main_figures_dir(processed_dir) / "M2_test_forecast_paths.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    return ExhibitRecord(
        exhibit_id="M2",
        title="Test-Period Forecast Paths: Target vs GNN vs VIX",
        section="main",
        output_paths=(str(out),),
        source_paths=(str(src),),
        filter_logic="direct-use existing test split figure",
        output_formats=("png",),
        build_timestamp_utc=ts,
    )


def _build_m3(hyp: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    _require_columns(
        hyp,
        (
            "hypothesis_id",
            "inference_procedure",
            "loss_name",
            "statistic",
            "p_value_primary",
            "tail",
            "n_obs",
            "hac_max_lags",
            "sample",
        ),
        name="M3 hypothesis_tests.csv",
    )
    sub = hyp.loc[
        (hyp["sample"].astype(str) == "test")
        & (hyp["hypothesis_id"].astype(str).isin(["H1", "H2", "H3"]))
    ].copy()
    if len(sub) != 4:
        raise ValueError(f"M3: expected 4 rows (H1, H2x2, H3), got {len(sub)}")
    h2 = sub.loc[sub["hypothesis_id"].astype(str) == "H2"]
    _validate_exact_set(set(h2["loss_name"].astype(str)), {"qlike", "mse_log"}, label="M3 H2 losses")
    key_dupes = sub.duplicated(subset=["hypothesis_id", "inference_procedure", "loss_name"])
    if key_dupes.any():
        raise ValueError("M3: duplicate rows for (hypothesis_id, inference_procedure, loss_name)")

    order_key = {"H1": 0, "H2": 1, "H3": 2}
    sub["_o"] = sub["hypothesis_id"].astype(str).map(order_key)
    sub["_l"] = sub["loss_name"].astype(str).map({"qlike": 0, "mse_log": 1}).fillna(0)
    sub = sub.sort_values(["_o", "_l"]).drop(columns=["_o", "_l"])

    cols = [
        "hypothesis_id",
        "inference_procedure",
        "loss_name",
        "statistic",
        "p_value_primary",
        "tail",
        "n_obs",
        "hac_max_lags",
    ]
    out_csv = _main_tables_dir(processed_dir) / "M3_h1_h3_formal_tests_fulltest.csv"
    out_tex = _main_tables_dir(processed_dir) / "M3_h1_h3_formal_tests_fulltest.tex"
    _write_csv(sub[cols], out_csv)

    disp = _display_labels(sub[cols]).copy()
    disp["statistic"] = disp["statistic"].map(_fmt_num)
    disp["p_value_primary"] = disp["p_value_primary"].map(_fmt_p)
    disp["n_obs"] = disp["n_obs"].astype(int).astype(str)
    disp["hac_max_lags"] = disp["hac_max_lags"].astype(int).astype(str)
    _write_latex(disp, out_tex)

    return ExhibitRecord(
        exhibit_id="M3",
        title="Formal Hypothesis Tests on Full Test Sample (H1-H3)",
        section="main",
        output_paths=(str(out_csv), str(out_tex)),
        source_paths=(str(processed_dir / "summaries" / "hypothesis_tests.csv"),),
        filter_logic="sample == test; hypothesis_id in {H1,H2,H3}; include H2 losses {qlike,mse_log}",
        output_formats=("csv", "tex"),
        build_timestamp_utc=ts,
    )


def _build_m4(hyp: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    _require_columns(
        hyp,
        (
            "hypothesis_id",
            "sample",
            "inference_procedure",
            "loss_name",
            "statistic",
            "p_value_primary",
            "n_obs",
            "rows_removed_vs_full_test",
        ),
        name="M4 hypothesis_tests.csv",
    )
    samples = {"test_excl_2020", "test_excl_stress", "test_excl_union"}
    sub = hyp.loc[
        (hyp["hypothesis_id"].astype(str) == "H4")
        & (hyp["sample"].astype(str).isin(samples))
        & (hyp["inference_procedure"].astype(str).isin(["dm_loss_diff", "incremental_gnn_vix"]))
    ].copy()
    if len(sub) != 9:
        raise ValueError(f"M4: expected 9 rows, got {len(sub)}")
    _validate_exact_set(set(sub["sample"].astype(str)), samples, label="M4 sample set")

    dm = sub.loc[sub["inference_procedure"].astype(str) == "dm_loss_diff"]
    if len(dm) != 6:
        raise ValueError(f"M4: expected 6 DM rows, got {len(dm)}")
    for s in samples:
        lset = set(dm.loc[dm["sample"].astype(str) == s, "loss_name"].astype(str))
        _validate_exact_set(lset, {"qlike", "mse_log"}, label=f"M4 DM losses for {s}")
    inc = sub.loc[sub["inference_procedure"].astype(str) == "incremental_gnn_vix"]
    if len(inc) != 3:
        raise ValueError(f"M4: expected 3 incremental rows, got {len(inc)}")

    if sub.duplicated(subset=["sample", "inference_procedure", "loss_name"]).any():
        raise ValueError("M4: duplicate rows for (sample, inference_procedure, loss_name)")

    sample_order = {"test_excl_2020": 0, "test_excl_stress": 1, "test_excl_union": 2}
    proc_order = {"dm_loss_diff": 0, "incremental_gnn_vix": 1}
    loss_order = {"qlike": 0, "mse_log": 1, "": 2}
    sub["_s"] = sub["sample"].astype(str).map(sample_order)
    sub["_p"] = sub["inference_procedure"].astype(str).map(proc_order)
    sub["_l"] = sub["loss_name"].astype(str).map(loss_order).fillna(2)
    sub = sub.sort_values(["_s", "_p", "_l"]).drop(columns=["_s", "_p", "_l"])

    cols = [
        "sample",
        "inference_procedure",
        "loss_name",
        "statistic",
        "p_value_primary",
        "n_obs",
        "rows_removed_vs_full_test",
    ]
    out_csv = _main_tables_dir(processed_dir) / "M4_h4_robustness_tests.csv"
    out_tex = _main_tables_dir(processed_dir) / "M4_h4_robustness_tests.tex"
    _write_csv(sub[cols], out_csv)

    disp = _display_labels(sub[cols]).copy()
    disp["statistic"] = disp["statistic"].map(_fmt_num)
    disp["p_value_primary"] = disp["p_value_primary"].map(_fmt_p)
    disp["n_obs"] = disp["n_obs"].astype(int).astype(str)
    disp["rows_removed_vs_full_test"] = disp["rows_removed_vs_full_test"].astype(int).astype(str)
    _write_latex(disp, out_tex)

    return ExhibitRecord(
        exhibit_id="M4",
        title="Robustness to Excluding Extreme Episodes (H4)",
        section="main",
        output_paths=(str(out_csv), str(out_tex)),
        source_paths=(str(processed_dir / "summaries" / "hypothesis_tests.csv"),),
        filter_logic="H4 rows; samples in {test_excl_2020,test_excl_stress,test_excl_union}; procedures DM+incremental",
        output_formats=("csv", "tex"),
        build_timestamp_utc=ts,
    )


def _build_m5(diag: pd.DataFrame, processed_dir: Path, ts: str) -> ExhibitRecord:
    _require_columns(
        diag,
        ("sample", "diagnostic", "model", "value", "detail"),
        name="M5 diagnostics_smoothing.csv",
    )
    sub = diag.loc[
        (diag["sample"].astype(str) == "full_test")
        & (diag["diagnostic"].astype(str) == "mse_log_by_target_vol_quantile")
    ].copy()
    _validate_exact_set(set(sub["model"].astype(str)), set(MODEL_ORDER), label="M5 model set")
    _validate_exact_set(set(sub["detail"].astype(str)), set(QUINTILE_ORDER), label="M5 quintile labels")
    if len(sub) != 20:
        raise ValueError(f"M5: expected 20 rows, got {len(sub)}")
    if sub.duplicated(subset=["model", "detail"]).any():
        raise ValueError("M5: duplicate rows for (model, detail)")
    values = pd.to_numeric(sub["value"], errors="coerce")
    if values.isna().any():
        raise ValueError("M5: non-numeric or missing values in value column")
    sub = sub.copy()
    sub["value"] = values
    out_png = _main_figures_dir(processed_dir) / "M5_mse_log_by_vol_quintile.png"
    out_pdf = _main_figures_dir(processed_dir) / "M5_mse_log_by_vol_quintile.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = list(QUINTILE_ORDER)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for model in MODEL_ORDER:
        sm = sub.loc[sub["model"].astype(str) == model].copy()
        sm["detail"] = pd.Categorical(sm["detail"], categories=x, ordered=True)
        sm = sm.sort_values("detail")
        ax.plot(
            x,
            sm["value"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.set_title("MSE(log) by target-volatility quintile (full test)")
    ax.set_xlabel("Target-volatility quintile")
    ax.set_ylabel("MSE(log)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return ExhibitRecord(
        exhibit_id="M5",
        title="Forecast Error by Volatility Regime Quintile (Test Sample)",
        section="main",
        output_paths=(str(out_png), str(out_pdf)),
        source_paths=(str(processed_dir / "summaries" / "diagnostics_smoothing.csv"),),
        filter_logic="sample == full_test; diagnostic == mse_log_by_target_vol_quantile; model in {gnn,vix,har,placebo}",
        output_formats=("png", "pdf"),
        build_timestamp_utc=ts,
    )


def _build_a1(
    hyp: pd.DataFrame, processed_dir: Path, ts: str
) -> tuple[ExhibitRecord, ExhibitRecord, ExhibitRecord, ExhibitRecord]:
    src = processed_dir / "summaries" / "hypothesis_tests.csv"
    if not src.is_file():
        raise FileNotFoundError(f"A1 source missing: {src}")
    out = _appendix_tables_dir(processed_dir) / "OLD_A1_full_hypothesis_output.csv"
    out_t05 = _appendix_tables_dir(processed_dir) / "OLD_Table_T05_full_formal_hypothesis_output.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    shutil.copy2(src, out_t05)

    c02_src = _t05_c02_candidates(hyp)
    c03_src = _t05_c03_candidates(hyp)
    if c02_src.empty:
        raise ValueError("hypothesis display C.2: no H1–H3 full-test rows found")
    if c03_src.empty:
        raise ValueError("hypothesis display C.3: no restricted-subsample rows found")

    disp_c02 = _display_formal_hypothesis_frame(c02_src)
    disp_c03 = _display_formal_hypothesis_frame(c03_src)

    c02_csv = _appendix_tables_dir(processed_dir) / "Table_C02_full_test_formal_hypothesis_results.csv"
    c03_csv = (
        _appendix_tables_dir(processed_dir) / "Table_C03_restricted_subsample_robustness_results.csv"
    )

    _write_csv(disp_c02, c02_csv)
    _write_csv(disp_c03, c03_csv)

    r_a1 = ExhibitRecord(
        exhibit_id="A1",
        title="OLD: full hypothesis_tests.csv copy (archival audit trail)",
        section="appendix",
        output_paths=(str(out),),
        source_paths=(str(src),),
        filter_logic="verbatim copy of summaries/hypothesis_tests.csv",
        output_formats=("csv",),
        build_timestamp_utc=ts,
    )
    r_t05 = ExhibitRecord(
        exhibit_id="T05",
        title="OLD: verbatim hypothesis_tests.csv (archival; use Table_C02 / Table_C03 for thesis)",
        section="appendix",
        output_paths=(str(out_t05),),
        source_paths=(str(src),),
        filter_logic="verbatim copy of summaries/hypothesis_tests.csv; thesis display in Table_C02_*, Table_C03_*",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="",
    )
    r_c02 = ExhibitRecord(
        exhibit_id="C02",
        title="Table C.2. Full-test formal hypothesis results (H1–H3)",
        section="appendix",
        output_paths=(str(c02_csv),),
        source_paths=(str(src),),
        filter_logic="reader-facing columns; full test only; derived from hypothesis_tests.csv",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table C.2",
    )
    r_c03 = ExhibitRecord(
        exhibit_id="C03",
        title="Table C.3. Restricted-subsample robustness results (H4)",
        section="appendix",
        output_paths=(str(c03_csv),),
        source_paths=(str(src),),
        filter_logic="reader-facing columns; restricted formal subsamples; derived from hypothesis_tests.csv",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table C.3",
    )
    return r_a1, r_t05, r_c02, r_c03


def _build_a2(reg_inc: pd.DataFrame, processed_dir: Path, ts: str) -> tuple[ExhibitRecord, ExhibitRecord]:
    _require_columns(
        reg_inc,
        ("sample", "term", "coef", "se_hac", "t", "p_two_sided", "ci_lo_95", "ci_hi_95", "r_squared", "n_obs"),
        name="A2 regression_incremental.csv",
    )
    out_csv = _appendix_tables_dir(processed_dir) / "OLD_A2_incremental_regression.csv"
    out_t6_csv = (
        _appendix_tables_dir(processed_dir)
        / "OLD_Table_T06_incremental_regression_gnn_conditional_on_vix.csv"
    )
    _write_csv(reg_inc, out_csv)
    _write_csv(reg_inc, out_t6_csv)
    r_a2 = ExhibitRecord(
        exhibit_id="A2",
        title="OLD: incremental regression regression_incremental.csv (archival)",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(processed_dir / "summaries" / "regression_incremental.csv"),),
        filter_logic="verbatim copy of summaries/regression_incremental.csv",
        output_formats=("csv",),
        build_timestamp_utc=ts,
    )
    r_t06 = ExhibitRecord(
        exhibit_id="T06",
        title="OLD: duplicate incremental regression export (same as A2)",
        section="appendix",
        output_paths=(str(out_t6_csv),),
        source_paths=(str(processed_dir / "summaries" / "regression_incremental.csv"),),
        filter_logic="duplicate of OLD_A2_*",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="",
    )
    return r_a2, r_t06


def _build_a3(reg_mz: pd.DataFrame, processed_dir: Path, ts: str) -> tuple[ExhibitRecord, ExhibitRecord]:
    _require_columns(
        reg_mz,
        ("sample", "term", "coef", "se_hac", "t", "p_two_sided", "ci_lo_95", "ci_hi_95", "r_squared", "n_obs"),
        name="A3 regression_mz_gnn.csv",
    )
    out_csv = _appendix_tables_dir(processed_dir) / "OLD_A3_mz_regression.csv"
    out_t7_csv = _appendix_tables_dir(processed_dir) / "OLD_Table_T07_mincer_zarnowitz_regression_gnn.csv"
    _write_csv(reg_mz, out_csv)
    _write_csv(reg_mz, out_t7_csv)
    r_a3 = ExhibitRecord(
        exhibit_id="A3",
        title="OLD: MZ regression regression_mz_gnn.csv (archival)",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(processed_dir / "summaries" / "regression_mz_gnn.csv"),),
        filter_logic="verbatim copy of summaries/regression_mz_gnn.csv",
        output_formats=("csv",),
        build_timestamp_utc=ts,
    )
    r_t07 = ExhibitRecord(
        exhibit_id="T07",
        title="OLD: duplicate MZ regression export (same as A3)",
        section="appendix",
        output_paths=(str(out_t7_csv),),
        source_paths=(str(processed_dir / "summaries" / "regression_mz_gnn.csv"),),
        filter_logic="duplicate of OLD_A3_*",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="",
    )
    return r_a3, r_t07


def _build_a4(processed_dir: Path, ts: str) -> tuple[ExhibitRecord, ExhibitRecord]:
    src1 = processed_dir / "figures" / "universe_churn" / "universe_turnover_timeseries.png"
    src2 = processed_dir / "figures" / "universe_churn" / "universe_rankbucket_replacement_heatmap.png"
    src3 = processed_dir / "figures" / "universe_churn" / "universe_tenure_distribution.png"
    for p in (src1, src2, src3):
        if (not p.is_file()) or p.stat().st_size <= 0:
            raise FileNotFoundError(f"A4 source figure missing/empty: {p}")

    img1 = plt.imread(src1)
    img2 = plt.imread(src2)
    img3 = plt.imread(src3)
    out_png = _appendix_figures_dir(processed_dir) / "A4_universe_diagnostics_plate.png"
    out_pdf = _appendix_figures_dir(processed_dir) / "A4_universe_diagnostics_plate.pdf"
    out_f1_png = _appendix_figures_dir(processed_dir) / "Figure_F01_universe_diagnostics_plate.png"
    out_f1_pdf = _appendix_figures_dir(processed_dir) / "Figure_F01_universe_diagnostics_plate.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    for ax, img, label in zip(axes, (img1, img2, img3), ("(a)", "(b)", "(c)")):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_f1_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_f1_pdf, bbox_inches="tight")
    plt.close(fig)
    r_a4 = ExhibitRecord(
        exhibit_id="A4",
        title="Universe Replacement Diagnostics Plate",
        section="appendix",
        output_paths=(str(out_png), str(out_pdf)),
        source_paths=(str(src1), str(src2), str(src3)),
        filter_logic="scripted 3-panel composition of existing universe_churn figures",
        output_formats=("png", "pdf"),
        build_timestamp_utc=ts,
    )
    r_f01 = ExhibitRecord(
        exhibit_id="F01",
        title="Universe diagnostics plate (thesis filename; same composition as A4)",
        section="appendix",
        output_paths=(str(out_f1_png), str(out_f1_pdf)),
        source_paths=(str(src1), str(src2), str(src3)),
        filter_logic="same raster as A4",
        output_formats=("png", "pdf"),
        build_timestamp_utc=ts,
        thesis_label="Figure F01",
    )
    return r_a4, r_f01


def _build_a5(univ_summary: pd.DataFrame, processed_dir: Path, ts: str) -> tuple[ExhibitRecord, ExhibitRecord]:
    if len(univ_summary) == 0:
        raise ValueError("A5: universe_churn_summary.csv is empty")
    src = processed_dir / "summaries" / "universe_churn_summary.csv"
    out_full = _appendix_tables_dir(processed_dir) / "OLD_A5_universe_churn_summary_full.csv"
    out_compact = _appendix_tables_dir(processed_dir) / "OLD_A5_universe_churn_summary_compact.csv"
    out_t8_full = _appendix_tables_dir(processed_dir) / "OLD_Table_T08_universe_churn_summary_full.csv"
    out_t8_compact = _appendix_tables_dir(processed_dir) / "OLD_Table_T08_universe_churn_summary_compact.csv"
    out_full.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_full)
    shutil.copy2(src, out_t8_full)

    cols = [
        "run_id",
        "universe_mode",
        "n_feature_dates",
        "avg_universe_size",
        "turnover_rate_mean",
        "turnover_rate_p95",
        "tenure_mean",
        "tenure_p90",
        "rankbucket_replacement_mean",
        "rankbucket_replacement_p95",
    ]
    missing = [c for c in cols if c not in univ_summary.columns]
    if missing:
        raise ValueError(f"A5 compact export missing expected columns: {missing}")
    compact = univ_summary.loc[:, cols].copy()
    for c in [
        "avg_universe_size",
        "turnover_rate_mean",
        "turnover_rate_p95",
        "tenure_mean",
        "tenure_p90",
        "rankbucket_replacement_mean",
        "rankbucket_replacement_p95",
    ]:
        compact[c] = compact[c].map(_fmt_num)
    _write_csv(compact, out_compact)
    _write_csv(compact, out_t8_compact)
    r_a5 = ExhibitRecord(
        exhibit_id="A5",
        title="OLD: universe churn summary (full + compact subset, archival)",
        section="appendix",
        output_paths=(str(out_full), str(out_compact)),
        source_paths=(str(src),),
        filter_logic="verbatim full CSV + compact numeric subset as CSV",
        output_formats=("csv",),
        build_timestamp_utc=ts,
    )
    r_t08 = ExhibitRecord(
        exhibit_id="T08",
        title="OLD: duplicate universe churn summary (same as A5)",
        section="appendix",
        output_paths=(str(out_t8_full), str(out_t8_compact)),
        source_paths=(str(src),),
        filter_logic="duplicate of OLD_A5_*",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="",
    )
    return r_a5, r_t08


def _build_a6(diag: pd.DataFrame, processed_dir: Path, ts: str) -> tuple[ExhibitRecord, ExhibitRecord]:
    _require_columns(diag, ("sample", "diagnostic", "model", "value"), name="A6 diagnostics_smoothing.csv")
    sub_full = _diagnostics_calibration_long_subset(diag)
    sub = sub_full[["sample", "diagnostic", "model", "value"]].copy()

    out_csv = _appendix_tables_dir(processed_dir) / "OLD_A6_compact_calibration_stats.csv"
    out_t9_csv = _appendix_tables_dir(processed_dir) / "OLD_Table_T09_compact_calibration_statistics.csv"
    _write_csv(sub, out_csv)
    _write_csv(sub, out_t9_csv)
    r_a6 = ExhibitRecord(
        exhibit_id="A6",
        title="OLD: compact calibration long-format export (archival)",
        section="appendix",
        output_paths=(str(out_csv),),
        source_paths=(str(processed_dir / "summaries" / "diagnostics_smoothing.csv"),),
        filter_logic="sample in {full_test,excl_2020}; diagnostics in {forecast_variance_log,corr_with_target_log,mz_intercept,mz_slope}; models in {gnn,vix,har,placebo}",
        output_formats=("csv",),
        build_timestamp_utc=ts,
    )
    r_t09 = ExhibitRecord(
        exhibit_id="T09",
        title="OLD: duplicate compact calibration long-format (same as A6)",
        section="appendix",
        output_paths=(str(out_t9_csv),),
        source_paths=(str(processed_dir / "summaries" / "diagnostics_smoothing.csv"),),
        filter_logic="duplicate of OLD_A6_*",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="",
    )
    return r_a6, r_t09


def _build_d01_d02_calibration_thesis(
    diag: pd.DataFrame, processed_dir: Path, ts: str
) -> tuple[ExhibitRecord, ExhibitRecord]:
    """Thesis-wide calibration tables (Appendix D.1 / D.2) from the same diagnostics as OLD A6."""
    _require_columns(diag, ("sample", "diagnostic", "model", "value"), name="D01/D02 diagnostics_smoothing.csv")
    sub = _diagnostics_calibration_long_subset(diag)
    d01 = _calibration_wide_display_for_sample(sub, "full_test")
    d02 = _calibration_wide_display_for_sample(sub, "excl_2020")
    p01 = _appendix_tables_dir(processed_dir) / "Table_D01_calibration_diagnostics_full_test_sample.csv"
    p02 = _appendix_tables_dir(processed_dir) / "Table_D02_calibration_diagnostics_test_sample_excluding_2020.csv"
    _write_csv(d01, p01)
    _write_csv(d02, p02)
    src = str(processed_dir / "summaries" / "diagnostics_smoothing.csv")
    r1 = ExhibitRecord(
        exhibit_id="D01",
        title="Table D.1. Calibration diagnostics for the full test sample.",
        section="appendix",
        output_paths=(str(p01),),
        source_paths=(src,),
        filter_logic="Wide calibration diagnostics; GNN, VIX, HAR, Placebo; four decimal places",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table D.1",
    )
    r2 = ExhibitRecord(
        exhibit_id="D02",
        title="Table D.2. Calibration diagnostics for the test sample excluding 2020.",
        section="appendix",
        output_paths=(str(p02),),
        source_paths=(src,),
        filter_logic="Same columns as Table D.1; test sample excluding configured calendar years",
        output_formats=("csv",),
        build_timestamp_utc=ts,
        thesis_label="Table D.2",
    )
    return r1, r2


def _manifest_paths(processed_dir: Path) -> tuple[Path, Path]:
    md = _manifest_dir(processed_dir) / "thesis_exhibits_manifest.md"
    js = _manifest_dir(processed_dir) / "thesis_exhibits_manifest.json"
    return md, js


def _write_manifest(processed_dir: Path, records: list[ExhibitRecord]) -> None:
    mdir = _manifest_dir(processed_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    md_path, js_path = _manifest_paths(processed_dir)
    payload = [r.__dict__ for r in records]
    js_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Thesis Exhibits Manifest",
        "",
        "Appendix tables: CSV filenames under `appendix/tables/` per `manifest/thesis_table_index.md` "
        "(thesis-facing `Table_A01_…` / `OLD_…`). Main tables may include `.tex` under `main/tables/`. "
        "Cite **Figure F01** for the appendix figure.",
        "",
        "| exhibit_id | thesis_label | title | section | output_paths | source_paths | filter_logic | output_formats | build_timestamp_utc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        tl = r.thesis_label or "—"
        lines.append(
            f"| {r.exhibit_id} | {tl} | {r.title} | {r.section} | "
            f"{'<br>'.join(r.output_paths)} | {'<br>'.join(r.source_paths)} | "
            f"{r.filter_logic} | {', '.join(r.output_formats)} | {r.build_timestamp_utc} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_paths_for_thesis_export(cfg: ResolvedConfig) -> list[Path]:
    proc = Path(cfg.processed_dir)
    app_tables = _appendix_tables_dir(proc)
    app_fig = _appendix_figures_dir(proc)
    man = _manifest_dir(proc)
    thesis_tables = [
        # Thesis-facing appendix tables (CSV only)
        "Table_A01_data_inputs_and_roles.csv",
        "Table_A02_wrds_crsp_raw_input_schema.csv",
        "Table_A03_required_target_and_benchmark_schema.csv",
        "Table_B01_canonical_configuration.csv",
        "Table_B02_hypothesis_to_procedure_artifact_map.csv",
        "Table_C01_formal_test_subsamples_used_in_robustness_analysis.csv",
        "Table_C02_full_test_formal_hypothesis_results.csv",
        "Table_C03_restricted_subsample_robustness_results.csv",
        "Table_D01_calibration_diagnostics_full_test_sample.csv",
        "Table_D02_calibration_diagnostics_test_sample_excluding_2020.csv",
        "Table_D03_log_scale_forecast_error_by_target_volatility_quintile_full_test_sample.csv",
        "Table_D04_log_scale_forecast_error_by_target_volatility_quintile_test_sample_excluding_2020.csv",
        # Archival / pipeline (OLD_ prefix)
        "OLD_A1_full_hypothesis_output.csv",
        "OLD_Table_T05_full_formal_hypothesis_output.csv",
        "OLD_A2_incremental_regression.csv",
        "OLD_Table_T06_incremental_regression_gnn_conditional_on_vix.csv",
        "OLD_A3_mz_regression.csv",
        "OLD_Table_T07_mincer_zarnowitz_regression_gnn.csv",
        "OLD_A5_universe_churn_summary_full.csv",
        "OLD_A5_universe_churn_summary_compact.csv",
        "OLD_Table_T08_universe_churn_summary_full.csv",
        "OLD_Table_T08_universe_churn_summary_compact.csv",
        "OLD_A6_compact_calibration_stats.csv",
        "OLD_Table_T09_compact_calibration_statistics.csv",
    ]
    out: list[Path] = [
        _main_tables_dir(proc) / "M1_oos_fulltest_accuracy.csv",
        _main_tables_dir(proc) / "M1_oos_fulltest_accuracy.tex",
        _main_figures_dir(proc) / "M2_test_forecast_paths.png",
        _main_tables_dir(proc) / "M3_h1_h3_formal_tests_fulltest.csv",
        _main_tables_dir(proc) / "M3_h1_h3_formal_tests_fulltest.tex",
        _main_tables_dir(proc) / "M4_h4_robustness_tests.csv",
        _main_tables_dir(proc) / "M4_h4_robustness_tests.tex",
        _main_figures_dir(proc) / "M5_mse_log_by_vol_quintile.png",
        _main_figures_dir(proc) / "M5_mse_log_by_vol_quintile.pdf",
        app_fig / "A4_universe_diagnostics_plate.png",
        app_fig / "A4_universe_diagnostics_plate.pdf",
    ]
    out.extend([app_tables / name for name in thesis_tables])
    out.extend(
        [
            app_fig / "Figure_F01_universe_diagnostics_plate.png",
            app_fig / "Figure_F01_universe_diagnostics_plate.pdf",
            man / "Reproducibility_metadata.json",
            man / "thesis_table_index.md",
            man / "thesis_exhibits_manifest.md",
            man / "thesis_exhibits_manifest.json",
        ]
    )
    return out


def run_thesis_export(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Build thesis exhibit package from existing processed artifacts."""
    proc = Path(cfg.processed_dir)
    if not proc.is_dir():
        raise FileNotFoundError(f"Processed directory missing: {proc}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary = pd.read_csv(proc / "summaries" / "summary_table.csv")
    hyp = pd.read_csv(proc / "summaries" / "hypothesis_tests.csv")
    diag = pd.read_csv(proc / "summaries" / "diagnostics_smoothing.csv")
    reg_inc = pd.read_csv(proc / "summaries" / "regression_incremental.csv")
    reg_mz = pd.read_csv(proc / "summaries" / "regression_mz_gnn.csv")
    univ_summary = pd.read_csv(proc / "summaries" / "universe_churn_summary.csv")

    if overwrite:
        root = thesis_root(proc)
        if root.is_dir():
            shutil.rmtree(root)

    records: list[ExhibitRecord] = []
    records.append(_build_m1(summary, proc, ts))
    records.append(_build_m2(proc, ts))
    records.append(_build_m3(hyp, proc, ts))
    records.append(_build_m4(hyp, proc, ts))
    records.append(_build_m5(diag, proc, ts))
    r_a1, r_t05, r_c02, r_c03 = _build_a1(hyp, proc, ts)
    r_a2, r_t06 = _build_a2(reg_inc, proc, ts)
    r_a3, r_t07 = _build_a3(reg_mz, proc, ts)
    r_a4, r_f01 = _build_a4(proc, ts)
    r_a5, r_t08 = _build_a5(univ_summary, proc, ts)
    r_a6, r_t09 = _build_a6(diag, proc, ts)
    r_d01, r_d02 = _build_d01_d02_calibration_thesis(diag, proc, ts)
    records.extend([r_a1, r_a2, r_a3, r_a4, r_a5, r_a6])
    records.extend(_build_appendix_a_input_schema_tables(proc, ts))
    records.append(_build_t02(cfg, proc, ts))
    records.append(_build_t03(cfg, proc, ts))
    records.append(_build_t04(proc, ts))
    records.extend([r_t05, r_c02, r_c03, r_t06, r_t07, r_t08, r_t09, r_d01, r_d02])
    records.append(_build_d03_quintile_table(diag, proc, ts))
    records.append(_build_d04_quintile_table(diag, proc, ts))
    records.append(r_f01)
    records.append(_write_reproducibility_metadata(cfg, ts))
    _write_thesis_table_index(proc, ts)
    _write_manifest(proc, records)

