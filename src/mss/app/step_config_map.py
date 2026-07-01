"""Map pipeline steps to editable TOML config fields (experiment graph app)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FieldType = Literal["int", "float", "bool", "enum", "str"]


@dataclass(frozen=True)
class ConfigField:
    section: str
    key: str
    label: str
    help: str
    field_type: FieldType
    enum_values: tuple[str, ...] | None = None

    @property
    def dot_key(self) -> str:
        return f"{self.section}.{self.key}"


def _fields(*items: ConfigField) -> tuple[ConfigField, ...]:
    return items


_STEP_FIELDS: dict[tuple[str, str], tuple[ConfigField, ...]] = {
    ("graph.prepare", "feature_dates"): _fields(
        ConfigField(
            "graph",
            "align_feature_dates_with_targets",
            "Align feature dates with targets",
            "Restrict graph dates to targets spine when true.",
            "bool",
        ),
    ),
    ("graph.prepare", "universe"): _fields(
        ConfigField(
            "graph.universe",
            "universe_mode",
            "Universe mode",
            "fixed_replace (sticky top-N) or monthly_rebalance.",
            "enum",
            ("fixed_replace", "monthly_rebalance"),
        ),
        ConfigField(
            "graph.universe",
            "n_nodes",
            "Node count",
            "Broad point-in-time universe size per date. Scale presets: 500 / 1000 / 1500.",
            "int",
        ),
        ConfigField(
            "graph.universe",
            "selection_rule",
            "Selection rule",
            "mcap (canonical broad point-in-time universe) or dollar_volume (robustness only).",
            "enum",
            ("mcap", "dollar_volume"),
        ),
        ConfigField(
            "graph.universe",
            "restrict_to_sp500",
            "Restrict to S&P 500",
            "Robustness only: filter to S&P 500 members as-of date. Canonical universe is broad mcap.",
            "bool",
        ),
        ConfigField(
            "graph.universe",
            "rebalance_freq",
            "Rebalance frequency",
            "Used when universe_mode is monthly_rebalance.",
            "enum",
            ("daily", "monthly", "quarterly"),
        ),
        ConfigField(
            "graph.universe",
            "n_jobs",
            "Parallel jobs",
            "Universe build parallelism (1 on Windows if IO-bound).",
            "int",
        ),
    ),
    ("graph.prepare", "node_features"): _fields(
        ConfigField(
            "graph.rolling_window",
            "length",
            "Rolling window (days)",
            "Trading-day window for rolling stats.",
            "int",
        ),
        ConfigField(
            "graph.node_features",
            "include_log_dollar_volume",
            "Log dollar volume feature",
            "Liquidity feature (log price*volume). Enabled in the v2 default.",
            "bool",
        ),
        ConfigField(
            "graph.node_features",
            "include_turnover",
            "Turnover feature",
            "Volume/shares-outstanding turnover. Enabled in the v2 default.",
            "bool",
        ),
        ConfigField(
            "graph.node_features",
            "include_downside_semivol",
            "Downside semi-volatility",
            "Rolling sqrt(mean of squared negative returns). Enabled in the v2 default.",
            "bool",
        ),
        ConfigField(
            "graph.node_features",
            "include_skew",
            "Return skew feature",
            "Rolling return skewness over the window. Enabled in the v2 default.",
            "bool",
        ),
    ),
    ("graph.prepare", "edges"): _fields(
        ConfigField(
            "graph.edges",
            "top_k",
            "Top-K neighbors",
            "Correlation neighbors kept per node per date.",
            "int",
        ),
        ConfigField(
            "graph.edges",
            "dependence",
            "Dependence measure",
            "Correlation type for edge weights.",
            "enum",
            ("spearman", "pearson"),
        ),
        ConfigField(
            "graph.edges",
            "symmetrize",
            "Symmetrize edges",
            "Mirror edges for undirected graph.",
            "bool",
        ),
        ConfigField(
            "graph.edges",
            "n_jobs",
            "Parallel jobs",
            "Edge build worker count.",
            "int",
        ),
    ),
    ("dataset.package", "splits"): _fields(
        ConfigField("dataset", "train_end", "Train end", "Last train date (inclusive).", "str"),
        ConfigField("dataset", "val_end", "Val end", "Last validation date (inclusive).", "str"),
        ConfigField("dataset", "test_end", "Test end", "Last test date (inclusive).", "str"),
    ),
    ("dataset.package", "scaler"): _fields(
        ConfigField(
            "dataset",
            "scaler_type",
            "Scaler type",
            "Feature scaler fit on train only.",
            "enum",
            ("standard", "robust"),
        ),
    ),
    ("dataset.package", "manifest"): _fields(
        ConfigField(
            "graph.output",
            "target_column",
            "Target column",
            "v2 default log_rv_fwd_30cal (forward 30-calendar-day RV). log_rv_fwd_30 is legacy/robustness.",
            "str",
        ),
    ),
    ("model.train", "train"): _fields(
        ConfigField(
            "model.train",
            "use_edge_weights",
            "Use edge weights",
            "Pass abs(edge_attr) into GraphSAGE/GAT convolutions.",
            "bool",
        ),
        ConfigField(
            "model.train",
            "hidden_channels",
            "Hidden channels",
            "GNN hidden width.",
            "int",
        ),
        ConfigField(
            "model.train",
            "num_gnn_layers",
            "GNN layers",
            "Number of message-passing layers.",
            "int",
        ),
        ConfigField(
            "model.train",
            "max_epochs",
            "Max epochs",
            "Training epoch cap before early stopping.",
            "int",
        ),
        ConfigField(
            "model.train",
            "early_stopping_patience",
            "Early-stop patience",
            "Validation epochs without improvement.",
            "int",
        ),
    ),
    ("model.evaluate", "score_splits"): _fields(
        ConfigField(
            "model.evaluate",
            "enable_placebo_ablation",
            "Placebo ablation",
            "Score permuted/zero placebo topology at eval.",
            "bool",
        ),
        ConfigField(
            "model.evaluate",
            "placebo_edge_mode",
            "Placebo edge mode",
            "zero_attr (default) or permute_only.",
            "enum",
            ("zero_attr", "permute_only"),
        ),
        ConfigField(
            "model.evaluate",
            "placebo_retrain",
            "Retrain placebo arm",
            "Train a second model on placebo graphs for H5 (doubles train cost). Off by default.",
            "bool",
        ),
    ),
    ("model.evaluate", "write_forecast_panel"): _fields(
        ConfigField(
            "model.evaluate",
            "fit_vix_log_calibration",
            "Fold calibration into raw VIX",
            "Legacy: also apply the train affine map to raw_vix. calibrated_vix is always emitted separately.",
            "bool",
        ),
    ),
    ("model.evaluate", "run_hypothesis_tests"): _fields(
        ConfigField(
            "model.evaluate",
            "hypothesis_stress_preset",
            "H4 stress preset",
            "covid_2020 or rate_shock_2022.",
            "enum",
            ("covid_2020", "rate_shock_2022"),
        ),
    ),
    ("analysis.summarize", "loss_figure"): _fields(
        ConfigField("analysis.summarize", "dpi", "Figure DPI", "Raster figure resolution.", "int"),
    ),
}


def fields_for_step(pipeline: str, step_id: str) -> tuple[ConfigField, ...]:
    return _STEP_FIELDS.get((pipeline, step_id), ())


def _nested_get(data: dict[str, Any], section: str, key: str) -> Any:
    parts = section.split(".")
    cur: Any = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if not isinstance(cur, dict):
        return None
    return cur.get(key)


def read_field_values(config_path: Path, fields: tuple[ConfigField, ...]) -> list[dict[str, Any]]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for f in fields:
        val = _nested_get(raw, f.section, f.key)
        out.append(
            {
                "dot_key": f.dot_key,
                "section": f.section,
                "key": f.key,
                "label": f.label,
                "help": f.help,
                "field_type": f.field_type,
                "enum_values": list(f.enum_values) if f.enum_values else None,
                "value": val,
            }
        )
    return out
