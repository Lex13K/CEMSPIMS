"""analysis.report pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path

from mss.analysis.config import load_analysis_report_config
from mss.analysis.h1 import run_h1_regression_table
from mss.analysis.pairwise import run_pairwise_report
from mss.evaluation.checks import forecasts_path
from mss.io.config import ResolvedConfig


def run_analysis_report(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    acfg = load_analysis_report_config(cfg.source_config_path)
    out_dir = Path(cfg.processed_dir)
    pairwise_json = out_dir / "pairwise_comparison.json"
    formal_csv = out_dir / "tables" / "formal_comparison.csv"
    insample_csv = out_dir / "tables" / "insample_pairwise.csv"
    h1_csv = out_dir / "tables" / "h1_regression.csv"
    fc_path = forecasts_path(cfg)
    if not fc_path.is_file():
        raise FileNotFoundError("analysis.report requires processed/forecasts.parquet. Run model.evaluate first.")
    if not overwrite and pairwise_json.is_file() and formal_csv.is_file() and insample_csv.is_file():
        if (not acfg.include_h1) or h1_csv.is_file():
            return

    run_pairwise_report(fc_path, pairwise_json, formal_csv, reps=acfg.bootstrap_reps)
    # Minimal insample table: train+val subset from formal table.
    import pandas as pd

    formal_df = pd.read_csv(formal_csv)
    insample = formal_df[formal_df["sample"].isin(["train_full", "val_full"])].copy()
    insample_csv.parent.mkdir(parents=True, exist_ok=True)
    insample.to_csv(insample_csv, index=False)

    if acfg.include_h1:
        run_h1_regression_table(fc_path, h1_csv)

