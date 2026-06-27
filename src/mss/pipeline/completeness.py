"""
Semantic step completion (not just file existence).

Used by artifacts.step_is_complete for resume/skip decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from mss.data.ingest import IngestPaths
from mss.data.returns_panel import check_returns_panel
from mss.data.targets import check_targets
from mss.graph.config import GraphConfig, load_graph_config
from mss.graph.expected import compute_expected_feature_dates, normalized_date_set
from mss.io.config import ResolvedConfig


def _graphs_dir(cfg: ResolvedConfig) -> Path:
    return IngestPaths.from_resolved_config(cfg).interim_dir / "graphs"


def ingest_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Tier-1: manifest + required parquet outputs exist (same as before, no stale detection)."""
    paths = IngestPaths.from_resolved_config(cfg)
    interim = paths.interim_dir
    if not (interim / "ingest_manifest.json").is_file():
        return False
    if not (interim / "sp500_returns.parquet").is_file():
        return False
    if not (interim / "vix.parquet").is_file():
        return False
    crsp = interim / "crsp_parquet"
    if not crsp.is_dir():
        return False
    return any(crsp.rglob("*.parquet"))


def returns_panel_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    paths = IngestPaths.from_resolved_config(cfg)
    p = paths.interim_dir / "returns_panel.parquet"
    if not p.is_file():
        return False
    try:
        chk = check_returns_panel(p)
    except Exception:
        return False
    if not chk.get("passed"):
        return False
    summary = chk.get("summary")
    if summary is None or len(summary) == 0:
        return False
    n_rows = summary.iloc[0].get("n_rows")
    return n_rows is not None and int(n_rows) > 0


def targets_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    interim = IngestPaths.from_resolved_config(cfg).interim_dir
    tg = interim / "targets.parquet"
    man = interim / "targets_manifest.json"
    if not tg.is_file() or not man.is_file():
        return False
    try:
        with man.open(encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    stats = meta.get("stats") or {}
    expected_n = stats.get("n_rows")
    if expected_n is None:
        return False
    try:
        con = duckdb.connect(database=":memory:")
        n = con.execute(
            "SELECT COUNT(*) AS n FROM read_parquet(?)",
            [str(tg.resolve())],
        ).fetchone()[0]
        con.close()
    except Exception:
        return False
    if int(n) != int(expected_n):
        return False
    chk = check_targets(tg, expect_vix_column=True)
    return bool(chk.get("passed"))


def feature_dates_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """stage04_dates matches recomputed expected dates from returns + graph config."""
    try:
        gc = load_graph_config(cfg.source_config_path)
    except OSError:
        return False
    paths = IngestPaths.from_resolved_config(cfg)
    interim = paths.interim_dir
    out = interim / "stage04_dates.parquet"
    rp = interim / "returns_panel.parquet"
    if not out.is_file() or not rp.is_file():
        return False
    tg_path = interim / "targets.parquet" if gc.align_feature_dates_with_targets else None
    if gc.align_feature_dates_with_targets and (tg_path is None or not tg_path.is_file()):
        return False
    try:
        expected = compute_expected_feature_dates(rp, gc, tg_path)
    except (OSError, RuntimeError, FileNotFoundError):
        return False
    try:
        actual = pd.read_parquet(out, columns=["date"])["date"]
    except Exception:
        return False
    e = pd.to_datetime(expected).dt.normalize().sort_values().reset_index(drop=True)
    a = pd.to_datetime(actual).dt.normalize().sort_values().reset_index(drop=True)
    return len(e) == len(a) and e.equals(a)


def universe_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Universe dates match stage04_dates exactly (same set)."""
    paths = IngestPaths.from_resolved_config(cfg)
    interim = paths.interim_dir
    univ_path = _graphs_dir(cfg) / "universe.parquet"
    stage04 = interim / "stage04_dates.parquet"
    if not univ_path.is_file() or not stage04.is_file():
        return False
    try:
        u_dates = pd.read_parquet(univ_path, columns=["date"])["date"]
        s_dates = pd.read_parquet(stage04, columns=["date"])["date"]
    except Exception:
        return False
    return normalized_date_set(u_dates) == normalized_date_set(s_dates)


def node_features_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Row count matches universe; contract checks pass."""
    from mss.graph import checks as graph_checks

    gdir = _graphs_dir(cfg)
    nf_path = gdir / "node_features.parquet"
    u_path = gdir / "universe.parquet"
    if not nf_path.is_file() or not u_path.is_file():
        return False
    try:
        con = duckdb.connect(database=":memory:")
        n_nf = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(nf_path.resolve())],
        ).fetchone()[0]
        n_u = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(u_path.resolve())],
        ).fetchone()[0]
        con.close()
    except Exception:
        return False
    if int(n_nf) != int(n_u):
        return False
    res = graph_checks.check_node_features(
        nf_path, feature_cols=("rolling_mean", "rolling_vol")
    )
    return bool(res.get("passed"))


def edges_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Distinct edge dates match universe (catches partial edges checkpoint)."""
    from mss.graph import checks as graph_checks

    gdir = _graphs_dir(cfg)
    e_path = gdir / "edges.parquet"
    u_path = gdir / "universe.parquet"
    if not e_path.is_file() or not u_path.is_file():
        return False
    try:
        con = duckdb.connect(database=":memory:")
        n_e = con.execute(
            "SELECT COUNT(DISTINCT date) FROM read_parquet(?)",
            [str(e_path.resolve())],
        ).fetchone()[0]
        n_u = con.execute(
            "SELECT COUNT(DISTINCT date) FROM read_parquet(?)",
            [str(u_path.resolve())],
        ).fetchone()[0]
        con.close()
    except Exception:
        return False
    if int(n_u) == 0:
        return False
    n_e = int(n_e)
    n_u = int(n_u)
    if n_e != n_u:
        return False
    res = graph_checks.check_edges(e_path)
    return bool(res.get("passed"))


def dataset_splits_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.dataset.checks import splits_step_semantically_complete

    return splits_step_semantically_complete(cfg)


def dataset_scaler_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.dataset.checks import scaler_step_semantically_complete

    return scaler_step_semantically_complete(cfg)


def dataset_manifest_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.dataset.checks import manifest_step_semantically_complete

    return manifest_step_semantically_complete(cfg)


def model_train_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.model_train.checks import model_train_step_semantically_complete as mt_done

    return mt_done(cfg)


def model_cache_graphs_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Semantic completion for the canonical PyG graph cache materialization step.

    This returns True only when *all* expected per-date `.pt` files exist and are non-empty.
    """

    paths = IngestPaths.from_resolved_config(cfg)
    interim = paths.interim_dir

    man_path = interim / "dataset" / "manifest.json"
    if not man_path.is_file():
        return False

    try:
        from mss.model_train.dates import train_val_expected_dates_and_labels_df
        from mss.model_train.manifest_io import load_dataset_manifest
        from mss.model_train.checks import model_train_dir
    except Exception:
        return False

    try:
        manifest = load_dataset_manifest(man_path)
        expected_dates, _labels_df = train_val_expected_dates_and_labels_df(manifest)
    except Exception:
        return False

    if not expected_dates:
        return False

    cache_dir = model_train_dir(cfg) / "graph_cache"
    for date in expected_dates:
        pt_path = cache_dir / f"{pd.Timestamp(date).strftime('%Y-%m-%d')}.pt"
        if not pt_path.is_file():
            return False
        try:
            if pt_path.stat().st_size <= 0:
                return False
        except OSError:
            return False

    return True


def model_evaluate_score_splits_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_score_splits_semantically_complete as done

    return done(cfg)


def model_evaluate_compute_error_series_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_aggregate_test_loss_semantically_complete as done

    return done(cfg)


def model_evaluate_aggregate_test_loss_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_aggregate_test_loss_semantically_complete as done
    return done(cfg)


def model_evaluate_write_summary_table_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_write_summary_table_semantically_complete as done

    return done(cfg)


def model_evaluate_write_forecast_panel_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_write_forecast_panel_semantically_complete as done

    return done(cfg)


def model_evaluate_run_hypothesis_tests_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.evaluation.checks import model_evaluate_run_hypothesis_tests_semantically_complete as done

    return done(cfg)


def model_evaluate_step_semantically_complete(cfg: ResolvedConfig) -> bool:
    """Full model.evaluate pipeline."""
    from mss.evaluation.checks import model_evaluate_pipeline_semantically_complete as done

    return done(cfg)


def analysis_summarize_loss_figure_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.analysis.figures import expected_paths_for_summarize

    for p in expected_paths_for_summarize(cfg):
        if not p.is_file():
            return False
        try:
            if p.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def thesis_export_semantically_complete(cfg: ResolvedConfig) -> bool:
    from mss.thesis.export import expected_paths_for_thesis_export

    for p in expected_paths_for_thesis_export(cfg):
        if not p.is_file():
            return False
        try:
            if p.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True
