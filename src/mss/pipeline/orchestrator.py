from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from mss.data.ingest import IngestPaths, ingest_raw_to_parquet, validate_ingest
from mss.data.config import load_returns_panel_config
from mss.data.returns_panel import build_returns_panel, check_returns_panel
from mss.data.targets import build_targets, check_targets
from mss.dataset.checks import (
    check_manifest,
    check_scaler_json,
    check_splits,
    splits_paths,
)
from mss.dataset.config import load_dataset_config
from mss.dataset.manifest import package_scaled_and_manifest
from mss.dataset.scaler import fit_and_save_scaler
from mss.dataset.splits import write_splits_parquet
from mss.graph import checks as graph_checks
from mss.graph.config import load_graph_config, min_obs_from_config
from mss.graph.edges import build_edge_lists
from mss.graph.feature_dates import build_feature_date_index
from mss.graph.node_features import build_node_features
from mss.graph.universe import build_universe_per_date
from mss.io.config import ResolvedConfig, load_resolved_config
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.artifacts import ensure_step_inputs_ready, step_is_complete, step_stale_reason
from mss.run.step_journal import record_step_finish, record_step_start
from mss.config.fingerprints import (
    write_dataset_fingerprint_sidecar,
    write_evaluate_fingerprint_sidecar,
    write_graph_fingerprint_sidecar,
)
from mss.run.manifest import ensure_run_manifest, refresh_prepare_hashes
from mss.run.paths import graphs_dir

StepFn = Callable[[ResolvedConfig, bool, float, float], None]

def _step_ingest(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    paths = IngestPaths.from_resolved_config(cfg)
    ingest_raw_to_parquet(paths, overwrite=overwrite)
    validate_ingest(paths, atol=atol, rtol=rtol)


def _step_returns_panel(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol  # unused; signature matches StepFn
    paths = IngestPaths.from_resolved_config(cfg)
    rp_cfg = load_returns_panel_config(cfg.source_config_path)
    out_path = paths.prepare_interim_dir / "returns_panel.parquet"
    build_returns_panel(
        paths.prepare_interim_dir / "crsp_parquet",
        out_path,
        use_ret=rp_cfg.use_ret,
        apply_delisting_adjustment=rp_cfg.apply_delisting_adjustment,
        common_shares_only=rp_cfg.common_shares_only,
        major_exchanges_only=rp_cfg.major_exchanges_only,
        overwrite=overwrite,
    )
    result = check_returns_panel(out_path)
    if not result["passed"]:
        n = result["duplicates"]["n_duplicate_pairs"]
        raise ValueError(
            f"returns_panel contract check failed: {n} duplicate (date, permno) pair(s)"
        )


def _step_targets(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    paths = IngestPaths.from_resolved_config(cfg)
    paths.prepare_interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.prepare_interim_dir / "targets.parquet"
    manifest_path = paths.prepare_interim_dir / "targets_manifest.json"
    build_targets(
        paths.prepare_interim_dir / "sp500_returns.parquet",
        out_path,
        manifest_path,
        vix_parquet_path=paths.prepare_interim_dir / "vix.parquet",
        overwrite=overwrite,
    )
    result = check_targets(out_path, expect_vix_column=True)
    if not result["passed"]:
        msg = "; ".join(result.get("issues", [])) or result.get("error", "check_targets failed")
        raise ValueError(f"targets contract check failed: {msg}")


def _graphs_interim(cfg: ResolvedConfig) -> Path:
    return graphs_dir(cfg)


def _step_feature_dates(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    gc = load_graph_config(cfg.source_config_path)
    paths = IngestPaths.from_resolved_config(cfg)
    shared = paths.prepare_interim_dir
    run_interim = cfg.run_interim_dir
    run_interim.mkdir(parents=True, exist_ok=True)
    tg = shared / "targets.parquet" if gc.align_feature_dates_with_targets else None
    build_feature_date_index(
        shared / "returns_panel.parquet",
        run_interim / "stage04_dates.parquet",
        gc=gc,
        targets_parquet_path=tg,
        overwrite=overwrite,
    )


def _step_universe(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    gc = load_graph_config(cfg.source_config_path)
    paths = IngestPaths.from_resolved_config(cfg)
    shared = paths.prepare_interim_dir
    run_interim = cfg.run_interim_dir
    gdir = _graphs_interim(cfg)
    gdir.mkdir(parents=True, exist_ok=True)
    min_obs = max(1, min_obs_from_config(gc))
    membership_path = gc.sp500_membership_path
    if gc.restrict_to_sp500 and not membership_path:
        membership_path = str(shared / "sp500_membership.parquet")
    df = build_universe_per_date(
        shared / "returns_panel.parquet",
        run_interim / "stage04_dates.parquet",
        window_length=gc.window_length,
        min_obs=min_obs,
        n_nodes=gc.n_nodes,
        selection_rule=gc.selection_rule,
        universe_mode=gc.universe_mode,
        rebalance_freq=gc.rebalance_freq,
        restrict_to_sp500=gc.restrict_to_sp500,
        sp500_membership_path=membership_path if gc.restrict_to_sp500 else None,
        progress_every=gc.universe_progress_every,
        n_jobs=gc.universe_n_jobs,
        show_progress=gc.show_progress,
        verbose=gc.verbose,
    )
    out_path = gdir / "universe.parquet"
    df.to_parquet(out_path, index=False)
    res = graph_checks.check_universe(out_path)
    if not res["passed"]:
        raise ValueError(f"universe contract check failed: {res['issues']}")


def _step_node_features(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    gc = load_graph_config(cfg.source_config_path)
    paths = IngestPaths.from_resolved_config(cfg)
    shared = paths.prepare_interim_dir
    gdir = _graphs_interim(cfg)
    univ = pd.read_parquet(gdir / "universe.parquet")
    min_obs = max(1, min_obs_from_config(gc))
    nf = build_node_features(
        shared / "returns_panel.parquet",
        univ,
        window_length=gc.window_length,
        ret_col=gc.ret_col,
        rolling_mean=gc.rolling_mean,
        rolling_vol=gc.rolling_vol,
        include_log_dollar_volume=gc.include_log_dollar_volume,
        include_turnover=gc.include_turnover,
        include_downside_semivol=gc.include_downside_semivol,
        include_skew=gc.include_skew,
        progress_every=gc.node_features_progress_every,
        n_jobs=gc.node_features_n_jobs,
        min_obs=min_obs,
        verbose=gc.verbose,
        debug_progress=gc.debug_progress,
        show_progress=gc.show_progress,
    )
    out_path = gdir / "node_features.parquet"
    nf.to_parquet(out_path, index=False)
    res = graph_checks.check_node_features(
        out_path, feature_cols=gc.node_feature_column_names()
    )
    if not res["passed"]:
        raise ValueError(f"node_features contract check failed: {res['issues']}")


def _step_edges(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    gc = load_graph_config(cfg.source_config_path)
    paths = IngestPaths.from_resolved_config(cfg)
    shared = paths.prepare_interim_dir
    gdir = _graphs_interim(cfg)
    univ = pd.read_parquet(gdir / "universe.parquet")
    out_path = gdir / "edges.parquet"
    build_edge_lists(
        shared / "returns_panel.parquet",
        univ,
        window_length=gc.window_length,
        ret_col=gc.ret_col,
        dependence=gc.dependence,
        top_k=gc.top_k,
        symmetrize=gc.symmetrize,
        update_every=gc.edges_update_every,
        out_path=out_path,
        overwrite=overwrite,
        n_jobs=gc.edges_n_jobs,
        show_progress=gc.show_progress,
        verbose=gc.verbose,
    )
    res = graph_checks.check_edges(out_path)
    if not res["passed"]:
        raise ValueError(f"edges contract check failed: {res['issues']}")
    write_graph_fingerprint_sidecar(cfg)


def _step_dataset_splits(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    dc = load_dataset_config(cfg.source_config_path)
    p = splits_paths(cfg)
    p["dataset_dir"].mkdir(parents=True, exist_ok=True)
    write_splits_parquet(
        p["universe"],
        p["splits"],
        train_end=dc.train_end,
        val_end=dc.val_end,
        test_end=dc.test_end,
    )
    res = check_splits(p["splits"])
    if not res["passed"]:
        raise ValueError(f"splits contract check failed: {res['issues']}")


def _step_dataset_scaler(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    dc = load_dataset_config(cfg.source_config_path)
    p = splits_paths(cfg)
    fit_and_save_scaler(
        p["node_features"],
        p["splits"],
        p["scaler"],
        feature_columns=dc.feature_columns,
        scaler_type=dc.scaler_type,
    )
    res = check_scaler_json(p["scaler"], dc.feature_columns)
    if not res["passed"]:
        raise ValueError(f"scaler contract check failed: {res['issues']}")


def _step_dataset_manifest(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    dc = load_dataset_config(cfg.source_config_path)
    p = splits_paths(cfg)
    scaled_out = p["scaled_nf"] if dc.write_scaled_features else None
    package_scaled_and_manifest(
        splits_path=p["splits"],
        node_features_path=p["node_features"],
        scaler_path=p["scaler"],
        edges_path=p["edges"],
        universe_path=p["universe"],
        targets_path=p["targets"],
        out_manifest_path=p["manifest"],
        out_scaled_path=scaled_out,
        target_column=dc.target_column,
        feature_columns=dc.feature_columns,
        write_scaled_features=dc.write_scaled_features,
    )
    res = check_manifest(p["manifest"])
    if not res["passed"]:
        raise ValueError(f"manifest contract check failed: {res['issues']}")
    write_dataset_fingerprint_sidecar(cfg)


def _step_model_train(cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float) -> None:
    del atol, rtol
    from mss.model_train.train_loop import run_model_train, run_placebo_retrain

    run_model_train(cfg, overwrite=overwrite)
    # Optional retrained-placebo arm (H5); no-op unless [model.evaluate].placebo_retrain=true.
    run_placebo_retrain(cfg, overwrite=overwrite)


def _step_model_cache_graphs(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.model_train.cache_materialize import materialize_pyg_graph_cache

    materialize_pyg_graph_cache(cfg, overwrite=overwrite)


def _step_model_evaluate_score_splits(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.evaluation.inference import run_score_splits

    run_score_splits(cfg, overwrite=overwrite)


def _step_model_evaluate_write_forecast_panel(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.evaluation.inference import run_write_forecast_panel

    run_write_forecast_panel(cfg, overwrite=overwrite)
    write_evaluate_fingerprint_sidecar(cfg)


def _step_model_evaluate_compute_error_series(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.evaluation.inference import run_aggregate_test_loss

    run_aggregate_test_loss(cfg, overwrite=overwrite)


def _step_model_evaluate_write_summary_table(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.evaluation.inference import run_write_summary_table

    run_write_summary_table(cfg, overwrite=overwrite)


def _step_model_evaluate_run_hypothesis_tests(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.evaluation.inference import run_hypothesis_tests_step

    run_hypothesis_tests_step(cfg, overwrite=overwrite)
    write_evaluate_fingerprint_sidecar(cfg)


def _step_summarize_pairwise_stats(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.analysis.summarize import run_summarize_loss_figure

    run_summarize_loss_figure(cfg, overwrite=overwrite)


def _step_thesis_export_build(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.thesis.export import run_thesis_export

    run_thesis_export(cfg, overwrite=overwrite)


def _step_compare_runs_build(
    cfg: ResolvedConfig, overwrite: bool, atol: float, rtol: float
) -> None:
    del atol, rtol
    from mss.analysis.compare_runs import run_compare_runs_from_anchor_config

    run_compare_runs_from_anchor_config(cfg, overwrite=overwrite)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: tuple[tuple[str, StepFn], ...]


PIPELINES: dict[str, PipelineSpec] = {
    "data.prepare": PipelineSpec(
        name="data.prepare",
        steps=(
            ("ingest", _step_ingest),
            ("returns_panel", _step_returns_panel),
            ("targets", _step_targets),
        ),
    ),
    "graph.prepare": PipelineSpec(
        name="graph.prepare",
        steps=(
            ("feature_dates", _step_feature_dates),
            ("universe", _step_universe),
            ("node_features", _step_node_features),
            ("edges", _step_edges),
        ),
    ),
    "dataset.package": PipelineSpec(
        name="dataset.package",
        steps=(
            ("splits", _step_dataset_splits),
            ("scaler", _step_dataset_scaler),
            ("manifest", _step_dataset_manifest),
        ),
    ),
    "model.cache_graphs": PipelineSpec(
        name="model.cache_graphs",
        steps=(("materialize", _step_model_cache_graphs),),
    ),
    "model.train": PipelineSpec(
        name="model.train",
        steps=(("train", _step_model_train),),
    ),
    "model.evaluate": PipelineSpec(
        name="model.evaluate",
        steps=(
            ("score_splits", _step_model_evaluate_score_splits),
            ("write_forecast_panel", _step_model_evaluate_write_forecast_panel),
            ("aggregate_test_loss", _step_model_evaluate_compute_error_series),
            ("write_summary_table", _step_model_evaluate_write_summary_table),
            ("run_hypothesis_tests", _step_model_evaluate_run_hypothesis_tests),
        ),
    ),
    "analysis.summarize": PipelineSpec(
        name="analysis.summarize",
        steps=(("loss_figure", _step_summarize_pairwise_stats),),
    ),
    "thesis.export": PipelineSpec(
        name="thesis.export",
        steps=(("build", _step_thesis_export_build),),
    ),
    "analysis.compare_runs": PipelineSpec(
        name="analysis.compare_runs",
        steps=(("build", _step_compare_runs_build),),
    ),
}


def list_pipeline_names() -> list[str]:
    return sorted(PIPELINES.keys())


def describe_pipelines() -> str:
    lines: list[str] = []
    canon = ", ".join(CANONICAL_PIPELINE_ORDER)
    lines.append(f"  (canonical order: {canon})")
    for name in list_pipeline_names():
        spec = PIPELINES[name]
        step_list = ", ".join(s[0] for s in spec.steps)
        lines.append(f"  {name}: [{step_list}]")
    return "\n".join(lines)


def _pipelines_to_run(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(CANONICAL_PIPELINE_ORDER)
    unknown = [p for p in requested if p not in PIPELINES]
    if unknown:
        known = ", ".join(list_pipeline_names())
        raise ValueError(f"Unknown pipeline(s): {unknown}. Known: {known}")
    req_set = set(requested)
    return [p for p in CANONICAL_PIPELINE_ORDER if p in req_set]


def _overwrite_targets(
    overwrite_arg: list[str] | None, pipelines_to_run: list[str]
) -> set[str]:
    if overwrite_arg is None:
        return set()
    if len(overwrite_arg) == 0:
        return set(pipelines_to_run)
    unknown = set(overwrite_arg) - set(PIPELINES.keys())
    if unknown:
        known = ", ".join(list_pipeline_names())
        raise ValueError(f"Unknown pipeline(s) in --overwrite: {sorted(unknown)}. Known: {known}")
    extra = set(overwrite_arg) - set(pipelines_to_run)
    if extra:
        raise ValueError(
            "--overwrite pipeline(s) must be among the pipelines selected for this run: "
            f"{sorted(extra)} not in {pipelines_to_run}"
        )
    return set(overwrite_arg)


def run_single_pipeline(
    name: str,
    cfg: ResolvedConfig,
    *,
    overwrite: bool = False,
    skip_validate: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    step_from: str | None = None,
    step_to: str | None = None,
) -> None:
    if name not in PIPELINES:
        known = ", ".join(list_pipeline_names())
        raise ValueError(f"Unknown pipeline {name!r}. Known: {known}")

    spec = PIPELINES[name]
    step_ids = [s[0] for s in spec.steps]
    if step_from is not None and step_from not in step_ids:
        raise ValueError(f"Unknown --from step {step_from!r} for pipeline {name!r}. Steps: {step_ids}")
    if step_to is not None and step_to not in step_ids:
        raise ValueError(f"Unknown --to step {step_to!r} for pipeline {name!r}. Steps: {step_ids}")

    start_i = 0
    end_i = len(spec.steps)
    if step_from is not None:
        start_i = step_ids.index(step_from)
    if step_to is not None:
        end_i = step_ids.index(step_to) + 1
    if start_i > end_i:
        raise ValueError("--from must not be after --to")

    for step_id, fn in spec.steps[start_i:end_i]:
        print(f"--- {name}: step {step_id} ---")
        skipped = False
        if not overwrite and step_is_complete(name, step_id, cfg):
            print("skipped (outputs already present)")
            skipped = True
        stale = None if overwrite else step_stale_reason(name, step_id, cfg)
        record_step_start(
            cfg,
            name,
            step_id,
            overwritten=overwrite,
            stale_reason=stale,
        )
        from mss.run.events_emit import emit_event

        emit_event("step_started", run_id=cfg.run_id, pipeline=name, step_id=step_id)
        if skipped:
            record_step_finish(cfg, name, step_id, status="skipped", fingerprint_match=True)
            emit_event(
                "step_finished",
                run_id=cfg.run_id,
                pipeline=name,
                step_id=step_id,
                status="skipped",
            )
            continue
        if stale:
            print(f"running ({stale})")
        try:
            ensure_step_inputs_ready(name, step_id, cfg)
            if step_id == "ingest" and skip_validate:
                paths = IngestPaths.from_resolved_config(cfg)
                ingest_raw_to_parquet(paths, overwrite=overwrite)
            else:
                fn(cfg, overwrite, atol, rtol)
        except Exception as exc:
            record_step_finish(cfg, name, step_id, status="failed", error=str(exc))
            emit_event(
                "step_finished",
                run_id=cfg.run_id,
                pipeline=name,
                step_id=step_id,
                status="failed",
                error=str(exc),
            )
            raise
        fp_match = step_stale_reason(name, step_id, cfg) is None
        record_step_finish(cfg, name, step_id, status="completed", fingerprint_match=fp_match)
        emit_event(
            "step_finished",
            run_id=cfg.run_id,
            pipeline=name,
            step_id=step_id,
            status="completed",
        )


def run_pipelines(
    config_path: Path,
    run_id: str,
    *,
    pipelines: list[str] | None = None,
    overwrite_arg: list[str] | None = None,
    skip_validate: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    step_from: str | None = None,
    step_to: str | None = None,
) -> None:
    cfg = load_resolved_config(config_path, run_id)
    ensure_run_manifest(cfg)
    from mss.config.backfill import ensure_config_fingerprint_sidecars

    ensure_config_fingerprint_sidecars(cfg)
    to_run = _pipelines_to_run(pipelines)
    ow = _overwrite_targets(overwrite_arg, to_run)

    if len(to_run) > 1 and (step_from is not None or step_to is not None):
        raise ValueError("--from and --to are only allowed when exactly one pipeline is selected")

    for name in to_run:
        run_single_pipeline(
            name,
            cfg,
            overwrite=name in ow,
            skip_validate=skip_validate,
            atol=atol,
            rtol=rtol,
            step_from=step_from,
            step_to=step_to,
        )
        if name == "data.prepare":
            refresh_prepare_hashes(cfg)
