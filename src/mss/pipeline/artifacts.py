"""
Per-step artifact predicates for skip-if-complete (no content hashing).

When adding a new pipeline (e.g. graph.prepare), register completion checks here
alongside orchestrator PIPELINES.
"""

from __future__ import annotations

from pathlib import Path

from mss.data.ingest import IngestPaths
from mss.io.config import ResolvedConfig
from mss.pipeline import completeness as _complete


def _crsp_has_at_least_one_parquet(cfg: ResolvedConfig) -> bool:
    crsp = IngestPaths.from_resolved_config(cfg).interim_dir / "crsp_parquet"
    if not crsp.is_dir():
        return False
    return any(crsp.rglob("*.parquet"))


def ingest_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.ingest_step_semantically_complete(cfg)


def returns_panel_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.returns_panel_step_semantically_complete(cfg)


def targets_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.targets_step_semantically_complete(cfg)


def _graphs_dir(cfg: ResolvedConfig) -> Path:
    return IngestPaths.from_resolved_config(cfg).interim_dir / "graphs"


def feature_dates_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.feature_dates_step_semantically_complete(cfg)


def universe_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.universe_step_semantically_complete(cfg)


def node_features_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.node_features_step_semantically_complete(cfg)


def edges_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.edges_step_semantically_complete(cfg)


def dataset_splits_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.dataset_splits_step_semantically_complete(cfg)


def dataset_scaler_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.dataset_scaler_step_semantically_complete(cfg)


def dataset_manifest_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.dataset_manifest_step_semantically_complete(cfg)


def model_train_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_train_step_semantically_complete(cfg)


def model_cache_graphs_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_cache_graphs_step_semantically_complete(cfg)


def model_evaluate_score_splits_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_score_splits_semantically_complete(cfg)


def model_evaluate_compute_error_series_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_aggregate_test_loss_semantically_complete(cfg)


def model_evaluate_write_summary_table_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_write_summary_table_semantically_complete(cfg)


def model_evaluate_write_forecast_panel_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_write_forecast_panel_semantically_complete(cfg)


def model_evaluate_run_hypothesis_tests_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_run_hypothesis_tests_semantically_complete(cfg)


def model_evaluate_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.model_evaluate_step_semantically_complete(cfg)


def analysis_summarize_pairwise_stats_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.analysis_summarize_loss_figure_semantically_complete(cfg)


def thesis_export_step_complete(cfg: ResolvedConfig) -> bool:
    return _complete.thesis_export_semantically_complete(cfg)


def step_is_complete(pipeline: str, step_id: str, cfg: ResolvedConfig) -> bool:
    """Return True if this step's outputs are present (skip re-run when not overwriting)."""
    if pipeline == "data.prepare":
        if step_id == "ingest":
            return ingest_step_complete(cfg)
        if step_id == "returns_panel":
            return returns_panel_step_complete(cfg)
        if step_id == "targets":
            return targets_step_complete(cfg)
        return False
    if pipeline == "graph.prepare":
        if step_id == "feature_dates":
            return feature_dates_step_complete(cfg)
        if step_id == "universe":
            return universe_step_complete(cfg)
        if step_id == "node_features":
            return node_features_step_complete(cfg)
        if step_id == "edges":
            return edges_step_complete(cfg)
        return False
    if pipeline == "dataset.package":
        if step_id == "splits":
            return dataset_splits_step_complete(cfg)
        if step_id == "scaler":
            return dataset_scaler_step_complete(cfg)
        if step_id == "manifest":
            return dataset_manifest_step_complete(cfg)
        return False
    if pipeline == "model.train":
        if step_id == "train":
            return model_train_step_complete(cfg)
        return False
    if pipeline == "model.cache_graphs":
        if step_id == "materialize":
            return model_cache_graphs_step_complete(cfg)
        return False
    if pipeline == "model.evaluate":
        if step_id == "score_splits":
            return model_evaluate_score_splits_step_complete(cfg)
        if step_id == "write_forecast_panel":
            return model_evaluate_write_forecast_panel_step_complete(cfg)
        if step_id == "aggregate_test_loss":
            return model_evaluate_compute_error_series_step_complete(cfg)
        if step_id == "write_summary_table":
            return model_evaluate_write_summary_table_step_complete(cfg)
        if step_id == "run_hypothesis_tests":
            return model_evaluate_run_hypothesis_tests_step_complete(cfg)
        return False
    if pipeline == "analysis.summarize":
        if step_id == "loss_figure":
            return analysis_summarize_pairwise_stats_step_complete(cfg)
        return False
    if pipeline == "thesis.export":
        if step_id == "build":
            return thesis_export_step_complete(cfg)
        return False
    return False


def ensure_step_inputs_ready(pipeline: str, step_id: str, cfg: ResolvedConfig) -> None:
    """
    Before running a step (when not skipping), verify upstream artifacts exist.
    Raises ValueError with a clear message if not.
    """
    paths = IngestPaths.from_resolved_config(cfg)
    interim = paths.interim_dir

    if pipeline == "data.prepare":
        if step_id == "returns_panel":
            if not _crsp_has_at_least_one_parquet(cfg):
                raise ValueError(
                    "returns_panel needs CRSP parquet shards under interim/crsp_parquet/. "
                    "Run ingest first or use --overwrite data.prepare."
                )
        if step_id == "targets":
            if not (interim / "sp500_returns.parquet").is_file() or not (interim / "vix.parquet").is_file():
                raise ValueError(
                    "targets needs interim/sp500_returns.parquet and interim/vix.parquet. "
                    "Run ingest first or use --overwrite data.prepare."
                )
        return

    if pipeline == "graph.prepare":
        if step_id == "feature_dates":
            if not (interim / "returns_panel.parquet").is_file():
                raise ValueError(
                    "feature_dates needs interim/returns_panel.parquet. "
                    "Run data.prepare through returns_panel or use --overwrite graph.prepare."
                )
            if not (interim / "targets.parquet").is_file():
                raise ValueError(
                    "feature_dates needs interim/targets.parquet. "
                    "Run data.prepare through targets or use --overwrite graph.prepare."
                )
        elif step_id == "universe":
            if not (interim / "stage04_dates.parquet").is_file():
                raise ValueError(
                    "universe needs interim/stage04_dates.parquet. "
                    "Run graph.prepare step feature_dates or use --overwrite graph.prepare."
                )
            if not (interim / "returns_panel.parquet").is_file():
                raise ValueError("universe needs interim/returns_panel.parquet.")
        elif step_id == "node_features":
            if not (_graphs_dir(cfg) / "universe.parquet").is_file():
                raise ValueError(
                    "node_features needs interim/graphs/universe.parquet. "
                    "Run graph.prepare through universe or use --overwrite graph.prepare."
                )
            if not (interim / "returns_panel.parquet").is_file():
                raise ValueError("node_features needs interim/returns_panel.parquet.")
        elif step_id == "edges":
            if not (_graphs_dir(cfg) / "universe.parquet").is_file():
                raise ValueError(
                    "edges needs interim/graphs/universe.parquet. "
                    "Run graph.prepare through universe or use --overwrite graph.prepare."
                )
            if not (interim / "returns_panel.parquet").is_file():
                raise ValueError("edges needs interim/returns_panel.parquet.")
        return

    if pipeline == "dataset.package":
        gdir = _graphs_dir(cfg)
        if step_id == "splits":
            if not (gdir / "universe.parquet").is_file():
                raise ValueError(
                    "dataset.package splits needs interim/graphs/universe.parquet. "
                    "Run graph.prepare through universe or use --overwrite graph.prepare."
                )
        elif step_id == "scaler":
            if not (interim / "dataset" / "splits.parquet").is_file():
                raise ValueError(
                    "dataset.package scaler needs interim/dataset/splits.parquet. "
                    "Run dataset.package step splits or use --overwrite dataset.package."
                )
            if not (gdir / "node_features.parquet").is_file():
                raise ValueError(
                    "dataset.package scaler needs interim/graphs/node_features.parquet. "
                    "Run graph.prepare through node_features or use --overwrite graph.prepare."
                )
        elif step_id == "manifest":
            if not (interim / "dataset" / "splits.parquet").is_file():
                raise ValueError(
                    "dataset.package manifest needs interim/dataset/splits.parquet. "
                    "Run dataset.package through scaler or use --overwrite dataset.package."
                )
            if not (interim / "dataset" / "scaler_params.json").is_file():
                raise ValueError(
                    "dataset.package manifest needs interim/dataset/scaler_params.json."
                )
            for rel, msg in (
                (gdir / "node_features.parquet", "node_features"),
                (gdir / "edges.parquet", "edges"),
                (gdir / "universe.parquet", "universe"),
                (interim / "targets.parquet", "targets"),
            ):
                if not rel.is_file():
                    raise ValueError(
                        f"dataset.package manifest needs {rel.name} ({msg}). "
                        "Run upstream pipelines or use --overwrite."
                    )
        return

    if pipeline == "model.train":
        if step_id == "train":
            man = interim / "dataset" / "manifest.json"
            if not man.is_file():
                raise ValueError(
                    "model.train needs interim/dataset/manifest.json. "
                    "Run dataset.package or use --overwrite dataset.package."
                )
            try:
                from mss.model_train.manifest_io import load_dataset_manifest

                load_dataset_manifest(man)
            except (OSError, ValueError, FileNotFoundError) as e:
                raise ValueError(
                    f"model.train: invalid or incomplete dataset manifest: {e}"
                ) from e
        return

    if pipeline == "model.cache_graphs":
        if step_id == "materialize":
            man = interim / "dataset" / "manifest.json"
            if not man.is_file():
                raise ValueError(
                    "model.cache_graphs needs interim/dataset/manifest.json. "
                    "Run dataset.package or use --overwrite dataset.package."
                )
            try:
                from mss.model_train.manifest_io import load_dataset_manifest

                load_dataset_manifest(man)
            except (OSError, ValueError, FileNotFoundError) as e:
                raise ValueError(
                    f"model.cache_graphs: invalid or incomplete dataset manifest: {e}"
                ) from e
        return

    if pipeline == "model.evaluate":
        if step_id == "score_splits":
            man = interim / "dataset" / "manifest.json"
            if not man.is_file():
                raise ValueError(
                    "model.evaluate needs interim/dataset/manifest.json. "
                    "Run dataset.package or use --overwrite dataset.package."
                )
            try:
                from mss.model_train.manifest_io import load_dataset_manifest

                load_dataset_manifest(man)
            except (OSError, ValueError, FileNotFoundError) as e:
                raise ValueError(
                    f"model.evaluate: invalid or incomplete dataset manifest: {e}"
                ) from e
            from mss.model_train.checks import (
                CHECKPOINT_FILENAME,
                check_final_metrics_complete,
                final_metrics_path,
                model_train_dir,
            )

            ck = model_train_dir(cfg) / CHECKPOINT_FILENAME
            if not ck.is_file():
                raise ValueError(
                    "model.evaluate needs interim/model_train/checkpoint.pt. "
                    "Run model.train first."
                )
            fm_chk = check_final_metrics_complete(final_metrics_path(cfg))
            if not fm_chk.get("passed"):
                raise ValueError(
                    "model.evaluate needs completed interim/model_train/final_metrics.json. "
                    "Run model.train to completion first."
                )
        elif step_id == "write_forecast_panel":
            from mss.evaluation.checks import check_forecasts, forecasts_path

            fp = forecasts_path(cfg)
            if not fp.is_file():
                raise ValueError(
                    "model.evaluate write_forecast_panel needs processed/forecasts.parquet."
                )
            if not check_forecasts(fp).get("passed"):
                raise ValueError("model.evaluate write_forecast_panel: invalid forecasts.parquet.")
        elif step_id == "aggregate_test_loss":
            from mss.evaluation.checks import check_forecasts, forecasts_path

            fp = forecasts_path(cfg)
            if not fp.is_file():
                raise ValueError(
                    "model.evaluate aggregate_test_loss needs processed/forecasts.parquet "
                    "from score_splits."
                )
            if not check_forecasts(fp).get("passed"):
                raise ValueError("model.evaluate aggregate_test_loss: invalid forecasts.parquet.")
        elif step_id == "write_summary_table":
            from mss.evaluation.checks import (
                check_forecast_panel,
                check_forecasts,
                check_test_loss,
                forecast_panel_path,
                forecasts_path,
                test_loss_path,
            )

            fp = forecasts_path(cfg)
            if not fp.is_file():
                raise ValueError(
                    "model.evaluate write_summary_table needs processed/forecasts.parquet."
                )
            if not check_forecasts(fp).get("passed"):
                raise ValueError("model.evaluate write_summary_table: invalid forecasts.parquet.")
            fpp = forecast_panel_path(cfg)
            if not check_forecast_panel(fpp).get("passed"):
                raise ValueError(
                    "model.evaluate write_summary_table needs valid processed/forecast_panel.parquet."
                )
            tl = test_loss_path(cfg)
            if not check_test_loss(tl).get("passed"):
                raise ValueError(
                    "model.evaluate write_summary_table needs valid processed/test_loss.json."
                )
            man = interim / "dataset" / "manifest.json"
            if not man.is_file():
                raise ValueError(
                    "model.evaluate write_summary_table needs interim/dataset/manifest.json."
                )
        elif step_id == "run_hypothesis_tests":
            from mss.evaluation.checks import check_forecast_panel, forecast_panel_path

            fpp = forecast_panel_path(cfg)
            if not check_forecast_panel(fpp).get("passed"):
                raise ValueError(
                    "model.evaluate run_hypothesis_tests needs valid processed/forecast_panel.parquet."
                )
        return

    if pipeline == "analysis.summarize":
        if step_id == "loss_figure":
            from mss.evaluation.checks import (
                check_forecasts,
                check_test_loss,
                forecasts_path,
                test_loss_path,
            )
            from mss.model_train.checks import training_state_path

            tp = training_state_path(cfg)
            if not tp.is_file():
                raise ValueError(
                    "analysis.summarize loss_figure needs interim/model_train/training_state.json."
                )
            tl = test_loss_path(cfg)
            if not check_test_loss(tl).get("passed"):
                raise ValueError(
                    "analysis.summarize loss_figure needs processed/test_loss.json "
                    "(run model.evaluate first)."
                )
            fp = forecasts_path(cfg)
            if not fp.is_file():
                raise ValueError(
                    "analysis.summarize loss_figure needs processed/forecasts.parquet "
                    "(run model.evaluate score_splits first)."
                )
            if not check_forecasts(fp).get("passed"):
                raise ValueError("analysis.summarize loss_figure: invalid forecasts.parquet.")
            from mss.evaluation.checks import check_summary_table, summary_table_path

            st = summary_table_path(cfg)
            if not check_summary_table(st).get("passed"):
                raise ValueError(
                    "analysis.summarize loss_figure needs processed/summaries/summary_table.csv "
                    "(run model.evaluate write_summary_table first)."
                )

            universe_path = interim / "graphs" / "universe.parquet"
            if not universe_path.is_file():
                raise ValueError(
                    "analysis.summarize loss_figure needs interim/graphs/universe.parquet "
                    "(run graph.prepare universe first)."
                )
            if universe_path.stat().st_size <= 0:
                raise ValueError(
                    "analysis.summarize loss_figure needs non-empty interim/graphs/universe.parquet."
                )
            from mss.analysis.config import load_analysis_summarize_config
            from mss.evaluation.checks import check_hypothesis_tests, hypothesis_tests_path

            if load_analysis_summarize_config(cfg.source_config_path).draw_hypothesis_table:
                htp = hypothesis_tests_path(cfg)
                if not check_hypothesis_tests(htp).get("passed"):
                    raise ValueError(
                        "analysis.summarize loss_figure needs processed/summaries/hypothesis_tests.csv "
                        "(run model.evaluate run_hypothesis_tests first), or set "
                        "[analysis.summarize] draw_hypothesis_table = false."
                    )
        return

    if pipeline == "thesis.export":
        if step_id == "build":
            req_files = (
                "summaries/summary_table.csv",
                "summaries/hypothesis_tests.csv",
                "summaries/diagnostics_smoothing.csv",
                "summaries/regression_incremental.csv",
                "summaries/regression_mz_gnn.csv",
                "summaries/universe_churn_summary.csv",
                "figures/target_vs_model_vix/target_vs_model_vix_test.png",
                "figures/universe_churn/universe_turnover_timeseries.png",
                "figures/universe_churn/universe_rankbucket_replacement_heatmap.png",
                "figures/universe_churn/universe_tenure_distribution.png",
            )
            for rel in req_files:
                p = Path(cfg.processed_dir) / rel
                if not p.is_file():
                    raise ValueError(
                        f"thesis.export build needs processed/{rel}. Run model.evaluate/analysis.summarize first."
                    )
                if p.stat().st_size <= 0:
                    raise ValueError(f"thesis.export build needs non-empty processed/{rel}.")
        return
