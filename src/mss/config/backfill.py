"""Backfill config fingerprint sidecars for pre–Phase 2 runs."""

from __future__ import annotations

from mss.config.fingerprints import (
    dataset_fingerprint_matches,
    dataset_fingerprint_path,
    evaluate_fingerprint_matches,
    evaluate_fingerprint_path,
    graph_fingerprint_matches,
    graph_fingerprint_path,
    read_fingerprint_sidecar,
    write_dataset_fingerprint_sidecar,
    write_evaluate_fingerprint_sidecar,
    write_graph_fingerprint_sidecar,
)
from mss.io.config import ResolvedConfig
from mss.pipeline.artifacts import step_is_complete
from mss.run.step_journal import load_step_journal, save_step_journal


def ensure_config_fingerprint_sidecars(cfg: ResolvedConfig) -> None:
    """Stamp or refresh sidecars when artifacts are semantically complete."""
    from mss.pipeline.completeness import (
        dataset_package_semantic_only,
        graph_prepare_semantic_only,
        model_evaluate_pipeline_semantic_only,
    )

    if graph_prepare_semantic_only(cfg) and (
        read_fingerprint_sidecar(graph_fingerprint_path(cfg)) is None
        or not graph_fingerprint_matches(cfg)
    ):
        write_graph_fingerprint_sidecar(cfg)

    if dataset_package_semantic_only(cfg) and (
        read_fingerprint_sidecar(dataset_fingerprint_path(cfg)) is None
        or not dataset_fingerprint_matches(cfg)
    ):
        write_dataset_fingerprint_sidecar(cfg)

    if model_evaluate_pipeline_semantic_only(cfg) and (
        read_fingerprint_sidecar(evaluate_fingerprint_path(cfg)) is None
        or not evaluate_fingerprint_matches(cfg)
    ):
        write_evaluate_fingerprint_sidecar(cfg)


def repair_completed_journal_entries(cfg: ResolvedConfig) -> None:
    """Clear false stale flags on completed steps that have valid artifacts."""
    journal = load_step_journal(cfg)
    changed = False
    for entry in journal.entries.values():
        if entry.status != "completed":
            continue
        if step_is_complete(entry.pipeline, entry.step_id, cfg):
            if entry.fingerprint_match is not True or entry.stale_reason:
                entry.fingerprint_match = True
                entry.stale_reason = None
                changed = True
    if changed:
        save_step_journal(cfg, journal)
