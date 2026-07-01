"""Rebuild AppState from manifests and step journals (cold path / rescan)."""

from __future__ import annotations

from pathlib import Path

from mss.app.state_store import AppState, RunState, StepState, _step_key
from mss.io.config import load_resolved_config
from mss.io.paths import project_root
from mss.pipeline.step_catalog import iter_catalog_steps
from mss.run.registry import RunRecord, list_runs
from mss.run.step_journal import load_step_journal

_PREPARE_STEPS = {
    _step_key(p, s)
    for p, s in iter_catalog_steps()
    if p == "data.prepare"
}


def _journal_status_to_display(status: str) -> str:
    if status == "completed":
        return "complete"
    if status == "running":
        return "running"
    if status == "skipped":
        return "complete"
    return "missing"


def _raw_status_from_disk(configs_dir: Path) -> str:
    records = list_runs(configs_dir)
    if not records:
        return "missing"
    try:
        cfg = load_resolved_config(records[0].config_path, records[0].run_id)
        raw = cfg.raw_dir
        if not raw.is_dir():
            return "missing"
        needed = ("WRDS.csv", "VIX.csv", "snp500_volatility.csv")
        if all((raw / n).is_file() for n in needed):
            return "complete"
    except (OSError, ValueError):
        pass
    return "missing"


def build_state_from_disk(configs_dir: Path | None = None) -> AppState:
    """One-time / rescan rebuild from journals and manifests — no artifact scans."""
    cdir = configs_dir or (project_root() / "configs")
    state = AppState()
    state.raw_status = _raw_status_from_disk(cdir)

    for rec in list_runs(cdir):
        run = RunState(
            run_id=rec.run_id,
            parent_run_id=rec.parent_run_id,
            branch_pipeline=rec.branch_pipeline,
            branch_step_id=rec.branch_step_id,
            pipeline_scope=rec.manifest.pipeline_scope if rec.manifest else None,
            created_at=rec.created_at,
            show_terminal=rec.parent_run_id is None,
        )
        try:
            cfg = load_resolved_config(rec.config_path, rec.run_id)
            journal = load_step_journal(cfg)
            for entry in journal.entries.values():
                key = _step_key(entry.pipeline, entry.step_id)
                display = _journal_status_to_display(entry.status)
                if entry.pipeline == "data.prepare" and entry.step_id != "raw":
                    state.global_steps[key] = StepState(
                        status=display, updated_at=entry.completed_at or entry.started_at
                    )
                else:
                    run.steps[key] = StepState(
                        status=display,
                        updated_at=entry.completed_at or entry.started_at,
                        error=entry.error,
                    )
                    if entry.pipeline == "analysis.summarize" and entry.step_id == "loss_figure":
                        run.terminal_status = display
                        run.show_terminal = True
                if rec.parent_run_id and key in run.steps:
                    run.show_terminal = True
        except (OSError, ValueError):
            pass
        state.runs[rec.run_id] = run

    return state


def bootstrap_state_store(store, *, configs_dir: Path | None = None) -> None:
    """Load snapshot or migrate from disk."""
    if store.load_snapshot():
        return
    state = build_state_from_disk(configs_dir)
    store.replace_state(state)
