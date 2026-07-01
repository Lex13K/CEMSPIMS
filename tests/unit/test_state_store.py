"""Tests for event-sourced state store and log parsing."""

from __future__ import annotations

from mss.app import events as ev
from mss.app.log_line_parser import parse_headline, parse_step_banner
from mss.app.state_store import StateStore
from mss.run.events_emit import event_to_app_event, parse_event_line


def test_step_log_line_updates_live_headline() -> None:
    store = StateStore()
    store.apply(ev.job_started(job_id="j1", run_id="default", status="running", created_at="t"))
    store.apply(ev.step_started(run_id="default", pipeline="model.train", step_id="train"))
    deltas = store.apply(
        ev.step_log_line(
            job_id="j1",
            run_id="default",
            line="Epoch:  10%|█| 5/50 [00:30<04:00, best=1, pat=0/5, train=0.5, val=0.4]",
        )
    )
    assert deltas
    live = store.get_state().live_by_run["default"]
    assert live.live_headline is not None
    assert "5/50" in live.live_headline or "10%" in live.live_headline


def test_step_started_finished_updates_status() -> None:
    store = StateStore()
    store.apply(ev.run_created(run_id="child", parent_run_id="default", branch_pipeline="model.train"))
    deltas = store.apply(ev.step_started(run_id="child", pipeline="model.train", step_id="train"))
    assert any(d.get("type") == "snapshot" for d in deltas)
    state = store.get_state()
    assert state.runs["child"].steps["model.train/train"].status == "running"
    store.apply(
        ev.step_finished(run_id="child", pipeline="model.train", step_id="train", status="completed")
    )
    assert store.get_state().runs["child"].steps["model.train/train"].status == "complete"


def test_run_removed_drops_run() -> None:
    store = StateStore()
    store.apply(ev.run_created(run_id="x"))
    store.apply(ev.run_removed(run_id="x"))
    assert "x" not in store.get_state().runs


def test_parse_event_line_step_started() -> None:
    line = '@@EVENT {"kind":"step_started","run_id":"default","pipeline":"graph.prepare","step_id":"edges"}'
    data = parse_event_line(line)
    assert data is not None
    app_ev = event_to_app_event(data)
    assert app_ev is not None
    assert app_ev.kind == "step_started"


def test_parse_step_banner() -> None:
    banner = parse_step_banner("--- model.train: step train ---")
    assert banner == {"pipeline": "model.train", "step_id": "train"}


def test_parse_headline_edges() -> None:
    h = parse_headline("  [  100/ 5000] edges completed (chunk 1/50, latest date 2020-01-01)")
    assert h is not None
    assert "100" in h
