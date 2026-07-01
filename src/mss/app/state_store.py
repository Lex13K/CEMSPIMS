"""In-memory experiment app state with event sourcing and SSE broadcast."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mss.app.events import AppEvent
from mss.app.log_line_parser import parse_headline, parse_step_banner
from mss.io.paths import project_root

LIVE_LOG_MAX = 50
SNAPSHOT_DEBOUNCE_S = 0.5


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _step_key(pipeline: str, step_id: str) -> str:
    return f"{pipeline}/{step_id}"


@dataclass
class StepState:
    status: str = "missing"
    updated_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    parent_run_id: str | None = None
    branch_pipeline: str | None = None
    branch_step_id: str | None = None
    pipeline_scope: list[str] | None = None
    created_at: str = ""
    steps: dict[str, StepState] = field(default_factory=dict)
    terminal_status: str = "missing"
    show_terminal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "branch_pipeline": self.branch_pipeline,
            "branch_step_id": self.branch_step_id,
            "pipeline_scope": self.pipeline_scope,
            "created_at": self.created_at,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "terminal_status": self.terminal_status,
            "show_terminal": self.show_terminal,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunState:
        steps = {
            k: StepState(**v) if isinstance(v, dict) else StepState(status=str(v))
            for k, v in (d.get("steps") or {}).items()
        }
        return cls(
            run_id=d["run_id"],
            parent_run_id=d.get("parent_run_id"),
            branch_pipeline=d.get("branch_pipeline"),
            branch_step_id=d.get("branch_step_id"),
            pipeline_scope=d.get("pipeline_scope"),
            created_at=d.get("created_at", ""),
            steps=steps,
            terminal_status=d.get("terminal_status", "missing"),
            show_terminal=bool(d.get("show_terminal", True)),
        )


@dataclass
class JobState:
    job_id: str
    run_id: str
    status: str
    created_at: str
    pipelines: list[str] | None = None
    overwrite: list[str] | None = None
    error: str | None = None
    return_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobState:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class LiveRunState:
    active_step: dict[str, str] | None = None
    live_log: list[str] = field(default_factory=list)
    live_headline: str | None = None
    active_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_step": self.active_step,
            "live_log": list(self.live_log),
            "live_headline": self.live_headline,
            "active_job_id": self.active_job_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveRunState:
        return cls(
            active_step=d.get("active_step"),
            live_log=list(d.get("live_log") or []),
            live_headline=d.get("live_headline"),
            active_job_id=d.get("active_job_id"),
        )


@dataclass
class AppState:
    revision: int = 0
    raw_status: str = "missing"
    global_steps: dict[str, StepState] = field(default_factory=dict)
    runs: dict[str, RunState] = field(default_factory=dict)
    jobs: dict[str, JobState] = field(default_factory=dict)
    live_by_run: dict[str, LiveRunState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "raw_status": self.raw_status,
            "global_steps": {k: v.to_dict() for k, v in self.global_steps.items()},
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
            "live_by_run": {k: v.to_dict() for k, v in self.live_by_run.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AppState:
        return cls(
            revision=int(d.get("revision", 0)),
            raw_status=d.get("raw_status", "missing"),
            global_steps={
                k: StepState(**v) if isinstance(v, dict) else StepState(status=str(v))
                for k, v in (d.get("global_steps") or {}).items()
            },
            runs={k: RunState.from_dict(v) for k, v in (d.get("runs") or {}).items()},
            jobs={k: JobState.from_dict(v) for k, v in (d.get("jobs") or {}).items()},
            live_by_run={
                k: LiveRunState.from_dict(v) for k, v in (d.get("live_by_run") or {}).items()
            },
        )


class StateStore:
    def __init__(self, snapshot_path: Path | None = None) -> None:
        self._path = snapshot_path or (project_root() / "data" / ".app_state" / "state.json")
        self._lock = threading.RLock()
        self._state = AppState()
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._save_timer: threading.Timer | None = None
        self._pending_deltas: list[dict[str, Any]] = []

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def snapshot_path(self) -> Path:
        return self._path

    def get_state(self) -> AppState:
        with self._lock:
            return self._state

    def get_revision(self) -> int:
        with self._lock:
            return self._state.revision

    def load_snapshot(self) -> bool:
        if not self._path.is_file():
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            with self._lock:
                self._state = AppState.from_dict(data)
            return True
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            return False

    def replace_state(self, state: AppState) -> None:
        with self._lock:
            self._state = state
            self._state.revision += 1
        self._schedule_save()
        from mss.app.projection import project_graph

        snap = project_graph(self._state)
        self._broadcast({"type": "snapshot", **snap})

    def apply(self, event: AppEvent) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        with self._lock:
            self._apply_locked(event, deltas)
            if deltas:
                self._state.revision += 1
                for d in deltas:
                    d["revision"] = self._state.revision
        if deltas:
            self._schedule_save()
            self._broadcast_many(deltas)
        return deltas

    def _apply_locked(self, event: AppEvent, deltas: list[dict[str, Any]]) -> None:
        p = event.payload
        kind = event.kind
        s = self._state

        if kind == "run_created":
            rid = p["run_id"]
            s.runs[rid] = RunState(
                run_id=rid,
                parent_run_id=p.get("parent_run_id"),
                branch_pipeline=p.get("branch_pipeline"),
                branch_step_id=p.get("branch_step_id"),
                pipeline_scope=p.get("pipeline_scope"),
                created_at=p.get("created_at") or _now_iso(),
            )
            deltas.append({"type": "snapshot"})

        elif kind == "run_removed":
            rid = p["run_id"]
            s.runs.pop(rid, None)
            s.live_by_run.pop(rid, None)
            deltas.append({"type": "snapshot"})

        elif kind == "scope_extended":
            rid = p["run_id"]
            run = s.runs.get(rid)
            if run:
                run.pipeline_scope = list(p["pipeline_scope"])
                deltas.append({"type": "snapshot"})

        elif kind == "raw_status":
            s.raw_status = p["status"]
            deltas.append({"type": "raw_status", "status": s.raw_status})

        elif kind == "job_started":
            jid = p["job_id"]
            s.jobs[jid] = JobState(
                job_id=jid,
                run_id=p["run_id"],
                status=p.get("status", "running"),
                created_at=p.get("created_at", _now_iso()),
                pipelines=p.get("pipelines"),
                overwrite=p.get("overwrite"),
            )
            live = s.live_by_run.setdefault(p["run_id"], LiveRunState())
            live.active_job_id = jid
            deltas.append({"type": "job_update", "job": s.jobs[jid].to_dict()})

        elif kind == "job_updated":
            job = s.jobs.get(p["job_id"])
            if job:
                job.status = p["status"]
                job.error = p.get("error")
                deltas.append({"type": "job_update", "job": job.to_dict()})

        elif kind == "job_finished":
            job = s.jobs.get(p["job_id"])
            if job:
                job.status = p["status"]
                job.return_code = p.get("return_code")
                job.error = p.get("error")
            rid = p["run_id"]
            live = s.live_by_run.get(rid)
            if live and live.active_job_id == p["job_id"]:
                live.active_job_id = None
                if p["status"] in ("completed", "failed", "cancelled"):
                    live.active_step = None
            deltas.append({"type": "job_update", "job": job.to_dict() if job else p})

        elif kind == "step_started":
            rid, pipeline, step_id = p["run_id"], p["pipeline"], p["step_id"]
            key = _step_key(pipeline, step_id)
            if pipeline == "data.prepare" and step_id != "raw":
                s.global_steps[key] = StepState(status="running", updated_at=_now_iso())
            else:
                run = s.runs.setdefault(rid, RunState(run_id=rid))
                run.steps[key] = StepState(status="running", updated_at=_now_iso())
            live = s.live_by_run.setdefault(rid, LiveRunState())
            live.active_step = {"pipeline": pipeline, "step_id": step_id}
            live.live_log = []
            live.live_headline = None
            deltas.append(
                {
                    "type": "active_step",
                    "run_id": rid,
                    "active_step": live.active_step,
                    "status": "running",
                    "pipeline": pipeline,
                    "step_id": step_id,
                }
            )
            deltas.append({"type": "snapshot"})

        elif kind == "step_finished":
            rid, pipeline, step_id = p["run_id"], p["pipeline"], p["step_id"]
            st = p.get("status", "completed")
            display = "complete" if st == "completed" else ("running" if st == "running" else "missing")
            if st == "skipped":
                display = "complete"
            key = _step_key(pipeline, step_id)
            step = StepState(status=display, updated_at=_now_iso(), error=p.get("error"))
            if pipeline == "data.prepare" and step_id != "raw":
                s.global_steps[key] = step
            else:
                run = s.runs.setdefault(rid, RunState(run_id=rid))
                run.steps[key] = step
                if pipeline == "analysis.summarize" and step_id == "loss_figure":
                    run.terminal_status = display
                    run.show_terminal = True
            live = s.live_by_run.get(rid)
            if live and live.active_step == {"pipeline": pipeline, "step_id": step_id}:
                live.active_step = None
            deltas.append(
                {
                    "type": "node_update",
                    "run_id": rid,
                    "pipeline": pipeline,
                    "step_id": step_id,
                    "status": display,
                }
            )

        elif kind == "step_log_line":
            rid, line = p["run_id"], p["line"]
            clean = line.rstrip("\n")
            live = s.live_by_run.setdefault(rid, LiveRunState())
            log = deque(live.live_log, maxlen=LIVE_LOG_MAX)
            log.append(clean)
            live.live_log = list(log)
            headline = parse_headline(clean)
            if headline:
                live.live_headline = headline
            banner = parse_step_banner(clean)
            if banner:
                live.active_step = banner
            deltas.append(
                {
                    "type": "step_log",
                    "run_id": rid,
                    "line": clean,
                    "live_headline": live.live_headline,
                    "live_log": live.live_log[-10:],
                }
            )

    def _schedule_save(self) -> None:
        if self._save_timer is not None:
            self._save_timer.cancel()

        def _do_save() -> None:
            with self._lock:
                data = self._state.to_dict()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._path)

        self._save_timer = threading.Timer(SNAPSHOT_DEBOUNCE_S, _do_save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _broadcast_many(self, deltas: list[dict[str, Any]]) -> None:
        for d in deltas:
            self._broadcast(d)

    def _broadcast(self, msg: dict[str, Any]) -> None:
        if self._loop is None:
            return
        for q in list(self._subscribers):

            def _put(queue: asyncio.Queue[dict[str, Any]] = q) -> None:
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

            self._loop.call_soon_threadsafe(_put)


_store: StateStore | None = None


def get_state_store() -> StateStore:
    global _store
    if _store is None:
        _store = StateStore()
    return _store


def init_state_store(*, snapshot_path: Path | None = None) -> StateStore:
    global _store
    _store = StateStore(snapshot_path)
    return _store
