"""Background pipeline job runner for the experiment app."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mss.app import events as app_events
from mss.app.state_store import get_state_store
from mss.io.paths import project_root
from mss.run.events_emit import event_to_app_event, parse_event_line
from mss.run.registry import is_materialized_run

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class JobRecord:
    job_id: str
    run_id: str
    status: JobStatus
    created_at: str
    pipelines: list[str] | None = None
    overwrite: list[str] | None = None
    pid: int | None = None
    return_code: int | None = None
    log_path: str = ""
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRunner:
    def __init__(self, jobs_dir: Path | None = None) -> None:
        self._jobs_dir = jobs_dir or (project_root() / "data" / ".app_jobs")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def _job_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.log"

    def _save(self, rec: JobRecord) -> None:
        self._job_path(rec.job_id).write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

    def load(self, job_id: str) -> JobRecord | None:
        p = self._job_path(job_id)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return JobRecord(**{k: data[k] for k in JobRecord.__dataclass_fields__ if k in data})
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _purge_job_files(self, job_id: str) -> None:
        self._job_path(job_id).unlink(missing_ok=True)
        self._log_path(job_id).unlink(missing_ok=True)

    def reconcile_job(self, rec: JobRecord, *, configs_dir: Path) -> JobRecord | None:
        if is_materialized_run(rec.run_id, configs_dir):
            return rec
        if rec.status == "running":
            rec.status = "failed"
            rec.error = rec.error or f"run no longer exists: {rec.run_id}"
            self._save(rec)
            return rec
        self._purge_job_files(rec.job_id)
        return None

    def prune_stale_jobs(self, *, configs_dir: Path) -> None:
        for p in self._jobs_dir.glob("*.json"):
            rec = self.load(p.stem)
            if rec is None:
                continue
            if is_materialized_run(rec.run_id, configs_dir):
                continue
            if rec.status == "running":
                continue
            self._purge_job_files(rec.job_id)

    def list_jobs(self, *, limit: int = 20) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for p in sorted(self._jobs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            rec = self.load(p.stem)
            if rec:
                jobs.append(rec)
            if len(jobs) >= limit:
                break
        return jobs

    def start(
        self,
        run_id: str,
        *,
        pipelines: list[str] | None = None,
        overwrite: list[str] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        log_path = self._log_path(job_id)
        rec = JobRecord(
            job_id=job_id,
            run_id=run_id,
            status="pending",
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            pipelines=pipelines,
            overwrite=overwrite,
            log_path=str(log_path),
        )
        self._save(rec)

        cmd = [
            sys.executable,
            str(project_root() / "scripts" / "run.py"),
            "run",
            "--run",
            run_id,
        ]
        if pipelines:
            for p in pipelines:
                cmd.extend(["--pipeline", p])
        if overwrite is not None:
            if len(overwrite) == 0:
                cmd.append("--overwrite")
            else:
                cmd.append("--overwrite")
                cmd.extend(overwrite)

        log_path.write_text("", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(project_root()),
        )
        with self._lock:
            self._processes[job_id] = proc
        rec.pid = proc.pid
        rec.status = "running"
        self._save(rec)

        store = get_state_store()
        store.apply(
            app_events.job_started(
                job_id=job_id,
                run_id=run_id,
                status="running",
                created_at=rec.created_at,
                pipelines=pipelines,
                overwrite=overwrite,
            )
        )

        def _reader() -> None:
            tail: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                tail.append(line.rstrip("\n"))
                if len(tail) > 500:
                    tail.pop(0)

                ev_data = parse_event_line(line)
                if ev_data is not None:
                    app_ev = event_to_app_event(ev_data)
                    if app_ev is not None:
                        store.apply(app_ev)
                else:
                    store.apply(
                        app_events.step_log_line(
                            job_id=job_id,
                            run_id=run_id,
                            line=line.rstrip("\n"),
                        )
                    )

                r = self.load(job_id)
                if r:
                    r.log_tail = tail.copy()
                    self._save(r)
            code = proc.wait()
            r = self.load(job_id)
            if r:
                r.return_code = code
                r.status = "completed" if code == 0 else "failed"
                if code != 0 and r.log_tail:
                    r.error = r.log_tail[-1]
                self._save(r)
                store.apply(
                    app_events.job_finished(
                        job_id=job_id,
                        run_id=run_id,
                        status=r.status,
                        return_code=code,
                        error=r.error,
                    )
                )
            with self._lock:
                self._processes.pop(job_id, None)

        threading.Thread(target=_reader, daemon=True).start()
        return rec

    def cancel(self, job_id: str) -> JobRecord:
        rec = self.load(job_id)
        if rec is None:
            raise ValueError(f"job not found: {job_id}")
        with self._lock:
            proc = self._processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        rec.status = "cancelled"
        self._save(rec)
        get_state_store().apply(
            app_events.job_finished(
                job_id=job_id,
                run_id=rec.run_id,
                status="cancelled",
            )
        )
        return rec

    def read_log(self, job_id: str, *, tail: int = 200) -> str:
        p = self._log_path(job_id)
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])


_runner: JobRunner | None = None


def get_job_runner() -> JobRunner:
    global _runner
    if _runner is None:
        _runner = JobRunner()
    return _runner
