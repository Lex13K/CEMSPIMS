"""Delete a materialized run from data/ and configs/."""

from __future__ import annotations

import shutil
from pathlib import Path

from mss.io.config import sanitize_run_id
from mss.io.paths import project_root
from mss.run.registry import _load_record, discover_run_ids, is_materialized_run, run_data_dir


def delete_run(run_id: str, *, configs_dir: Path | None = None) -> None:
    """Remove run data directory and config file."""
    rid = sanitize_run_id(run_id)
    root = project_root()
    cdir = (configs_dir or (root / "configs")).resolve()

    if not is_materialized_run(rid, cdir):
        raise FileNotFoundError(f"run not found: {rid}")

    children = [
        other
        for other in discover_run_ids(cdir)
        if other != rid and _parent_of(other, cdir) == rid
    ]
    if children:
        names = ", ".join(sorted(children))
        raise ValueError(f"cannot delete {rid}: delete child runs first ({names})")

    data_dir = run_data_dir(rid)
    if data_dir.is_dir():
        shutil.rmtree(data_dir)

    cfg_path = cdir / f"{rid}.toml"
    if cfg_path.is_file():
        cfg_path.unlink()


def _parent_of(run_id: str, configs_dir: Path) -> str | None:
    rec = _load_record(run_id, configs_dir)
    return rec.parent_run_id if rec else None
