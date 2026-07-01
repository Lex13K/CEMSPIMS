"""Copy one run's config + interim/processed to a new run id; rewrite absolute path prefixes in JSON."""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root


def _config_path_for_run(configs_dir: Path, run_id: str) -> Path:
    return (configs_dir / f"{run_id}.toml").resolve()


def _rewrite_string_paths(
    s: str,
    old_prefix_posix: str,
    new_prefix_posix: str,
    old_prefix_native: str,
    new_prefix_native: str,
) -> str:
    if s.startswith(old_prefix_posix):
        return new_prefix_posix + s[len(old_prefix_posix) :]
    if s.startswith(old_prefix_native):
        return new_prefix_native + s[len(old_prefix_native) :]
    return s


def _rewrite_json_obj(
    obj: Any,
    old_prefix_posix: str,
    new_prefix_posix: str,
    old_prefix_native: str,
    new_prefix_native: str,
) -> Any:
    if isinstance(obj, dict):
        return {
            k: _rewrite_json_obj(
                v,
                old_prefix_posix,
                new_prefix_posix,
                old_prefix_native,
                new_prefix_native,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _rewrite_json_obj(
                x,
                old_prefix_posix,
                new_prefix_posix,
                old_prefix_native,
                new_prefix_native,
            )
            for x in obj
        ]
    if isinstance(obj, str):
        return _rewrite_string_paths(
            obj,
            old_prefix_posix,
            new_prefix_posix,
            old_prefix_native,
            new_prefix_native,
        )
    return obj


def _rewrite_json_files_under(
    root: Path,
    old_interim: Path,
    new_interim: Path,
    old_processed: Path,
    new_processed: Path,
) -> None:
    if not root.is_dir():
        return
    old_i_pos = old_interim.resolve().as_posix()
    new_i_pos = new_interim.resolve().as_posix()
    old_i_nat = str(old_interim.resolve())
    new_i_nat = str(new_interim.resolve())
    old_p_pos = old_processed.resolve().as_posix()
    new_p_pos = new_processed.resolve().as_posix()
    old_p_nat = str(old_processed.resolve())
    new_p_nat = str(new_processed.resolve())

    for path in root.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        out = _rewrite_json_obj(data, old_i_pos, new_i_pos, old_i_nat, new_i_nat)
        out = _rewrite_json_obj(out, old_p_pos, new_p_pos, old_p_nat, new_p_nat)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def copy_run(
    src_run: str,
    dst_run: str,
    *,
    configs_dir: Path | None = None,
) -> None:
    """
    Copy configs/<src>.toml -> configs/<dst>.toml and data/<src>/{interim,processed}
    -> data/<dst>/{interim,processed}. Fails if destination config or data dirs exist.
    Rewrites absolute interim/processed path prefixes inside JSON under copied trees.

    .. deprecated:: Phase 1
        Prefer ``mss.run.branch.branch_run`` for experiment branching.
    """
    warnings.warn(
        "copy_run is deprecated; use `python scripts/run.py branch` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "Warning: `copy` duplicates full interim trees. "
        "Prefer `branch` for shared prepare + lineage.",
        file=sys.stderr,
    )
    src_id = sanitize_run_id(src_run)
    dst_id = sanitize_run_id(dst_run)
    if src_id == dst_id:
        raise ValueError("source and destination run ids must differ")

    root = project_root()
    cdir = (configs_dir or (root / "configs")).resolve()
    src_cfg = _config_path_for_run(cdir, src_id)
    dst_cfg = _config_path_for_run(cdir, dst_id)

    if not src_cfg.is_file():
        raise FileNotFoundError(f"Source config not found: {src_cfg}")
    if dst_cfg.exists():
        raise FileExistsError(f"Destination config already exists (refusing to overwrite): {dst_cfg}")

    cfg_src = load_resolved_config(src_cfg, src_id)
    # Destination config file must exist before load_resolved_config(dst) for path resolution
    shutil.copy2(src_cfg, dst_cfg)
    cfg_dst = load_resolved_config(dst_cfg, dst_id)

    src_interim = cfg_src.interim_dir
    src_processed = cfg_src.processed_dir
    dst_interim = cfg_dst.interim_dir
    dst_processed = cfg_dst.processed_dir

    if not src_interim.is_dir():
        raise FileNotFoundError(f"Source interim directory missing: {src_interim}")
    if not src_processed.is_dir():
        raise FileNotFoundError(f"Source processed directory missing: {src_processed}")
    if dst_interim.exists():
        raise FileExistsError(f"Destination interim already exists: {dst_interim}")
    if dst_processed.exists():
        raise FileExistsError(f"Destination processed already exists: {dst_processed}")

    shutil.copytree(src_interim, dst_interim)
    shutil.copytree(src_processed, dst_processed)

    _rewrite_json_files_under(
        dst_interim,
        src_interim,
        dst_interim,
        src_processed,
        dst_processed,
    )
    _rewrite_json_files_under(
        dst_processed,
        src_interim,
        dst_interim,
        src_processed,
        dst_processed,
    )
