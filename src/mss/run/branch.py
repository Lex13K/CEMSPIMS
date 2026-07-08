"""Branch a new run from a parent at a pipeline boundary."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.run.hashes import sha256_config
from mss.run.manifest import RunManifest, load_run_manifest, save_run_manifest
from mss.run_copy import _rewrite_json_files_under


def _config_path_for_run(configs_dir: Path, run_id: str) -> Path:
    return (configs_dir / f"{run_id}.toml").resolve()


def _pipeline_index(name: str) -> int:
    try:
        return CANONICAL_PIPELINE_ORDER.index(name)
    except ValueError as e:
        known = ", ".join(CANONICAL_PIPELINE_ORDER)
        raise ValueError(f"Unknown branch pipeline {name!r}. Known: {known}") from e


def _inherit_paths(parent_cfg, branch_pipeline: str) -> list[tuple[Path, Path]]:
    """Copy parent outputs strictly *before* the branch pipeline (inputs only)."""
    idx = _pipeline_index(branch_pipeline)
    parent_run = parent_cfg.run_data_dir
    copies: list[tuple[Path, Path]] = []

    if idx > _pipeline_index("graph.prepare"):
        copies.append((parent_run / "interim" / "graphs", Path("interim/graphs")))
        copies.append((parent_run / "interim" / "stage04_dates.parquet", Path("interim/stage04_dates.parquet")))

    if idx > _pipeline_index("dataset.package"):
        copies.append((parent_run / "interim" / "dataset", Path("interim/dataset")))

    if idx > _pipeline_index("model.train"):
        copies.append((parent_run / "interim" / "model_train", Path("interim/model_train")))
        cache = parent_run / "interim" / "model_cache"
        if cache.is_dir():
            copies.append((cache, Path("interim/model_cache")))

    if idx > _pipeline_index("model.evaluate"):
        copies.append((parent_run / "processed", Path("processed")))

    return copies


def _prune_downstream_artifacts(child_data: Path, branch_pipeline: str) -> None:
    """Remove outputs at/after the branch point that must be re-run on the child."""
    idx = _pipeline_index(branch_pipeline)
    if idx <= _pipeline_index("model.train"):
        shutil.rmtree(child_data / "interim" / "model_train", ignore_errors=True)
        shutil.rmtree(child_data / "interim" / "model_cache", ignore_errors=True)
    if idx <= _pipeline_index("model.evaluate"):
        shutil.rmtree(child_data / "processed", ignore_errors=True)


def branch_run(
    parent_run: str,
    child_run: str,
    *,
    at_pipeline: str,
    at_step_id: str | None = None,
    configs_dir: Path | None = None,
) -> Path:
    """
    Create configs/<child>.toml, data/<child>/run_manifest.json, and inherited artifacts.

    Does not copy shared prepare outputs (crsp, returns_panel, targets).
    """
    parent_id = sanitize_run_id(parent_run)
    child_id = sanitize_run_id(child_run)
    if parent_id == child_id:
        raise ValueError("parent and child run ids must differ")

    _pipeline_index(at_pipeline)

    root = project_root()
    cdir = (configs_dir or (root / "configs")).resolve()
    parent_cfg_path = _config_path_for_run(cdir, parent_id)
    child_cfg_path = _config_path_for_run(cdir, child_id)

    if not parent_cfg_path.is_file():
        raise FileNotFoundError(f"Parent config not found: {parent_cfg_path}")
    if child_cfg_path.exists():
        raise FileExistsError(f"Child config already exists: {child_cfg_path}")

    parent_cfg = load_resolved_config(parent_cfg_path, parent_id)
    if not parent_cfg.manifest_path.is_file():
        raise FileNotFoundError(
            f"Parent run manifest missing: {parent_cfg.manifest_path}. "
            "Run the parent pipeline at least once first."
        )
    parent_manifest = load_run_manifest(parent_cfg.manifest_path)

    child_data = root / "data" / child_id
    if child_data.exists():
        raise FileExistsError(f"Child data directory already exists: {child_data}")

    shutil.copy2(parent_cfg_path, child_cfg_path)
    child_cfg = load_resolved_config(child_cfg_path, child_id)

    child_data.mkdir(parents=True)
    (child_data / "interim").mkdir(parents=True, exist_ok=True)

    inherited: list[str] = []
    for src, rel_dst in _inherit_paths(parent_cfg, at_pipeline):
        if not src.exists():
            if rel_dst.name in ("graphs", "dataset", "model_train", "processed"):
                raise FileNotFoundError(
                    f"Cannot branch at {at_pipeline!r}: parent missing {src}"
                )
            continue
        dst = child_data / rel_dst
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        inherited.append(rel_dst.as_posix())

    _prune_downstream_artifacts(child_data, at_pipeline)

    if inherited:
        _rewrite_json_files_under(
            child_data,
            parent_cfg.run_interim_dir,
            child_cfg.run_interim_dir,
            parent_cfg.processed_dir,
            child_cfg.processed_dir,
        )

    child_manifest = RunManifest(
        schema_version=1,
        run_id=child_id,
        parent_run_id=parent_id,
        branch_pipeline=at_pipeline,
        branch_step_id=at_step_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_path=str(child_cfg_path.relative_to(root)).replace("\\", "/"),
        config_sha256=sha256_config(child_cfg_path),
        shared_raw_dir=parent_manifest.shared_raw_dir,
        shared_interim_dir=parent_manifest.shared_interim_dir,
        prepare_artifact_hashes=dict(parent_manifest.prepare_artifact_hashes),
        inherited_artifacts=inherited,
    )
    save_run_manifest(child_cfg.manifest_path, child_manifest)
    return child_cfg_path
