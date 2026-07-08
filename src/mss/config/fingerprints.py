"""Per-section TOML fingerprints for stale-artifact detection."""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mss.io.config import ResolvedConfig
from mss.run.paths import graphs_dir

GRAPH_SECTION = "graph"
DATASET_SECTION = "dataset"
MODEL_TRAIN_SECTION = "model.train"
MODEL_EVALUATE_SECTION = "model.evaluate"

GRAPH_FINGERPRINT_FILENAME = "config_fingerprint.json"
DATASET_FINGERPRINT_FILENAME = "config_fingerprint.json"
EVALUATE_FINGERPRINT_FILENAME = "evaluate_config_fingerprint.json"


def fingerprint_toml_table(table: dict[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of a TOML table."""
    payload = json.dumps(table, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nested_table(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return dict(node) if isinstance(node, dict) else {}


def fingerprint_config_section(config_path: Path, *keys: str) -> str:
    """Hash a nested TOML section (e.g. ``('graph',)`` or ``('model', 'train')``)."""
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return fingerprint_toml_table(_nested_table(data, *keys))


def graph_fingerprint_path(cfg: ResolvedConfig) -> Path:
    return graphs_dir(cfg) / GRAPH_FINGERPRINT_FILENAME


def dataset_fingerprint_path(cfg: ResolvedConfig) -> Path:
    return cfg.run_interim_dir / "dataset" / DATASET_FINGERPRINT_FILENAME


def evaluate_fingerprint_path(cfg: ResolvedConfig) -> Path:
    return Path(cfg.processed_dir) / EVALUATE_FINGERPRINT_FILENAME


def current_graph_fingerprint(config_path: Path) -> str:
    return fingerprint_config_section(config_path, "graph")


def current_dataset_fingerprint(config_path: Path) -> str:
    return fingerprint_config_section(config_path, "dataset")


def current_model_train_fingerprint(config_path: Path) -> str:
    return fingerprint_config_section(config_path, "model", "train")


def current_evaluate_fingerprint(config_path: Path) -> str:
    return fingerprint_config_section(config_path, "model", "evaluate")


def read_fingerprint_sidecar(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fp = data.get("fingerprint")
    return str(fp) if fp else None


def write_fingerprint_sidecar(
    path: Path,
    *,
    section: str,
    fingerprint: str,
    config_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "section": section,
        "fingerprint": fingerprint,
        "config_path": str(config_path.resolve()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stored_fingerprint_matches(
    config_path: Path,
    *,
    section: str,
    sidecar_path: Path,
    current_fingerprint: str | None = None,
) -> bool:
    """True when sidecar exists and matches the current section hash."""
    stored = read_fingerprint_sidecar(sidecar_path)
    if stored is None:
        return False
    current = current_fingerprint
    if current is None:
        if section == GRAPH_SECTION:
            current = current_graph_fingerprint(config_path)
        elif section == DATASET_SECTION:
            current = current_dataset_fingerprint(config_path)
        elif section == MODEL_TRAIN_SECTION:
            current = current_model_train_fingerprint(config_path)
        elif section == MODEL_EVALUATE_SECTION:
            current = current_evaluate_fingerprint(config_path)
        else:
            return False
    return stored == current


def write_graph_fingerprint_sidecar(cfg: ResolvedConfig) -> None:
    fp = current_graph_fingerprint(cfg.source_config_path)
    write_fingerprint_sidecar(
        graph_fingerprint_path(cfg),
        section=GRAPH_SECTION,
        fingerprint=fp,
        config_path=cfg.source_config_path,
    )


def write_dataset_fingerprint_sidecar(cfg: ResolvedConfig) -> None:
    fp = current_dataset_fingerprint(cfg.source_config_path)
    write_fingerprint_sidecar(
        dataset_fingerprint_path(cfg),
        section=DATASET_SECTION,
        fingerprint=fp,
        config_path=cfg.source_config_path,
    )


def write_evaluate_fingerprint_sidecar(cfg: ResolvedConfig) -> None:
    fp = current_evaluate_fingerprint(cfg.source_config_path)
    write_fingerprint_sidecar(
        evaluate_fingerprint_path(cfg),
        section=MODEL_EVALUATE_SECTION,
        fingerprint=fp,
        config_path=cfg.source_config_path,
    )


def read_train_fingerprint_from_final_metrics(cfg: ResolvedConfig) -> str | None:
    from mss.model_train.checks import final_metrics_path

    path = final_metrics_path(cfg)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fp = data.get("config_fingerprint")
    return str(fp) if fp else None


def train_fingerprint_matches(cfg: ResolvedConfig) -> bool:
    stored = read_train_fingerprint_from_final_metrics(cfg)
    if stored is None:
        return False
    return stored == current_model_train_fingerprint(cfg.source_config_path)


def graph_fingerprint_matches(cfg: ResolvedConfig) -> bool:
    return stored_fingerprint_matches(
        cfg.source_config_path,
        section=GRAPH_SECTION,
        sidecar_path=graph_fingerprint_path(cfg),
    )


def dataset_fingerprint_matches(cfg: ResolvedConfig) -> bool:
    return stored_fingerprint_matches(
        cfg.source_config_path,
        section=DATASET_SECTION,
        sidecar_path=dataset_fingerprint_path(cfg),
    )


def evaluate_fingerprint_matches(cfg: ResolvedConfig) -> bool:
    return stored_fingerprint_matches(
        cfg.source_config_path,
        section=MODEL_EVALUATE_SECTION,
        sidecar_path=evaluate_fingerprint_path(cfg),
    )
