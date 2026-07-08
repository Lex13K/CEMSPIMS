"""`[thesis.export]` TOML configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThesisExportConfig:
    legacy_archive: bool


def load_thesis_export_config(config_path: Path) -> ThesisExportConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    te = dict(data.get("thesis", {}).get("export", {}) or {})
    return ThesisExportConfig(legacy_archive=bool(te.get("legacy_archive", False)))
