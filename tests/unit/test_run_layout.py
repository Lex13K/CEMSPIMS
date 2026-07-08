"""ResolvedConfig path layout for shared + per-run interim."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.io.config import load_resolved_config


def test_shared_and_run_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg_path = configs / "demo.toml"
    cfg_path.write_text(
        "[paths]\n"
        'raw = "data/shared/raw"\n'
        'shared_interim = "data/shared/interim"\n'
        'interim = "data/{run_id}/interim"\n'
        'processed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    cfg = load_resolved_config(cfg_path, "demo")
    assert cfg.raw_dir == (tmp_path / "data" / "shared" / "raw").resolve()
    assert cfg.shared_interim_dir == (tmp_path / "data" / "shared" / "interim").resolve()
    assert cfg.run_interim_dir == (tmp_path / "data" / "demo" / "interim").resolve()
    assert cfg.interim_dir == cfg.run_interim_dir
    assert cfg.manifest_path == (tmp_path / "data" / "demo" / "run_manifest.json").resolve()
