"""Tests for mss.run_copy.copy_run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mss.run_copy import copy_run


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make config resolution use tmp_path as project root."""
    monkeypatch.setattr("mss.io.paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.io.config.project_root", lambda: tmp_path)
    monkeypatch.setattr("mss.run_copy.project_root", lambda: tmp_path)
    return tmp_path


def test_copy_run_rewrites_json_interim_and_processed_prefixes(fake_root: Path) -> None:
    configs = fake_root / "configs"
    configs.mkdir(parents=True)
    src_toml = configs / "src_run.toml"
    src_toml.write_text(
        "[paths]\n"
        'raw = "data/raw"\n'
        'interim = "data/{run_id}/interim"\n'
        'processed = "data/{run_id}/processed"\n',
        encoding="utf-8",
    )
    src_interim = fake_root / "data" / "src_run" / "interim"
    src_proc = fake_root / "data" / "src_run" / "processed"
    (src_interim / "dataset").mkdir(parents=True)
    src_proc.mkdir(parents=True)

    old_i = src_interim.resolve().as_posix()
    old_p = src_proc.resolve().as_posix()
    manifest = {
        "splits_path": f"{old_i}/dataset/splits.parquet",
        "note": "nested",
        "extra": [f"{old_p}/forecasts.parquet"],
    }
    (src_interim / "dataset" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (src_interim / "ingest_manifest.json").write_text(
        json.dumps({"csv_path": f"{old_i}/x.csv"}), encoding="utf-8"
    )

    copy_run("src_run", "dst_run", configs_dir=configs)

    dst_interim = fake_root / "data" / "dst_run" / "interim"
    dst_proc = fake_root / "data" / "dst_run" / "processed"
    new_i = dst_interim.resolve().as_posix()
    new_p = dst_proc.resolve().as_posix()

    out = json.loads((dst_interim / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert out["splits_path"] == f"{new_i}/dataset/splits.parquet"
    assert out["extra"] == [f"{new_p}/forecasts.parquet"]

    ingest = json.loads((dst_interim / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert ingest["csv_path"] == f"{new_i}/x.csv"

    assert (configs / "dst_run.toml").is_file()


def test_copy_run_fails_if_dst_config_exists(fake_root: Path) -> None:
    configs = fake_root / "configs"
    configs.mkdir(parents=True)
    for name in ("src_run.toml", "dst_run.toml"):
        (configs / name).write_text(
            "[paths]\nraw = \"data/raw\"\ninterim = \"data/{run_id}/interim\"\n"
            'processed = "data/{run_id}/processed"\n',
            encoding="utf-8",
        )
    src_interim = fake_root / "data" / "src_run" / "interim"
    src_proc = fake_root / "data" / "src_run" / "processed"
    src_interim.mkdir(parents=True)
    src_proc.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        copy_run("src_run", "dst_run", configs_dir=configs)
