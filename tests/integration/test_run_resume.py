"""Integration: second run without --overwrite skips completed steps."""

from __future__ import annotations

from mss.cli import main

from tests.conftest import paths_toml, write_minimal_raw


def test_second_run_skips_all_steps(tmp_path, capsys) -> None:
    raw = tmp_path / "raw"
    shared_interim = tmp_path / "shared_interim"
    run_interim = tmp_path / "run_interim"
    processed = tmp_path / "processed"
    write_minimal_raw(raw)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "resume1.toml"
    cfg.write_text(
        paths_toml(
            raw=raw,
            shared_interim=shared_interim,
            run_interim=run_interim,
            processed=processed,
        ),
        encoding="utf-8",
    )
    args = [
        "run",
        "--configs-dir",
        str(configs_dir),
        "--run",
        "resume1",
        "--pipeline",
        "data.prepare",
        "--skip-validate",
    ]
    assert main(args) == 0
    out1 = capsys.readouterr().out
    assert "skipped" not in out1.lower()

    capsys.readouterr()
    assert main(args) == 0
    out2 = capsys.readouterr().out
    assert out2.count("skipped (outputs already present)") == 3


def test_overwrite_second_run_runs_steps_again(tmp_path, capsys) -> None:
    raw = tmp_path / "raw"
    shared_interim = tmp_path / "shared_interim"
    run_interim = tmp_path / "run_interim"
    processed = tmp_path / "processed"
    write_minimal_raw(raw)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "resume2.toml"
    cfg.write_text(
        paths_toml(
            raw=raw,
            shared_interim=shared_interim,
            run_interim=run_interim,
            processed=processed,
        ),
        encoding="utf-8",
    )
    base = [
        "run",
        "--configs-dir",
        str(configs_dir),
        "--run",
        "resume2",
        "--pipeline",
        "data.prepare",
        "--skip-validate",
    ]
    assert main(base) == 0
    capsys.readouterr()
    assert main(base + ["--overwrite"]) == 0
    out = capsys.readouterr().out
    assert "skipped" not in out.lower()
