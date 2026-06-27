import tomllib
from pathlib import Path

from mss.cli import main


def test_validate_config_writes_resolved_paths(tmp_path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "myrun.toml"
    raw = tmp_path / "r"
    inter = tmp_path / "i"
    proc = tmp_path / "p"
    cfg.write_text(
        f'[paths]\nraw = "{raw.as_posix()}"\n'
        f'interim = "{inter.as_posix()}"\n'
        f'processed = "{proc.as_posix()}"\n',
        encoding="utf-8",
    )
    code = main(
        ["validate-config", "--configs-dir", str(configs_dir), "--run", "myrun"]
    )
    assert code == 0


def test_default_config_loads() -> None:
    """default.toml paths are relative; load via real default file in repo."""
    root = Path(__file__).resolve().parents[2]
    default_toml = root / "configs" / "default.toml"
    assert default_toml.is_file()
    data = tomllib.loads(default_toml.read_text(encoding="utf-8"))
    assert "paths" in data
    assert "{run_id}" in data["paths"].get("interim", "")
    assert "{run_id}" in data["paths"].get("processed", "")
