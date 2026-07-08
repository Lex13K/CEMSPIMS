"""Shared test fixtures (minimal raw CSV layout for ingest + returns_panel)."""

from __future__ import annotations

from pathlib import Path

from mss.io.config import ResolvedConfig


def write_minimal_raw(raw_dir: Path) -> None:
    """WRDS row shape matches columns selected in returns_panel.build_returns_panel."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wrds = """date,permno,permco,shrcd,exchcd,ret,retx,prc,vol,shrout,dlstcd,dlret,dlretx,dlprc,dlpdt,ticker,comnam,ncusip,cusip,secstat
2020-01-02,10001,20001,10,3,0.01,0.01,100.0,1000,1000000,,,,,,ACME,Acme Inc,,,
2020-01-03,10001,20001,10,3,-0.02,-0.02,98.0,1100,1000000,,,,,,ACME,Acme Inc,,,
"""
    (raw_dir / "WRDS.csv").write_text(wrds, encoding="utf-8")
    sp500 = """date,sprtrn
2020-01-02,0.001
2020-01-03,-0.002
"""
    (raw_dir / "snp500_volatility.csv").write_text(sp500, encoding="utf-8")
    vix = """observation_date,VIXCLS
2020-01-02,18.5
2020-01-03,19.2
"""
    (raw_dir / "VIX.csv").write_text(vix, encoding="utf-8")


def paths_toml(
    *,
    raw: Path,
    shared_interim: Path,
    run_interim: Path,
    processed: Path,
) -> str:
    return (
        f'[paths]\nraw = "{raw.as_posix()}"\n'
        f'shared_interim = "{shared_interim.as_posix()}"\n'
        f'interim = "{run_interim.as_posix()}"\n'
        f'processed = "{processed.as_posix()}"\n'
    )


def make_resolved_config(
    tmp: Path,
    *,
    run_id: str = "t",
    raw_dir: Path | None = None,
    shared_interim_dir: Path | None = None,
    run_interim_dir: Path | None = None,
    processed_dir: Path | None = None,
    source_config_path: Path | None = None,
) -> ResolvedConfig:
    raw = raw_dir or (tmp / "raw")
    shared = shared_interim_dir or (tmp / "shared_interim")
    run_i = run_interim_dir or (tmp / "interim")
    proc = processed_dir or (tmp / "processed")
    cfg_path = source_config_path or (tmp / "cfg.toml")
    return ResolvedConfig(
        run_id=run_id,
        project_root=tmp,
        raw_dir=raw,
        shared_interim_dir=shared,
        run_interim_dir=run_i,
        processed_dir=proc,
        run_data_dir=run_i.parent,
        manifest_path=run_i.parent / "run_manifest.json",
        source_config_path=cfg_path.resolve(),
    )
