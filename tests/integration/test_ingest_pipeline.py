import json

import pandas as pd

from mss.data.ingest import IngestPaths, ingest_raw_to_parquet, validate_ingest
from mss.data.returns_panel import build_returns_panel, check_returns_panel
from mss.data.targets import build_targets, check_targets

from tests.conftest import write_minimal_raw


def test_ingest_raw_to_parquet_and_validate(tmp_path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_minimal_raw(raw)
    paths = IngestPaths(raw_dir=raw, interim_dir=interim)

    out = ingest_raw_to_parquet(paths, overwrite=False)
    assert out.manifest_path.is_file()
    assert out.crsp_parquet_dir.is_dir()
    assert list((out.crsp_parquet_dir / "year=2020").glob("*.parquet"))
    assert out.sp500_parquet_path.is_file()
    assert out.vix_parquet_path.is_file()

    with open(out.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["stage"] == "01_ingest_raw_data"
    assert manifest["crsp"]["years_written"] == [2020]

    validate_ingest(paths, atol=1e-8, rtol=1e-8)

    df_sp = pd.read_parquet(out.sp500_parquet_path)
    assert "date" in df_sp.columns and "sprtrn" in df_sp.columns

    rp_path = interim / "returns_panel.parquet"
    build_returns_panel(out.crsp_parquet_dir, rp_path, overwrite=False)
    chk = check_returns_panel(rp_path)
    assert chk["passed"]
    df_rp = pd.read_parquet(rp_path)
    assert "date" in df_rp.columns and "permno" in df_rp.columns and "ret_used" in df_rp.columns

    interim.mkdir(parents=True, exist_ok=True)
    tg_path = interim / "targets.parquet"
    tg_man = interim / "targets_manifest.json"
    build_targets(
        out.sp500_parquet_path,
        tg_path,
        tg_man,
        vix_parquet_path=out.vix_parquet_path,
        overwrite=False,
    )
    assert check_targets(tg_path)["passed"]
    assert "rv_fwd_30" in pd.read_parquet(tg_path).columns
