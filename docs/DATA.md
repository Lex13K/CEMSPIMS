# Data requirements

Raw inputs are **not included** in this repository (size and WRDS licensing). You must obtain them separately and place them under `data/raw/`.

## Required files

| File | Description |
|------|-------------|
| `WRDS.csv` | CRSP daily stock extract (permno, date, returns, identifiers, delisting fields) |
| `snp500_volatility.csv` | S&P 500 daily series for realized-volatility target construction |
| `VIX.csv` | VIX level (e.g. FRED `VIXCLS` or a column named `vix`) |

Expected layout:

```
data/
  raw/
    WRDS.csv
    snp500_volatility.csv
    VIX.csv
  <run_id>/
    interim/     # created by pipeline (gitignored)
    processed/   # created by pipeline (gitignored)
```

## WRDS / CRSP

The ingest step (`data.prepare` → `ingest`) expects a WRDS-style CRSP daily export. Column names and types are normalized in `src/mss/data/ingest.py`. Use `scripts/list_raw_columns.py` to inspect an unfamiliar CSV header.

**Do not commit** raw WRDS extracts or credentials to a public repository.

## Validation

By default, ingest runs a CSV ↔ Parquet summary check. Skip with:

```bash
python scripts/run.py run --run default --pipeline data.prepare --skip-validate
```

## Outputs

All modeling artifacts are written under `data/<run_id>/` per run config (`configs/<run_id>.toml`). Only configuration files are version-controlled; run outputs stay local.
