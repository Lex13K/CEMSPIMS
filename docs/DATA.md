# Data requirements

Raw inputs are **not included** in this repository (size and WRDS licensing). You must obtain them separately and place them under `data/shared/raw/`.

## Required files

| File | Description |
|------|-------------|
| `WRDS.csv` | CRSP daily stock extract (permno, date, returns, identifiers, delisting fields) |
| `snp500_volatility.csv` | S&P 500 daily series for realized-volatility target construction |
| `VIX.csv` | VIX level (e.g. FRED `VIXCLS` or a column named `vix`) |

Expected layout (Phase 1):

```
data/
  shared/
    raw/              # WRDS.csv, snp500_volatility.csv, VIX.csv (sync from OneDrive here)
    interim/          # shared prepare: crsp_parquet, returns_panel, targets, …
  <run_id>/
    run_manifest.json # lineage + pointers to shared layer
    interim/          # per-run graph → model artifacts
    processed/        # evaluation summaries, figures, thesis exhibits
```

## WRDS / CRSP

The ingest step (`data.prepare` → `ingest`) expects a WRDS-style CRSP daily export. Column names and types are normalized in `src/mss/data/ingest.py`. Use `scripts/list_raw_columns.py` to inspect an unfamiliar CSV header.

**Do not commit** raw WRDS extracts or credentials to a public repository.

### WRDS / CRSP column usage

`ingest_crsp_wrds_to_parquet` writes **every column** from `WRDS.csv` into year-partitioned `crsp_parquet/` shards. Known names get typed casts (`permno`, `ret`, `retx`, `prc`, `vol`, `shrout`, `dlret`, delisting codes, etc.); other columns are kept as strings.

**`returns_panel.parquet`** (shared) selects a subset and is the main gate into the graph pipeline:

| Column / field | Used for |
|----------------|----------|
| `date`, `permno` | Spine, joins, universe membership |
| `ret_used` (`retx` or `ret` per config) | Returns panel, edge correlations, default node features |
| `ret_delisted` | Optional (`apply_delisting_adjustment=true`); set `[graph].ret_col` to use |
| `prc`, `vol`, `shrout` | Market cap (`abs(prc)*shrout`) for top-N universe; optional log volume & turnover features |
| `shrcd`, `exchcd` | Filters (`common_shares_only`, `major_exchanges_only`) |
| `dlstcd`, `dlret`, `dlretx`, `dlprc`, `dlpdt` | Delisting adjustment when enabled |
| `permco`, `ticker`, `comnam`, `ncusip`, `cusip`, `secstat` | Carried in panel; **not** used as GNN inputs today |

**Downstream graph (default config):**

| Stage | Columns consumed |
|-------|------------------|
| Universe | `date`, `permno`, `ret_used`, `prc`, `shrout` |
| Edges | `ret_used` rolling correlation |
| Node features | `ret_used` → `rolling_mean`, `rolling_vol` only (`default.toml`) |
| Optional features | `prc`, `vol`, `shrout` when `include_log_dollar_volume` / `include_turnover` are true |

**Typically unused for modeling** (stored but not in hot path): industry codes (`siccd`, `hsiccd`), most metadata strings, unused return variants (`ret` if `retx` selected), and any extra WRDS export fields (fundamentals, bid-ask, etc.) unless you add ingest + feature logic.

To use more WRDS content: extend `returns_panel` / `node_features`, add TOML flags, branch a new run, rerun `data.prepare` (if panel changes) or `graph.prepare` onward.

## Validation

By default, ingest runs a CSV ↔ Parquet summary check. Skip with:

```bash
python scripts/run.py run --run default --pipeline data.prepare --skip-validate
```

## Outputs

- **Shared prepare** (`data/shared/interim/`) is written once by `data.prepare` and reused by all runs.
- **Per-run** artifacts live under `data/<run_id>/` per config (`configs/<run_id>.toml`).
- Each run may have `run_manifest.json` recording parent run and branch step (see `python scripts/run.py list-runs`).

## Branching experiments

```bash
python scripts/run.py branch --parent default --child v2_test --at graph.prepare
```

The child inherits shared prepare outputs; only graph+downstream artifacts are created under `data/v2_test/`. Edit `configs/v2_test.toml`, then rerun with `--overwrite` from the changed step.

## Stale shared interim

If `data/shared/interim/` contains tiny test artifacts (e.g. from pytest writing to the project tree), `data.prepare` may skip incorrectly. Fix with:

```bash
python scripts/run.py run --run default --overwrite data.prepare
```

Or a full `--overwrite` when rebuilding from scratch.

## Full runtime reset (v2 default from scratch)

To wipe generated run outputs and shared interim while keeping raw inputs:

1. Preserve `data/shared/raw/` (and `data/shared/` layout).
2. Remove `data/shared/interim/`, all `data/<run_id>/` folders, `data/.app_state/`, and `data/.app_jobs/`.
3. Keep `configs/default.toml` as the canonical config (`configs/scale_gpu.toml` for GPU scale).

Then validate and run the full pipeline:

```bash
python scripts/run.py validate-config --run default
python scripts/run.py run --run default
```

If the experiment graph app was open, restart it or run `python scripts/sync_app_state.py --reload` after the first pipeline steps.

## Optional `[data.returns_panel]` keys

Loaded by `mss.data.config.load_returns_panel_config` (defaults apply when the section is omitted):

| Key | Default | Description |
|-----|---------|-------------|
| `use_ret` | `"retx"` | Base return column from CRSP shards (`"ret"` or `"retx"`) written as `ret_used`. |
| `apply_delisting_adjustment` | `false` | When `true`, writes `ret_delisted` using CRSP delisting merge on `dlstcd` dates. |
| `common_shares_only` | `true` | Filter `shrcd IN (10, 11)`. |
| `major_exchanges_only` | `false` | Filter `exchcd IN (1, 2, 3)` when enabled. |

Changing this section requires rerunning `data.prepare` (returns panel step). Graph `[graph].ret_col` must match an available column (`ret_used` or `ret_delisted`).
