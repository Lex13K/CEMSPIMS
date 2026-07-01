# Architecture

Modular pipeline for forecasting forward **30-calendar-day** realized volatility (`log_rv_fwd_30cal`) from equity correlation graphs. VIX is benchmark-only.

## Entry points

| Path | Role |
|------|------|
| `scripts/run.py` | CLI wrapper |
| `src/mss/cli.py` | `run`, `validate-config`, `list-pipelines`, `branch`, `list-runs`, `run-tree`, `diff-config`, `compare-runs`, `app`; `copy` (deprecated) |
| `app/server/` | FastAPI experiment graph API + job runner |
| `app/web/` | React Flow frontend |
| `src/mss/pipeline/orchestrator.py` | Pipeline registry and step execution |

## Package map

| Module | Responsibility |
|--------|----------------|
| `mss.io` | Config load, `{run_id}` path resolution |
| `mss.config` | Section fingerprints, `validate-config` warnings, sidecar backfill |
| `mss.run` | Run manifest, branching, registry (`branch`, `list-runs`, `run-tree`) |
| `mss.data` | Ingest, returns panel, targets |
| `mss.calendar` | Trading-day window indexing (`TradingCalendar`, shared by graph + universe SQL) |
| `mss.graph` | Feature dates, universe (`fixed_replace` / `monthly_rebalance`), node features, edges |
| `mss.model_train` | PyG cache, GraphSAGE/GAT (`use_edge_weights`), training loop |
| `mss.dataset` | Splits, train-only scaler, manifest |
| `mss.evaluation` | Scoring, forecast panel, formal H1–H5 tests, plus `H2_benchmark` / `incremental_vs_benchmark` rows |
| `mss.analysis` | Post-train figures (`analysis.summarize`); cross-run comparison (`analysis.compare_runs`) |
| `mss.thesis` | Manuscript exhibit export (`thesis.export`) |
| `mss.pipeline` | Orchestration, artifact completeness checks |

Modules `mss.analysis.h1`, `pairwise`, and `report` were removed in Phase 6 (superseded by `mss.evaluation.hypothesis_tests` and `mss.analysis.compare_runs`).

## Pipelines (canonical order)

1. **`data.prepare`** — `ingest` → `returns_panel` → `targets` (writes **shared** interim)
2. **`graph.prepare`** — `feature_dates` → `universe` → `node_features` → `edges` (per-run interim)
3. **`dataset.package`** — `splits` → `scaler` → `manifest`
4. **`model.cache_graphs`** — `materialize` (PyG `.pt` cache)
5. **`model.train`** — `train` (optional `placebo_retrain` when `[model.evaluate].placebo_retrain = true`)
6. **`model.evaluate`** — `score_splits` → `write_forecast_panel` → `aggregate_test_loss` → `write_summary_table` → `run_hypothesis_tests`
7. **`analysis.summarize`** — `loss_figure`
8. **`thesis.export`** — `build`
9. **`analysis.compare_runs`** — `build` (optional; cross-run comparison)
10. **`app`** — local experiment graph UI (`python scripts/run.py app`; see [`app.md`](app.md), SSE + `sync_app_state.py`)

## Paths (Phase 1 layout)

```
data/
  shared/
    raw/                 # WRDS, VIX, S&P CSVs (not in git)
    interim/             # crsp_parquet, returns_panel, targets, …
  <run_id>/
    run_manifest.json    # parent run, branch step, config path
    interim/             # graph → model_train artifacts
    processed/           # metrics, figures, thesis exhibits
configs/<run_id>.toml
```

- **`data.prepare`** reads/writes `data/shared/` (unless overridden in TOML).
- **`graph.prepare` onward** read shared targets/panel but write under `data/<run_id>/interim/`.
- Branch with `python scripts/run.py branch --parent default --child v2_test --at graph.prepare` to avoid duplicating CRSP parquet.

See [DATA.md](DATA.md) and [data_contracts.md](data_contracts.md) for artifact contracts.

## Invariants

- Scripts contain no business logic.
- Each step reads/writes contract-defined artifacts ([`data_contracts.md`](data_contracts.md)).
- Graph config is re-read from TOML at the start of each `graph.prepare` step.
- Without `--overwrite`, steps skip only when outputs are **semantically complete** and **config fingerprints** match the current TOML section (`mss.config.fingerprints`, `mss.pipeline.completeness`).

## Formal inference

`run_hypothesis_tests` consumes only `processed/forecast_panel.parquet`. Canonical thesis hypotheses **H1–H5** use Newey–West HAC errors (29 lags for a 30-day horizon):

- **H1:** Mincer–Zarnowitz on the GNN forecast
- **H2:** Diebold–Mariano vs **raw VIX** (primary benchmark)
- **H3:** incremental regression conditional on raw VIX
- **H4:** H1–H3 re-run on stress subsamples
- **H5:** Diebold–Mariano vs placebo

Secondary benchmarks add generalized rows: **`H2_benchmark`** (DM vs `calibrated_vix` / `vix_har` / `har`) and **`incremental_vs_benchmark`** (`H3_benchmark`, identified by the `benchmark` column). See root [README.md](../README.md) and [`data_contracts.md`](data_contracts.md).

## Further reading

- [`data_contracts.md`](data_contracts.md) — schemas, resume semantics
- [`experiments.md`](experiments.md) — branching and ablation matrix
