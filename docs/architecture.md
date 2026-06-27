# Architecture

Modular pipeline for forecasting forward 30-day realized volatility from equity correlation graphs. VIX is benchmark-only.

## Entry points

| Path | Role |
|------|------|
| `scripts/run.py` | CLI wrapper |
| `src/mss/cli.py` | `run`, `validate-config`, `list-pipelines`, `copy` |
| `src/mss/pipeline/orchestrator.py` | Pipeline registry and step execution |

## Package map

| Module | Responsibility |
|--------|----------------|
| `mss.io` | Config load, `{run_id}` path resolution |
| `mss.data` | Ingest, returns panel, targets |
| `mss.graph` | Feature dates, universe, node features, edges |
| `mss.dataset` | Splits, train-only scaler, manifest |
| `mss.model_train` | Graph cache, GraphSAGE/GAT training |
| `mss.evaluation` | Scoring, forecast panel, H1–H4 hypothesis tests |
| `mss.analysis` | Post-train figures (`analysis.summarize`) |
| `mss.thesis` | Manuscript exhibit export (`thesis.export`) |
| `mss.pipeline` | Orchestration, artifact completeness checks |

Modules `mss.analysis.pairwise`, `h1`, and `report` are retained but **not** wired into the orchestrator.

## Pipelines (canonical order)

1. **`data.prepare`** — `ingest` → `returns_panel` → `targets`
2. **`graph.prepare`** — `feature_dates` → `universe` → `node_features` → `edges`
3. **`dataset.package`** — `splits` → `scaler` → `manifest`
4. **`model.cache_graphs`** — `materialize` (PyG `.pt` cache)
5. **`model.train`** — `train`
6. **`model.evaluate`** — `score_splits` → `write_forecast_panel` → `aggregate_test_loss` → `write_summary_table` → `run_hypothesis_tests`
7. **`analysis.summarize`** — `loss_figure`
8. **`thesis.export`** — `build`

## Paths

- **Shared:** `data/raw/`
- **Per run:** `data/<run_id>/interim/` (modeling tables), `data/<run_id>/processed/` (metrics, figures, thesis exhibits)
- **Config:** `configs/<run_id>.toml`

## Invariants

- Scripts contain no business logic.
- Each step reads/writes contract-defined artifacts ([`data_contracts.md`](data_contracts.md)).
- Graph config is re-read from TOML at the start of each `graph.prepare` step.
- Without `--overwrite`, steps skip only when outputs are **semantically complete** (`mss.pipeline.completeness`).

## Formal inference

`run_hypothesis_tests` consumes only `processed/forecast_panel.parquet`. Hypothesis IDs H1–H4 map to Mincer–Zarnowitz, Diebold–Mariano, and incremental regressions with Newey–West HAC errors (29 lags for a 30-day horizon). See root [README.md](../README.md) for economic interpretations.

## Further reading

- [`data_contracts.md`](data_contracts.md) — schemas, resume semantics
