# Controlled experiment matrix

Branch new runs from a parent with **one primary knob** per run id. Do not change universe mode, edge weights, and model width in the same branch.

## Active baseline

**`configs/default.toml` is already the v2 baseline:** `log_rv_fwd_30cal` target, raw/calibrated/HAR VIX benchmarks, weighted edges, six node features, and broad point-in-time mcap universe. Branch from `default` for ablations — do not treat older migration run ids as prerequisites.

## Run lineage (examples)

| Run id | Parent | Primary knob | Rerun from |
|--------|--------|--------------|------------|
| `default` | — | Canonical v2 baseline | — |
| `v2_monthly_univ` | `default` | `universe_mode = monthly_rebalance` | `graph.prepare` |
| `scale_gpu` | `default` | wider model, `top_k = 20`, `placebo_retrain = true` | `graph.prepare` |

Historical note: early v2 migration runs (`v2_calendar_fix`, `v2_weighted_sage`) predated merging calendar target and edge weights into `default`. New work should branch from `default` directly.

## Branch commands

```bash
python scripts/run.py branch --parent default --child v2_monthly_univ --at graph.prepare
python scripts/run.py branch --parent default --child scale_gpu --at graph.prepare
```

## Compare churn (monthly vs fixed)

After `analysis.summarize` on both runs:

```bash
# Compare turnover_rate_mean in:
#   data/default/processed/figures/universe_churn/universe_churn_summary.csv
#   data/v2_monthly_univ/processed/figures/universe_churn/universe_churn_summary.csv
```

`v2_monthly_univ` should show higher `turnover_rate_mean` and non-null `month_boundary_turnover_*` columns.

## Compare weighted vs unweighted GNN (optional ablation)

To ablate edge weights, branch from `default` and set `[model.train].use_edge_weights = false`, then compare:

```bash
python scripts/run.py branch --parent default --child v2_unweighted --at model.train
python scripts/run.py run --run v2_unweighted --pipeline model.train --pipeline model.evaluate
python scripts/run.py compare-runs \
  --anchor default \
  --runs default,v2_unweighted \
  --comparison-id weighted_ablation
```

## Config keys (Phase 4)

| Section | Key | Values |
|---------|-----|--------|
| `[graph.universe]` | `universe_mode` | `fixed_replace`, `monthly_rebalance` |
| `[graph.universe]` | `rebalance_freq` | `daily`, `monthly`, `quarterly` (used when `monthly_rebalance`) |
| `[model.train]` | `use_edge_weights` | `true` (v2 default), `false` (ablation) |
| `[model.train]` | `gat_num_heads` | GAT only (default `1`) |

See [`data_contracts.md`](data_contracts.md) for edge-weight semantics.

## Benchmarks & formal tests (v2)

The active default target is `log_rv_fwd_30cal` (forward 30-calendar-day realized vol). The benchmark
suite is:

- `raw_vix` — raw spot VIX, the primary market-implied benchmark (canonical H2 and H3).
- `calibrated_vix` — train-calibrated (bias-adjusted) VIX; a secondary benchmark, not risk-premium removal.
- `vix_har` — hybrid implied-plus-historical volatility benchmark.
- `har` — standalone realized-vol HAR benchmark.

Formal inference keeps the canonical `H1`, `H2` (DM vs raw VIX), `H3` (incremental vs raw VIX), `H4`
(subsamples), and `H5` (vs placebo). It adds generalized rows distinguished by a `benchmark` column:
`H2_benchmark` (DM vs each secondary benchmark) and `incremental_vs_benchmark` (`H3_benchmark`), which
tests whether the GNN adds information conditional on a given benchmark — deliberately not labeled
"beyond VIX" for non-VIX benchmarks.

Branch from `default` for evaluate-only changes:

```bash
python scripts/run.py branch --parent default --child v2_bench --at model.train
python scripts/run.py run --run v2_bench --pipeline model.train --pipeline model.evaluate
pytest tests/unit/test_hypothesis_tests.py tests/unit/test_forecasts_targets.py -q
python scripts/run.py validate-config --run default
```

### Config keys (benchmarks & tests)

| Section | Key | Values |
|---------|-----|--------|
| `[model.evaluate]` | `hypothesis_dm_benchmarks` | secondary DM benchmarks, default `["calibrated_vix","vix_har","har"]` |
| `[model.evaluate]` | `hypothesis_incremental_benchmarks` | secondary incremental benchmarks, same default |
| `[model.evaluate]` | `placebo_edge_mode` | `zero_attr` (default), `permute_only` (legacy) |
| `[model.evaluate]` | `placebo_retrain` | `false` (default); `true` trains a second model on placebo graphs for H5 |
| `[model.evaluate]` | `fit_vix_log_calibration` | `false`; when `true`, also folds calibration into `raw_vix` (legacy) |
| `[model.evaluate]` | `hypothesis_dm_losses_vs_placebo` | `["qlike", "mse_log"]` (H5) |

## Scale runs (Phase 9)

Use [`configs/scale_gpu.toml`](../configs/scale_gpu.toml) for GPU runs. Keep one primary knob per branch.

### Node-count presets

Set `[graph.universe].n_nodes` and rebuild from `graph.prepare`:

| Preset | `n_nodes` | Suggested `top_k` | Suggested `hidden_channels` |
|--------|-----------|-------------------|-----------------------------|
| baseline | 500 | 10 | 256 |
| medium | 1000 | 20 | 512 |
| large | 1500 | 30 | 768 |

```bash
python scripts/run.py branch --parent default --child scale_1000 --at graph.prepare
# edit configs/scale_1000.toml: n_nodes=1000, top_k=20, hidden_channels=512
python scripts/run.py run --run scale_1000
```

### Optional robustness modes (off by default)

| Section | Key | Values |
|---------|-----|--------|
| `[graph.universe]` | `selection_rule` | `mcap` (canonical), `dollar_volume` (robustness) |
| `[graph.universe]` | `restrict_to_sp500` | `false` (canonical broad mcap), `true` (S&P-membership robustness) |
| `[graph.universe]` | `sp500_membership_path` | path to normalized membership spans parquet (permno, start_date, end_date) |

The broad point-in-time mcap universe remains canonical. `dollar_volume` and `sp500_membership` are
alternative/robustness modes only; do not treat them as the baseline unless promoted explicitly.
