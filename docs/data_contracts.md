# Data and Artifact Contracts

## Purpose

Define non-negotiable artifact interfaces between pipeline steps to prevent drift during refactoring.

Contract fields for every artifact:
- Producer step/module
- Consumer step/module
- Path and format
- Required schema
- Key constraints
- Semantic constraints
- Validation checks

## Resume and skip semantics

Without `--overwrite`, a step is skipped only when its outputs are **semantically complete** (not merely present on disk). Logic lives in **`mss.pipeline.completeness`** and is used by **`mss.pipeline.artifacts`**. Examples: **`targets.parquet`** row count must match **`targets_manifest.json`** `stats.n_rows`; **`stage04_dates.parquet`** must match the recomputed expected feature-date set from the returns panel and **`[graph]`** settings; **`edges.parquet`** must have the same number of **distinct dates** as **`universe.parquet`** (so a partial edges checkpoint after **Ctrl+C** is **not** treated as complete—the `edges` step runs again and **`build_edge_lists(..., overwrite=False)`** resumes from the partial file).

### Config fingerprints (Phase 2)

Per-section SHA-256 hashes detect stale artifacts when TOML changes without `--overwrite`:

| Config section | Stored in | Checked by |
|----------------|-----------|------------|
| `[graph]` | `interim/graphs/config_fingerprint.json` | `graph.prepare` substeps |
| `[dataset]` | `interim/dataset/config_fingerprint.json` | `dataset.package` substeps |
| `[model.train]` | `interim/model_train/final_metrics.json` → `config_fingerprint` | `model.train` skip |
| `[model.evaluate]` | `processed/evaluate_config_fingerprint.json` | `model.evaluate` substeps |

`model.train` also requires upstream graph + dataset fingerprints to match. `model.cache_graphs` requires graph + dataset fingerprints. `model.evaluate` requires evaluate fingerprint + train fingerprint.

On pipeline start, **`ensure_config_fingerprint_sidecars`** stamps missing sidecars when semantics already pass (migration from pre–Phase 2 runs). When a step reruns due to fingerprint drift, the orchestrator prints e.g. `running (config fingerprint changed for [graph])`.

`python scripts/run.py validate-config --run <id>` prints non-fatal warnings (`rebalance_freq` unwired, calibration asymmetry, etc.); use `--strict` to exit non-zero.

## Path layout (Phase 1)

| Layer | Directory | Owned by |
|-------|-----------|----------|
| Raw inputs | `data/shared/raw/` | user (not in git) |
| Shared prepare | `data/shared/interim/` | `data.prepare` only |
| Per-run modeling | `data/<run_id>/interim/` | `graph.prepare` → `model.train` |
| Per-run reporting | `data/<run_id>/processed/` | `model.evaluate` onward |

Templates in TOML: `[paths].raw`, `[paths].shared_interim`, `[paths].interim`, `[paths].processed` (see `configs/default.toml`).

## Ingest (shared interim)

These artifacts are produced by the `ingest` step inside `data.prepare`. They live under **`data/shared/interim/`**. **`graph.prepare` onward** writes under the run’s **`data/<run_id>/interim/`**; **`processed/`** holds evaluation summaries, figures, and exhibit exports.

### `ingest_manifest.json`
- **Producer:** `data.prepare` / step `ingest` — `mss.data.ingest.ingest_raw_to_parquet`
- **Consumers:** operational checks, auditing, future stages that need resolved paths
- **Format:** JSON with `inputs` (raw CSV paths), `outputs` (interim paths), CRSP partition metadata, `flags.overwrite`

### `sp500_returns.parquet`, `vix.parquet`
- **Producer:** same as above (`ingest_sp500_returns_to_parquet`, `ingest_vix_to_parquet`)
- **Consumers:** later `data.prepare` (e.g. targets), evaluation/benchmark alignment; VIX remains benchmark-only downstream
- **Format:** Parquet; normalized columns include `date`; VIX level column `vix`

### CRSP shard directory `crsp_parquet/`
- **Producer:** `ingest_crsp_wrds_to_parquet` — layout `crsp_parquet/year=YYYY/data.parquet`
- **Consumers:** later `data.prepare` (returns panel)
- **Format:** Hive-style year partitions; schema derived from raw WRDS CSV with fixed typing rules (`ingest` module)

**Validation:** optional CSV vs Parquet summary check (`validate_ingest`); skip with CLI `--skip-validate` when needed.

### `returns_panel.parquet` (interim)

- **Producer:** `data.prepare` / step `returns_panel` — `mss.data.returns_panel.build_returns_panel` (reads `crsp_parquet/**` shards).
- **Consumers:** `graph.prepare` (universe, node features, edges) and diagnostics; any step needing a clean daily equity panel.
- **Path:** `data/shared/interim/returns_panel.parquet`.
- **Format:** Parquet (ZSTD). Column set includes at least `date`, `permno`, `ret_used`, identifiers, and delisting fields; see `mss.data.returns_panel` and `check_returns_panel` for the enforced contract.
- **Return columns:** `ret_used` is the primary return (`retx` or `ret` per `[data.returns_panel].use_ret`). When `[data.returns_panel].apply_delisting_adjustment = true`, `ret_delisted` is populated on delisting dates as `(1 + ret_used) * (1 + dlret) - 1` when `dlstcd` and `dlret` are present; otherwise `ret_delisted` is null. Graph steps may set `[graph].ret_col` to `ret_used` or `ret_delisted`.
- **Key constraints:** unique (`date`, `permno`); enforced after build via `check_returns_panel` in the orchestrator step.

### Trading-day rolling windows (Phase 3)

Graph rolling windows (universe eligibility, edges, node features, feature dates) share one definition via `mss.calendar.trading_windows.TradingCalendar`:

- **Window at anchor date `d`:** inclusive trading dates with positions `[pos(d) - window_length + 1, pos(d)]` on the sorted distinct `returns_panel` dates (same as universe DuckDB `dates_index` SQL).
- **Feature dates:** first date with a full `window_length` history on that calendar (`TradingCalendar.feature_dates`).
- **No calendar `Timedelta` buffers** — holidays and missing market days are handled by the trading spine only.

Optional node features (behind `[graph.node_features]` flags, default off): `log_dollar_volume` = mean of `log1p(abs(prc) * vol)`; `turnover` = mean of `vol / shrout` over the same window.

### Universe modes (Phase 4)

- **`fixed_replace`:** seed top `n_nodes` by mcap; replace only on attrition (sticky cohort). `rebalance_freq` in TOML is informational only.
- **`monthly_rebalance`:** fresh top-N at rebalance anchor from `rebalance_freq` (`monthly` = last trading day of prior month; `daily` = feature date; `quarterly` = last trading day of prior quarter). Universe constant between rebalance dates within a period.

### Edge weights (Phase 4)

- **`edges.parquet` `weight`:** signed correlation magnitude from graph build; loaded as PyG `edge_attr` shape `[E, 1]`.
- **`[model.train].use_edge_weights`:** when `true`, GraphSAGE uses weighted neighbor aggregation with `abs(weight)`; GAT uses `edge_attr`. **`default.toml` sets `true`**; set `false` for unweighted ablations.

### Forward realized volatility (`targets.parquet`)

- **Active default — forward 30-calendar-day RV (`rv_fwd_30cal`):** `sqrt(annualization / N_t × Σ sprtrn²)` over S&P trading returns in the calendar window `(t, t + 30 days]` (DuckDB `RANGE BETWEEN INTERVAL 1 DAY FOLLOWING AND INTERVAL 30 DAY FOLLOWING`), where `N_t = n_fwd_30cal_obs` is the actual number of trading observations, scaled by `scale` (default 100). Horizon-aligned with the VIX.
  - **Gate:** `n_fwd_30cal_obs` must be at least `min_obs_cal` (default 15) and the forward window must fit within the sample; otherwise `rv_fwd_30cal` / `log_rv_fwd_30cal` is null.
- **Legacy/robustness — forward 30-trading-day RV (`rv_fwd_30`):** `sqrt(annualization / 30 × Σ sprtrn²)` over the next **30 ordered spine rows** (DuckDB `ROWS BETWEEN 1 FOLLOWING AND 30 FOLLOWING`). Retained for robustness comparisons; **not** the default target.
  - **Gate:** `n_fwd_30_obs` must equal 30 or `rv_fwd_30` / `log_rv_fwd_30` is null.

## Core Contracts

### 1) `targets.parquet`
- **Producer:** `data.prepare` / step `targets` — `mss.data.targets.build_targets` (reads `interim/sp500_returns.parquet`, `interim/vix.parquet` by default).
- **Path:** `data/shared/interim/targets.parquet`.
- **Sidecar:** `data/shared/interim/targets_manifest.json`.
- **Consumers:** `dataset.package`, `model.evaluate`, `analysis.summarize`
- **Required columns (contract minimum):**
  - `date`
  - `rv_fwd_30cal`, `log_rv_fwd_30cal` (active default target)
  - `rv_fwd_30`, `log_rv_fwd_30` (legacy/robustness target)
  - optional `vix`
- **Also present (extended spine):** e.g. `sprtrn`, `n_fwd_30cal_obs`, `n_fwd_30_obs`, lag RV columns and logs for configured trailing windows — superset of the minimum above.
- **Constraints:**
  - unique `date`
  - sorted by `date`
  - finite numeric values where defined (validated via `check_targets` after build)
- **Semantics:**
  - `log_rv_fwd_30cal` is the default modeling target (`[graph.output].target_column`)
  - `log_rv_fwd_30` is retained for robustness only
  - `vix` remains benchmark-only

### 2) `stage04_dates.parquet` (interim)

- **Producer:** `graph.prepare` / step `feature_dates` — valid feature dates with a full rolling window of history (optionally intersected with `targets.parquet` when `[graph].align_feature_dates_with_targets` is true).
- **Path:** `data/<run_id>/interim/stage04_dates.parquet`
- **Consumers:** `graph.prepare` / step `universe`

### 3) `universe.parquet`
- **Producer:** `graph.prepare` / step `universe`
- **Path:** `data/<run_id>/interim/graphs/universe.parquet`
- **Consumers:** `graph.prepare` (feature/edge substeps), `dataset.package`
- **Required columns:** `date`, `permno`, `rank`, `mcap`
- **Constraints:** unique (`date`, `permno`)

### 4) `node_features.parquet`
- **Producer:** `graph.prepare` / step `node_features`
- **Path:** `data/<run_id>/interim/graphs/node_features.parquet`
- **Consumers:** `dataset.package`, `model.train`
- **Required columns:** `date`, `permno`, feature columns
- **Constraints:**
  - unique (`date`, `permno`)
  - no forbidden nulls in required feature columns

### 5) `edges.parquet`
- **Producer:** `graph.prepare` / step `edges`
- **Path:** `data/<run_id>/interim/graphs/edges.parquet`
- **Consumers:** `model.train`, `model.evaluate`
- **Required columns:** `date`, `src`, `dst`, `weight`
- **Constraints:**
  - unique (`date`, `src`, `dst`)
  - no self-loops unless explicitly allowed by config
  - finite `weight`

### 6) `splits.parquet`
- **Producer:** `dataset.package` / step **`splits`** — `mss.dataset.splits.write_splits_parquet` (distinct dates from **`interim/graphs/universe.parquet`**; boundaries from **`[dataset]`** in TOML).
- **Path:** `data/<run_id>/interim/dataset/splits.parquet`
- **Consumers:** `dataset.package` / **`scaler`**, `model.train`, `model.evaluate`
- **Required columns:** `date`, `split`
- **Constraints:**
  - split in `{train, val, test}`
  - chronological split boundaries (`train_end` / `val_end` / `test_end` inclusive-end semantics; see `build_split_assignment`)
  - disjoint assignments per date
- **Resume:** step is complete when file matches recomputation from current **`universe.parquet`** + **`[dataset]`** (see **`mss.dataset.checks.splits_matches_universe_and_config`**).

### 7) `manifest.json` (dataset package)
- **Producer:** `dataset.package` / step **`manifest`**
- **Path:** `data/<run_id>/interim/dataset/manifest.json`
- **Consumers:** `model.train`, `model.evaluate`
- **Sidecar:** `interim/dataset/scaler_params.json` (train-only scaler); optional **`interim/dataset/node_features_scaled.parquet`** when **`[dataset].write_scaled_features`** is true.
- **Required JSON keys (minimum):**
  - `splits_path`, `node_features_path`, `edges_path`, `universe_path`, `scaler_path`, **`targets_path`** ( **`interim/targets.parquet`** )
  - `target_column`, `feature_columns` (list of strings)
  - `scale_target` (bool; typically false at packaging time)
  - optional `node_features_scaled_path` when scaled parquet is written
- **Constraints:**
  - all referenced paths exist
  - `target_column` exists in **`targets.parquet`**
  - feature columns exist in **`node_features.parquet`**

### 8) `model.cache_graphs` outputs (`interim/model_train/graph_cache/`)

- **Producer:** `model.cache_graphs` / step **`materialize`** — `mss.model_train.cache_materialize.materialize_pyg_graph_cache`
- **Path:** `data/<run_id>/interim/model_train/graph_cache/`
- **Artifacts:** one PyG object per expected train/val date with valid label:
  - **`YYYY-MM-DD.pt`** — serialized `torch_geometric.data.Data` (x, edge_index, edge_attr, y)
- **Completeness (orchestrator):** step is complete only when **all expected `.pt` files** are present and non-empty.
- **Overwrite:** **`--overwrite model.cache_graphs`** rebuilds cache files for the run id.

### 9) `model.train` outputs (`interim/model_train/`)

- **Producer:** `model.train` / step **`train`** — `mss.model_train.train_loop.run_model_train`
- **Path:** `data/<run_id>/interim/model_train/` (default layout)
- **Artifacts:**
  - **`checkpoint.pt`** — latest model weights, optimizer state, epoch counters (written during training; used for resume)
  - **`training_state.json`** — epoch, best validation loss, early-stopping counter, metrics history, **`seed`** (from config), **`config_fingerprint`** (hash of **`[model.train]`** TOML)
  - **`final_metrics.json`** — present **only** when training finished successfully; must include **`status: "completed"`**, **`best_epoch`**, **`best_val_loss`**, optional **`seed`**, and related summary fields
- **Reproducibility:** set **`[model.train].seed`** (default **42**) for Python / NumPy / PyTorch; changing **`seed`** changes the config fingerprint and affects DataLoader shuffle order (applied before loaders are built).
- **I/O:** **`[model.train].parquet_date_pushdown`** (default **true**) applies Parquet row-group filters on **`date`** when building each graph; set **false** if reads misbehave for a given Parquet layout.
- **Skip vs resume (orchestrator):** Without **`--overwrite`**, the **`train`** step is **skipped** only when **`final_metrics.json`** is present and semantically complete (**`check_final_metrics_complete`** in **`mss.model_train.checks`**). **Checkpoint presence alone does not skip the step** — the step runs and **resumes** from **`checkpoint.pt`** + **`training_state.json`** when the run is incomplete. **`--overwrite model.train`** clears training artifacts and starts cold **while preserving `graph_cache/`**.
- **DataLoader workers:** configure **`[model.train].dataloader_num_workers`** (default **0**, recommended on Windows); do not overload **`n_jobs`** (reserved for graph multiprocessing).

### 10) `forecasts.parquet`
- **Producer:** `model.evaluate` / step **`score_splits`** — `mss.evaluation.inference.run_score_splits`
- **Consumers:** `model.evaluate` / **`write_forecast_panel`**, **`aggregate_test_loss`**; `analysis.summarize` (forecast figures join VIX from manifest **`targets_path`**)
- **Path:** `data/<run_id>/processed/scoring/forecasts.parquet`
- **Required columns:** `date`, `split`, `sample`, `y_true`, `y_pred_model` (target and prediction in **log space**, same as training)
- **Extended columns (default run):** `y_pred_placebo` (same trained GNN re-scored with permuted graph edges; same node features/date alignment)
- **Constraints:** see **`mss.evaluation.checks.check_forecasts`** (finite numerics; `(date, split)` uniqueness as enforced by checks)
- **Skip semantics:** `model.evaluate` step **`score_splits`** is complete when this file passes **`check_forecasts`**.

### 10b) Descriptive vs formal inference (processed/)

- **Descriptive evaluation** — Aggregated metrics under **`metrics/descriptive/`** (e.g. **`summary_table.csv`**) includes **train** and **val** slices and **`excl_crisis`** (train dates **outside** the configured crisis window). These labels are **not** the same as the formal inference subsamples below.
- **Formal inference (thesis)** — Under **`metrics/formal/`**; date-level panel in **`scoring/forecast_panel.parquet`** + HAC-aware tests on **test split only** with pre-specified masks: **`test`**, **`test_excl_2020`**, **`test_excl_stress`** (dates outside **`[model.evaluate].summary_test_stress_excl_*`** or named **`hypothesis_stress_preset`**), **`test_excl_union`**. Stress preset IDs: **`covid_2020`**, **`rate_shock_2022`** (override with explicit start/end in config).

### 10c) `forecast_panel.parquet`

- **Producer:** `model.evaluate` / step **`write_forecast_panel`** — `mss.evaluation.forecast_panel.run_write_forecast_panel`
- **Consumers:** **`write_summary_table`** (column-aligned descriptive table); **`run_hypothesis_tests`**
- **Path:** `data/<run_id>/processed/scoring/forecast_panel.parquet`
- **Role:** Persists benchmark-merged **log and level** quantities and per-date loss differentials so formal tests do not re-derive them: `y_true_log`, `y_true_level`, `y_pred_model_log`, `y_pred_model_level`, `y_pred_vix_log`, `y_pred_vix_level`, `vix`, `qlike_*_t`, `d_qlike_*_t`, `mse_log_*_t`, `d_mse_log_*_t`, etc. The v2 benchmark suite adds `y_pred_calibrated_vix_*`, `y_pred_vix_har_*` (plus placebo `y_pred_placebo_*` and standalone `y_pred_har_*`) with matching per-date losses and GNN differentials (`d_qlike_{b}_t`, `d_mse_log_{b}_t`, positive ⇒ GNN better). See **`mss.evaluation.checks.check_forecast_panel`**.
- **Benchmark spaces (v2):**
  - **Target:** `y_true_log` = forward **30-calendar-day** realized vol (log); `y_true_level` in annualized vol (%). Horizon-aligned with the VIX.
  - **raw_vix (primary):** `y_pred_vix_log` = log(spot VIX); QLIKE uses the spot `vix` level (not `exp(log vix)`).
  - **calibrated_vix:** train-only affine fit `y_true_log ~ a + b·log_vix` applied unchanged to val/test — a bias/level correction, **not** risk-premium removal.
  - **vix_har:** hybrid implied-plus-historical benchmark (log VIX + `log_rv_lag_5/30/252`), fit train-only.
  - **har:** standalone realized-vol HAR from `log_rv_lag_*` in shared **`targets.parquet`** only.
  - **Placebo:** `d_qlike_placebo_t = L_placebo - L_GNN`, `d_mse_log_placebo_t` analogous for H5.

### 11) `test_loss.json`
- **Producer:** `model.evaluate` / step **`aggregate_test_loss`** — scalar loss on **test** split (default metric matches **`[model.train].loss`**, e.g. **`mse_log`**; override with **`[model.evaluate].test_metric`**)
- **Consumers:** `analysis.summarize` (horizontal reference line on loss-over-epochs figures)
- **Path:** `data/<run_id>/processed/scoring/test_loss.json`
- **Required keys (minimum):** `status` = `"completed"`, `metric` (allowed training loss id), `split` = `"test"`, `n_rows`, `value` (scalar loss)
- **Skip semantics:** step **`aggregate_test_loss`** is satisfied when **`test_loss.json`** passes **`check_test_loss`** (given valid forecasts).

### 11b) `metrics/descriptive/summary_table.csv`
- **Producer:** `model.evaluate` / step **`write_summary_table`** — `mss.evaluation.inference.run_write_summary_table` (builds from **`scoring/forecast_panel.parquet`**; same aggregate metrics as before via **`mss.evaluation.summary_table.build_summary_table_rows`**).
- **Path:** `data/<run_id>/processed/metrics/descriptive/summary_table.csv`
- **Columns:** `model` (`gnn` \| `placebo` \| `har` \| `vix` \| `calibrated_vix` \| `vix_har`), `sample`, `mse_log`, `mae_log`, `mse`, `mae`, `qlike`, `n_samples`. `vix` is raw spot VIX (primary); `calibrated_vix` and `vix_har` are secondary benchmark rows.
- **Samples (when non-empty slices exist):** `train`, `val`, **`excl_crisis`** (train rows **outside** the inclusive window **`[model.evaluate].summary_train_crisis_excl_start`**–**`summary_train_crisis_excl_end`**, defaults **2008-09-01**–**2009-03-31**), **`full_test`**, **`excl_2020`** (test rows excluding **`summary_test_exclusion_years`**, default **2020**)
- **Skip semantics:** step complete when this file passes **`check_summary_table`** after valid **`forecast_panel.parquet`**, **`test_loss.json`**, and **`forecasts.parquet`**.

### 11e) `metrics/descriptive/diagnostics_smoothing.csv`

- **Producer:** `model.evaluate` / step **`write_summary_table`**
- **Path:** `data/<run_id>/processed/metrics/descriptive/diagnostics_smoothing.csv`
- **Columns:** `sample`, `diagnostic`, `model`, `value`, `detail`
- **Purpose:** compact diagnostics for potential smoothness-driven wins (`forecast_variance_log`, `corr_with_target_log`, `mz_intercept`, `mz_slope`, and `mse_log_by_target_vol_quantile` rows by `detail=q1..q5`) across `gnn`, `placebo`, `har`, `vix`.

### 11c) `metrics/formal/hypothesis_tests.csv` (formal inference)

- **Producer:** `model.evaluate` / step **`run_hypothesis_tests`** — `mss.evaluation.inference.run_hypothesis_tests_step` (statistics in **`mss.evaluation.hypothesis_tests`**; **only** reads **`scoring/forecast_panel.parquet`** + **`[model.evaluate]`** TOML).
- **Path:** `data/<run_id>/processed/metrics/formal/hypothesis_tests.csv`
- **Columns:** **`hypothesis_id`** (thesis **H1–H5**, plus **`H2_benchmark`** / **`H3_benchmark`** for the generalized secondary-benchmark rows); **`inference_procedure`** (method, not thesis id): **`mz_gnn`**, **`dm_loss_diff`**, **`incremental_gnn_vix`** (canonical raw-VIX incremental), **`incremental_vs_benchmark`** (generalized), or **`subsample_audit`** when inference was skipped because the subsample would duplicate another; **`statistic_type`**: **`hac_t`** (HAC **t** from OLS on slopes) or **`hac_t_mean`** (HAC **t** for mean of **`d_t`** in the DM step); verbatim **`null_hypothesis`** / **`alternative`**; **`test_name`**; **`sample`** (**formal** subsample: **`test`**, **`test_excl_2020`**, **`test_excl_stress`**, **`test_excl_union`**); **`benchmark`** (which comparator a row uses: **`raw_vix`** for canonical H2/H3, **`calibrated_vix`** / **`vix_har`** / **`har`** for generalized rows, **`placebo`** for H5, **`none`** for MZ); **`coefficient_tested`**; **`test_scope`**; **`joint_hypothesis`**; **`tail`**; **`alternative_direction`**; **`better_model`**; **`loss_name`** / **`loss_definition`** (H2/H5); **`statistic`**; **`p_value_primary`**, **`p_value_two_sided`**, **`p_value_one_sided_upper`**; **`hac_max_lags`**; **`n_obs`**; audit: **`rows_removed_vs_full_test`**, **`effective_sample_start`**, **`effective_sample_end`**, **`stress_rows_removed`**, **`year_rows_removed`** (counts on the full **test** split for transparency), **`subsample_operational`** (false for **`subsample_audit`** rows); **`notes`**.
- **Non-operative subsamples:** If **`[model.evaluate].summary_test_stress_excl_*`** does not intersect any **test** date, **`test_excl_stress`** is identical to full **`test`**—the step emits a single **`subsample_audit`** row (no duplicate p-values). If stress removes no test rows, **`test_excl_union`** can match **`test_excl_2020`** exactly; that case also emits **`subsample_audit`** instead of duplicating H2/H3.
- **Thesis hypotheses (wording in code = README):**
  - **H1:** Predictive content of the GNN — H0: no predictive information about future **30-calendar-day** realized volatility; H1: contains predictive information. **Test:** Mincer–Zarnowitz slope of **`y_true_log`** on **`y_pred_model_log`**; **coefficient tested** = **`y_pred_model_log`**; **slope only**; **\(H_0: \beta=0\)**; **primary `p_value` = two-sided** HAC **t**.
  - **H2:** Relative OOS accuracy vs **raw VIX** — H0: GNN does not achieve lower OOS loss than raw VIX; H1: GNN does. **Test:** Diebold–Mariano on **`d_t = L^{VIX}_t - L^{GNN}_t`** (positive mean ⇒ GNN better for QLIKE or MSE-log per row). **Primary `p_value` = one-sided upper** (\(E[d]>0\)); **`p_value_two_sided`** also reported. `raw_vix` is the primary benchmark; the same DM test runs against each secondary benchmark as **`H2_benchmark`** rows (distinguished by **`benchmark`**).
  - **H3:** Incremental information conditional on **raw VIX** — H0: conditional on raw VIX, GNN adds no incremental information; H1: it does. **Test:** slope on **`y_pred_model_log`** in **`y_true_log ~ const + y_pred_model_log + y_pred_vix_log`**; **slope only**; **primary `p_value` = two-sided** HAC **t** on that slope.
  - **incremental_vs_benchmark (`H3_benchmark`):** Generalized incremental test conditional on a secondary benchmark (`calibrated_vix` / `vix_har` / `har`), identified by the **`benchmark`** column. This deliberately is **not** labeled "beyond VIX"; each row conditions on the named benchmark.
  - **H4:** Robustness — **H4** is implemented by **re-running the same procedures as H2 and H3** (and optional **H1** when **`hypothesis_h4_include_h1`**) on restricted **test** subsamples. Rows use **`hypothesis_id = H4`** with **`inference_procedure`** naming the procedure (**`mz_gnn`** / **`dm_loss_diff`** / **`incremental_gnn_vix`**) or **`subsample_audit`** when skipped.
  - **H5:** Relative OOS accuracy vs placebo — H0: GNN does not beat placebo on OOS loss; H1: it does. **Test:** Diebold–Mariano on **`d_t = L^{placebo}_t - L^{GNN}_t`** (QLIKE or MSE-log per **`hypothesis_dm_losses_vs_placebo`**). **Primary `test` sample only** (not H4 subsamples). **Primary `p_value` = one-sided upper**.

### 11d) `metrics/formal/regression_mz_gnn.csv`, `metrics/formal/regression_incremental.csv`

- **Producer:** same step **`run_hypothesis_tests`**
- **Format:** Long table; per **`term`**: `coef`, `se_hac`, `t`, `p_two_sided`, `ci_lo_95`, `ci_hi_95`, plus **`r_squared`**, **`n_obs`**, **`hac_max_lags`**, `sample`, `spec`, `dep_var`, **`coefficient_tested_for_hypothesis`**, **`test_scope`**, and the same subsample audit fields as **`hypothesis_tests.csv`**: **`rows_removed_vs_full_test`**, **`effective_sample_start`**, **`effective_sample_end`**, **`stress_rows_removed`**, **`year_rows_removed`**, **`subsample_operational`**.

- **Skip semantics:** full **`model.evaluate`** pipeline is complete when **`scoring/forecasts.parquet`**, **`scoring/forecast_panel.parquet`**, **`scoring/test_loss.json`**, **`metrics/descriptive/summary_table.csv`**, **`metrics/formal/hypothesis_tests.csv`**, and the two regression CSVs exist and pass checks (**`mss.evaluation.checks.model_evaluate_pipeline_semantically_complete`**).

### 12) `analysis.summarize` figure outputs (Phase 6 layout)
- **Producer:** `analysis.summarize` / step **`loss_figure`** — `mss.analysis.summarize` → `mss.analysis.figures`
- **Forecast comparison (VIX joined at plot time from `interim/targets.parquet` via dataset manifest):**
  - `data/<run_id>/processed/figures/by_split/all/forecast_vs_benchmarks.png`
  - `data/<run_id>/processed/figures/by_split/{split}/forecast_vs_benchmarks.png` for each of **`train`**, **`val`**, **`test`** present in **`[model.evaluate].splits`**
- **Loss vs epochs (from **`training_state.json`** + **`scoring/test_loss.json`**):**
  - `data/<run_id>/processed/figures/diagnostics/loss_over_epochs_linear.png`
  - `data/<run_id>/processed/figures/diagnostics/loss_over_epochs_log.png`
- **Summary barplot (from **`metrics/descriptive/summary_table.csv`**):**
  - `data/<run_id>/processed/figures/diagnostics/summary_barplot.png`
- **Hypothesis table (optional; when **`[analysis.summarize] draw_hypothesis_table = true`**):**
  - `data/<run_id>/processed/figures/diagnostics/hypothesis_tests_table.png` — reads **`metrics/formal/hypothesis_tests.csv`**
- **Universe churn:** figures under **`figures/diagnostics/universe_churn/`**; CSVs under **`metrics/universe/`**
- **Semantics:** three series on forecast plots are comparable in log-target space; VIX baseline is **`log(max(vix, eps))`** aligned with project convention. Loss plots show train/val curves per epoch and a flat **test** horizontal line at **`test_loss.json`**'s scalar (**`metric`** matches training unless **`[model.evaluate].test_metric`** overrides).
- **Skip semantics:** all paths returned by **`expected_paths_for_summarize(cfg)`** must exist and be non-empty (see **`mss.pipeline.completeness.analysis_summarize_loss_figure_semantically_complete`**). **`loss_figure`** inputs include valid **`metrics/descriptive/summary_table.csv`** and, when **`draw_hypothesis_table`** is true, **`metrics/formal/hypothesis_tests.csv`**.

### 12b) `processed/tables/thesis/` (thesis manuscript exports)

- **Producer:** `thesis.export` / step **`build`** — `mss.thesis.export.run_thesis_export`
- **Consumers:** thesis / manuscript tooling (not consumed by other pipeline steps)
- **Path root:** `data/<run_id>/processed/tables/thesis/` (was `thesis_exhibits/`)
- **Layout:**
  - **`main/tables/`** — locked main-text tables (`M1_*`, `M3_*`, `M4_*` as `.csv` + `.tex`)
  - **`main/figures/`** — `M2_test_forecast_paths.png` (copy of test forecast figure), `M5_mse_log_by_vol_quintile.{png,pdf}`
  - **`appendix/tables/`** — **CSV only** (no appendix `.tex`). **Thesis-facing** filenames: `Table_A01_*` … `Table_D04_*`. **`OLD_*`** archival copies only when **`[thesis.export] legacy_archive = true`** (written to **`tables/archive/`**).
  - **`appendix/figures/`** — `A4_universe_diagnostics_plate.{png,pdf}`; **`Figure_F01_universe_diagnostics_plate.{png,pdf}`**
  - **`manifest/`** — `thesis_exhibits_manifest.{md,json}`, `thesis_table_index.md`, `Reproducibility_metadata.json`
- **Semantics:** derived from **`metrics/`**, **`scoring/`**, and **`figures/`**; **`--overwrite thesis.export`** removes and rebuilds **`tables/thesis/`** only.
- **Skip semantics:** step is complete when all paths returned by **`mss.thesis.export.expected_paths_for_thesis_export(cfg)`** exist and are non-empty (see **`mss.pipeline.completeness.thesis_export_semantically_complete`**).

### 14) `processed/comparisons/<comparison_id>/` (cross-run comparison)

- **Producer:** `analysis.compare_runs` / step **`build`** — `mss.analysis.compare_runs.run_compare_runs` (CLI: `compare-runs`)
- **Consumers:** Phase 8 app compare panel; manuscript Appendix E-style exports
- **Path root:** `data/<anchor_run_id>/processed/comparisons/<comparison_id>/` (anchor run holds the comparison package)
- **Inputs (read-only, per compared run):**
  - `metrics/descriptive/summary_table.csv`
  - `metrics/descriptive/diagnostics_smoothing.csv`
  - `metrics/formal/hypothesis_tests.csv`
- **Layout:**
  - `README.md`, `comparison_manifest.json`, `comparison_note.md`
  - `tables/Table_E01_alternative_configuration_comparison.csv` … `Table_E05_full_test_mse_log_by_target_volatility_quintile.csv`
  - `figures/Figure_E01_full_test_mse_log_by_target_volatility_quintile_across_configurations.png`
- **Configuration:** `[analysis.compare_runs]` on anchor TOML (`comparison_id`, `peer_runs`, optional `run_labels`) or CLI `--anchor`, `--runs`, `--comparison-id` / `--preset`
- **Semantics:** dynamic N-run columns; placebo rows omitted when `[model.evaluate] enable_placebo_ablation = false` on a run; VIX quintile column checked for agreement across runs before collapse
- **Skip semantics:** step is complete when all paths from **`mss.analysis.compare_runs.expected_paths_for_compare(anchor_cfg, comparison_id)`** exist and are non-empty (see **`mss.pipeline.completeness.analysis_compare_runs_semantically_complete`**)

### 13) Removed / out of scope
- **Removed (Phase 6):** legacy **`mss.analysis.h1`**, **`pairwise`**, **`report`** modules (superseded by **`mss.evaluation.hypothesis_tests`**).
- **Out of scope:** **`metrics_summary.json`**, **`processed/tables/eval_*.csv`** (legacy naming). **Formal inference** is **`metrics/formal/hypothesis_tests.csv`** only.
- **Migration:** Re-run **`model.evaluate`** → **`thesis.export`** on existing runs after upgrading from flat **`summaries/`** + root-level parquets (see **`processed/README.md`**).

## Validation Layers

1. Input existence checks
2. Schema checks (columns + basic types)
3. Key constraints (uniqueness, coverage, sortedness)
4. Numeric integrity (finite values, bounds where applicable)
5. Leakage checks (train-only fit for transforms, chronological discipline)

## Contract change discipline

When changing producers or consumers of an artifact:

- preserve math and definitions, or document the change explicitly
- keep paths and naming consistent with this document
- document intentional behavioral deviations in phase notes or run README sidecars
