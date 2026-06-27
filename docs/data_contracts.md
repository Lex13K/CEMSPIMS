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

## Ingest (interim)

These artifacts are produced by the `ingest` step inside `data.prepare`. By default they live under **`data/<run_id>/interim/`**, where `run_id` comes from the CLI **`--run`** flag (default `default`). Templates use **`{run_id}`** in `[paths].interim` / `[paths].processed` in TOML; **`data/raw/`** is shared across runs unless you override `[paths].raw`. Run-scoped **modeling tables** (returns panel, targets spine, graph snapshots) live under **`interim/`**; **`processed/`** is reserved for evaluation summaries, figures, and similar reporting outputs (not pipeline modeling tables).

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
- **Path:** `data/<run_id>/interim/returns_panel.parquet` (default layout).
- **Format:** Parquet (ZSTD). Column set includes at least `date`, `permno`, `ret_used`, identifiers, and delisting fields; see `mss.data.returns_panel` and `check_returns_panel` for the enforced contract.
- **Key constraints:** unique (`date`, `permno`); enforced after build via `check_returns_panel` in the orchestrator step.

## Core Contracts

### 1) `targets.parquet`
- **Producer:** `data.prepare` / step `targets` — `mss.data.targets.build_targets` (reads `interim/sp500_returns.parquet`, `interim/vix.parquet` by default).
- **Path:** `data/<run_id>/interim/targets.parquet` (default layout).
- **Sidecar:** `data/<run_id>/interim/targets_manifest.json` (inputs, parameters, column list, row stats).
- **Consumers:** `dataset.package`, `model.evaluate`, `analysis.summarize`
- **Required columns (contract minimum):**
  - `date`
  - `rv_fwd_30`
  - `log_rv_fwd_30`
  - optional `vix`
- **Also present (extended spine):** e.g. `sprtrn`, `n_fwd_30_obs`, lag RV columns and logs for configured trailing windows — superset of the minimum above.
- **Constraints:**
  - unique `date`
  - sorted by `date`
  - finite numeric values where defined (validated via `check_targets` after build)
- **Semantics:**
  - `log_rv_fwd_30` is the default modeling target
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
- **Path:** `data/<run_id>/processed/forecasts.parquet`
- **Required columns:** `date`, `split`, `sample`, `y_true`, `y_pred_model` (target and prediction in **log space**, same as training)
- **Extended columns (default run):** `y_pred_placebo` (same trained GNN re-scored with permuted graph edges; same node features/date alignment)
- **Constraints:** see **`mss.evaluation.checks.check_forecasts`** (finite numerics; `(date, split)` uniqueness as enforced by checks)
- **Skip semantics:** `model.evaluate` step **`score_splits`** is complete when this file passes **`check_forecasts`**.

### 10b) Descriptive vs formal inference (processed/)

- **Descriptive evaluation** — Aggregated metrics for exploration and figures; **not** the primary location for formal hypothesis tests. Example: **`summaries/summary_table.csv`** includes **train** and **val** slices and **`excl_crisis`** (train dates **outside** the configured crisis window). These labels are **not** the same as the formal inference subsamples below.
- **Formal inference (thesis)** — Date-level panel + HAC-aware tests on **test split only** with pre-specified masks: **`test`** (full holdout), **`test_excl_2020`** (test dates not in **`summary_test_exclusion_years`**), **`test_excl_stress`** (test dates **outside** **`summary_test_stress_excl_start`**–**`summary_test_stress_excl_end`**), **`test_excl_union`** (both exclusions). Implemented in **`model.evaluate` / `run_hypothesis_tests`**; **`run_hypothesis_tests` reads only `forecast_panel.parquet`** (not the descriptive summary CSV).

### 10c) `forecast_panel.parquet`

- **Producer:** `model.evaluate` / step **`write_forecast_panel`** — `mss.evaluation.forecast_panel.run_write_forecast_panel`
- **Consumers:** **`write_summary_table`** (column-aligned descriptive table); **`run_hypothesis_tests`**
- **Path:** `data/<run_id>/processed/forecast_panel.parquet`
- **Role:** Persists VIX-merged **log and level** quantities and per-date loss differentials so thesis tests do not re-derive them: `y_true_log`, `y_true_level`, `y_pred_model_log`, `y_pred_model_level`, `y_pred_vix_log`, `y_pred_vix_level`, `vix`, `qlike_*_t`, `d_qlike_t`, `mse_log_*_t`, `d_mse_log_t`, etc. Extended benchmark columns include placebo (`y_pred_placebo_*`) and HAR-like (`y_pred_har_*`) predictions for aligned comparator diagnostics. See **`mss.evaluation.checks.check_forecast_panel`**.

### 11) `test_loss.json`
- **Producer:** `model.evaluate` / step **`aggregate_test_loss`** — scalar loss on **test** split (default metric matches **`[model.train].loss`**, e.g. **`mse_log`**; override with **`[model.evaluate].test_metric`**)
- **Consumers:** `analysis.summarize` (horizontal reference line on loss-over-epochs figures)
- **Path:** `data/<run_id>/processed/test_loss.json`
- **Required keys (minimum):** `status` = `"completed"`, `metric` (allowed training loss id), `split` = `"test"`, `n_rows`, `value` (scalar loss)
- **Skip semantics:** step **`aggregate_test_loss`** is satisfied when **`test_loss.json`** passes **`check_test_loss`** (given valid forecasts).

### 11b) `summaries/summary_table.csv`
- **Producer:** `model.evaluate` / step **`write_summary_table`** — `mss.evaluation.inference.run_write_summary_table` (builds from **`forecast_panel.parquet`**; same aggregate metrics as before via **`mss.evaluation.summary_table.build_summary_table_rows`**).
- **Path:** `data/<run_id>/processed/summaries/summary_table.csv`
- **Columns:** `model` (`gnn` \| `placebo` \| `har` \| `vix`), `sample`, `mse_log`, `mae_log`, `mse`, `mae`, `qlike`, `n_samples`
- **Samples (when non-empty slices exist):** `train`, `val`, **`excl_crisis`** (train rows **outside** the inclusive window **`[model.evaluate].summary_train_crisis_excl_start`**–**`summary_train_crisis_excl_end`**, defaults **2008-09-01**–**2009-03-31**), **`full_test`**, **`excl_2020`** (test rows excluding **`summary_test_exclusion_years`**, default **2020**)
- **Skip semantics:** step complete when this file passes **`check_summary_table`** after valid **`forecast_panel.parquet`**, **`test_loss.json`**, and **`forecasts.parquet`**.

### 11e) `summaries/diagnostics_smoothing.csv`

- **Producer:** `model.evaluate` / step **`write_summary_table`**
- **Path:** `data/<run_id>/processed/summaries/diagnostics_smoothing.csv`
- **Columns:** `sample`, `diagnostic`, `model`, `value`, `detail`
- **Purpose:** compact diagnostics for potential smoothness-driven wins (`forecast_variance_log`, `corr_with_target_log`, `mz_intercept`, `mz_slope`, and `mse_log_by_target_vol_quantile` rows by `detail=q1..q5`) across `gnn`, `placebo`, `har`, `vix`.

### 11c) `summaries/hypothesis_tests.csv` (formal inference)

- **Producer:** `model.evaluate` / step **`run_hypothesis_tests`** — `mss.evaluation.inference.run_hypothesis_tests_step` (statistics in **`mss.evaluation.hypothesis_tests`**; **only** reads **`forecast_panel.parquet`** + **`[model.evaluate]`** TOML).
- **Path:** `data/<run_id>/processed/summaries/hypothesis_tests.csv`
- **Columns:** **`hypothesis_id`** (thesis **H1–H4**); **`inference_procedure`** (method, not thesis id): **`mz_gnn`**, **`dm_loss_diff`**, **`incremental_gnn_vix`**, or **`subsample_audit`** when inference was skipped because the subsample would duplicate another; **`statistic_type`**: **`hac_t`** (HAC **t** from OLS on slopes) or **`hac_t_mean`** (HAC **t** for mean of **`d_t`** in the DM step); verbatim **`null_hypothesis`** / **`alternative`**; **`test_name`**; **`sample`** (**formal** subsample: **`test`**, **`test_excl_2020`**, **`test_excl_stress`**, **`test_excl_union`**); **`coefficient_tested`**; **`test_scope`**; **`joint_hypothesis`**; **`tail`**; **`alternative_direction`**; **`better_model`**; **`loss_name`** / **`loss_definition`** (H2); **`statistic`**; **`p_value_primary`**, **`p_value_two_sided`**, **`p_value_one_sided_upper`**; **`hac_max_lags`**; **`n_obs`**; audit: **`rows_removed_vs_full_test`**, **`effective_sample_start`**, **`effective_sample_end`**, **`stress_rows_removed`**, **`year_rows_removed`** (counts on the full **test** split for transparency), **`subsample_operational`** (false for **`subsample_audit`** rows); **`notes`**.
- **Non-operative subsamples:** If **`[model.evaluate].summary_test_stress_excl_*`** does not intersect any **test** date, **`test_excl_stress`** is identical to full **`test`**—the step emits a single **`subsample_audit`** row (no duplicate p-values). If stress removes no test rows, **`test_excl_union`** can match **`test_excl_2020`** exactly; that case also emits **`subsample_audit`** instead of duplicating H2/H3.
- **Thesis hypotheses (wording in code = README):**
  - **H1:** Predictive content of the GNN — H0: no predictive information about future **30-trading-day** realized volatility; H1: contains predictive information. **Test:** Mincer–Zarnowitz slope of **`y_true_log`** on **`y_pred_model_log`**; **coefficient tested** = **`y_pred_model_log`**; **slope only**; **\(H_0: \beta=0\)**; **primary `p_value` = two-sided** HAC **t**.
  - **H2:** Relative OOS accuracy vs VIX — H0: GNN does not achieve lower OOS loss than VIX; H1: GNN does. **Test:** Diebold–Mariano on **`d_t = L^{VIX}_t - L^{GNN}_t`** (positive mean ⇒ GNN better for QLIKE or MSE-log per row). **Primary `p_value` = one-sided upper** (\(E[d]>0\)); **`p_value_two_sided`** also reported.
  - **H3:** Incremental information beyond VIX — H0: conditional on VIX, GNN adds no incremental information; H1: it does. **Test:** slope on **`y_pred_model_log`** in **`y_true_log ~ const + y_pred_model_log + y_pred_vix_log`**; **slope only**; **primary `p_value` = two-sided** HAC **t** on that slope.
  - **H4:** Robustness — **H4** is implemented by **re-running the same procedures as H2 and H3** (and optional **H1** when **`hypothesis_h4_include_h1`**) on restricted **test** subsamples. Rows use **`hypothesis_id = H4`** with **`inference_procedure`** naming the procedure (**`mz_gnn`** / **`dm_loss_diff`** / **`incremental_gnn_vix`**) or **`subsample_audit`** when skipped.

### 11d) `summaries/regression_mz_gnn.csv`, `summaries/regression_incremental.csv`

- **Producer:** same step **`run_hypothesis_tests`**
- **Format:** Long table; per **`term`**: `coef`, `se_hac`, `t`, `p_two_sided`, `ci_lo_95`, `ci_hi_95`, plus **`r_squared`**, **`n_obs`**, **`hac_max_lags`**, `sample`, `spec`, `dep_var`, **`coefficient_tested_for_hypothesis`**, **`test_scope`**, and the same subsample audit fields as **`hypothesis_tests.csv`**: **`rows_removed_vs_full_test`**, **`effective_sample_start`**, **`effective_sample_end`**, **`stress_rows_removed`**, **`year_rows_removed`**, **`subsample_operational`**.

- **Skip semantics:** full **`model.evaluate`** pipeline is complete when **`forecasts.parquet`**, **`forecast_panel.parquet`**, **`test_loss.json`**, **`summary_table.csv`**, **`hypothesis_tests.csv`**, and the two regression CSVs exist and pass checks (**`mss.evaluation.checks.model_evaluate_pipeline_semantically_complete`**).

### 12) `analysis.summarize` figure outputs (current)
- **Producer:** `analysis.summarize` / step **`loss_figure`** — `mss.analysis.summarize` → `mss.analysis.figures`
- **Forecast comparison (VIX joined at plot time from `interim/targets.parquet` via dataset manifest):**
  - `data/<run_id>/processed/figures/target_vs_model_vix/target_vs_model_vix_all_splits.png`
  - `data/<run_id>/processed/figures/target_vs_model_vix/target_vs_model_vix_{split}.png` for each of **`train`**, **`val`**, **`test`** present in **`[model.evaluate].splits`**
- **Loss vs epochs (from **`training_state.json`** + **`test_loss.json`**):**
  - `data/<run_id>/processed/figures/loss_over_epochs/loss_over_epochs_with_test_line_linear.png`
  - `data/<run_id>/processed/figures/loss_over_epochs/loss_over_epochs_with_test_line_log.png`
- **Summary barplot (from **`summaries/summary_table.csv`**):**
  - `data/<run_id>/processed/figures/general/summary_barplot.png`
- **Hypothesis table (optional; when **`[analysis.summarize] draw_hypothesis_table = true`**):**
  - `data/<run_id>/processed/figures/general/hypothesis_tests_table.png` — reads **`summaries/hypothesis_tests.csv`** (requires completed **`model.evaluate`** through **`run_hypothesis_tests`**).
- **Semantics:** three series on forecast plots are comparable in log-target space; VIX baseline is **`log(max(vix, eps))`** aligned with project convention. Loss plots show train/val curves per epoch and a flat **test** horizontal line at **`test_loss.json`**'s scalar (**`metric`** matches training unless **`[model.evaluate].test_metric`** overrides).
- **Skip semantics:** all paths returned by **`expected_paths_for_summarize(cfg)`** must exist and be non-empty (see **`mss.pipeline.completeness.analysis_summarize_loss_figure_semantically_complete`**). **`loss_figure`** inputs include valid **`processed/summaries/summary_table.csv`** and, when **`draw_hypothesis_table`** is true, **`processed/summaries/hypothesis_tests.csv`**.

### 12b) `processed/thesis_exhibits/` (thesis manuscript exports)

- **Producer:** `thesis.export` / step **`build`** — `mss.thesis.export.run_thesis_export`
- **Consumers:** thesis / manuscript tooling (not consumed by other pipeline steps)
- **Path root:** `data/<run_id>/processed/thesis_exhibits/`
- **Layout:**
  - **`main/tables/`** — locked main-text tables (`M1_*`, `M3_*`, `M4_*` as `.csv` + `.tex`)
  - **`main/figures/`** — `M2_test_forecast_paths.png` (copy of test forecast figure), `M5_mse_log_by_vol_quintile.{png,pdf}`
  - **`appendix/tables/`** — **CSV only** (no appendix `.tex`). **Thesis-facing** filenames: `Table_A01_*` (data/sources), `Table_B01_*` / `Table_B02_*` (configuration + hypothesis map), `Table_C01_*`–`Table_C03_*` (formal evaluation displays), `Table_D01_*`–`Table_D03_*` (calibration wide + quintile). **Archival / pipeline** copies use the **`OLD_`** prefix (e.g. `OLD_A1_*`, `OLD_Table_T05_*`). See `manifest/thesis_table_index.md`.
  - **`appendix/figures/`** — `A4_universe_diagnostics_plate.{png,pdf}` (scripted 3-panel composition from existing `figures/universe_churn/*.png`); duplicate **`Figure_F01_universe_diagnostics_plate.{png,pdf}`** (cite **Figure F01**).
  - **`manifest/`** — `thesis_exhibits_manifest.{md,json}` (includes **`thesis_label`** when set), `thesis_table_index.md`, `Reproducibility_metadata.json` (paths/commands only)
- **Semantics:** derived from existing **`summaries/`** and **`figures/`**; strict validation enforces expected row counts and label sets for the locked exhibit spec. **`--overwrite thesis.export`** removes and rebuilds **`thesis_exhibits/`** only.
- **Skip semantics:** step is complete when all paths returned by **`mss.thesis.export.expected_paths_for_thesis_export(cfg)`** exist and are non-empty (see **`mss.pipeline.completeness.thesis_export_semantically_complete`**).

### 13) Artifacts not produced by the current orchestrator
- The following remain **out of scope**: **`metrics_summary.json`**, **`processed/tables/eval_*.csv`** (legacy naming), pairwise JSON packs not wired to the orchestrator. **Formal regression tables** are now **`summaries/regression_*.csv`** as above. Standalone helpers in **`mss.analysis.report`** / **`pairwise.py`** remain optional and are not part of **`thesis.export`**.

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
- document intentional behavioral deviations in [`legacy/reproducibility.md`](legacy/reproducibility.md)
