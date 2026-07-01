"""Build minimal analysis figures from training/evaluation artifacts."""



from __future__ import annotations



import json

import sys

from pathlib import Path



import pandas as pd

import math

import numpy as np



from mss.analysis.config import load_analysis_summarize_config

from mss.analysis.plot_utils import save_line_plot, save_summary_barplot

from mss.evaluation.checks import (
    check_hypothesis_tests,
    forecasts_path,
    hypothesis_tests_path,
    summary_table_path,
    test_loss_path,
)
from mss.processed.paths import (
    FORECAST_VS_BENCHMARKS,
    figures_universe_churn_dir,
    forecast_comparison_path,
    hypothesis_table_figure_path,
    loss_figure_path_linear,
    loss_figure_path_log,
    metrics_universe_dir,
    summary_barplot_path,
    universe_churn_summary_csv_path,
    universe_rankbucket_replacement_heatmap_path,
    universe_rankbucket_replacement_long_csv_path,
    universe_tenure_distribution_csv_path,
    universe_tenure_distribution_path,
    universe_turnover_timeseries_csv_path,
    universe_turnover_timeseries_path,
)
from mss.processed.readmes import ensure_processed_readmes

from mss.evaluation.forecasts_targets import merge_forecasts_with_vix

from mss.evaluation.config import load_model_evaluate_config

from mss.io.config import ResolvedConfig

from mss.model_train.checks import training_state_path


_FORECAST_Y_COLS = ["y_true", "y_pred_model", "y_pred_vix"]

_FORECAST_COLORS = {

    "y_true": "#1a1a1a",

    "y_pred_model": "tab:blue",

    "y_pred_vix": "tab:red",

}

_FORECAST_LABELS = {

    "y_true": "Target",

    "y_pred_model": "Model",

    "y_pred_vix": "VIX baseline",

}





def expected_paths_for_summarize(cfg: ResolvedConfig) -> list[Path]:

    """Artifacts produced by analysis.summarize for this config (completeness / resume)."""

    proc = Path(cfg.processed_dir)

    paths: list[Path] = [forecast_comparison_path(proc, "all")]

    me = load_model_evaluate_config(cfg.source_config_path)

    for sp in me.splits:

        if sp in ("train", "val", "test"):

            paths.append(forecast_comparison_path(proc, sp))

    paths.append(loss_figure_path_linear(proc))

    paths.append(loss_figure_path_log(proc))

    paths.append(summary_barplot_path(proc))

    # Universe churn diagnostics (computed from graph.prepare universe.parquet).
    paths.append(universe_turnover_timeseries_path(proc))
    paths.append(universe_rankbucket_replacement_heatmap_path(proc))
    paths.append(universe_tenure_distribution_path(proc))
    paths.append(universe_turnover_timeseries_csv_path(proc))
    paths.append(universe_rankbucket_replacement_long_csv_path(proc))
    paths.append(universe_tenure_distribution_csv_path(proc))
    paths.append(universe_churn_summary_csv_path(proc))

    if load_analysis_summarize_config(cfg.source_config_path).draw_hypothesis_table:

        paths.append(hypothesis_table_figure_path(proc))

    return paths





def _load_training_history(cfg: ResolvedConfig) -> pd.DataFrame:

    state_path = training_state_path(cfg)

    if not state_path.is_file():

        raise FileNotFoundError(

            "analysis.summarize needs interim/model_train/training_state.json. Run model.train first."

        )

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    metrics = payload.get("metrics_history", [])

    df = pd.DataFrame(metrics)

    if df.empty or not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):

        raise ValueError("training_state.json is missing metrics_history with epoch/train_loss/val_loss")

    return df.sort_values("epoch").copy()





def _load_forecasts_with_vix(cfg: ResolvedConfig) -> pd.DataFrame:

    merged = merge_forecasts_with_vix(cfg)

    return merged.drop(columns=["vix"])





def _forecast_split_vline_dates(df: pd.DataFrame) -> list[pd.Timestamp]:

    split_order = ("train", "val", "test")

    starts: list[pd.Timestamp] = []

    for sp in split_order:

        sub = df[df["split"].astype(str) == sp]

        if len(sub) == 0:

            continue

        starts.append(pd.Timestamp(sub["date"].min()))

    if len(starts) < 2:

        return []

    return starts[1:]





def _skip_output(path: Path, *, overwrite: bool) -> bool:

    return not overwrite and path.is_file() and path.stat().st_size > 0





def run_build_forecast_comparison_figures(cfg: ResolvedConfig, *, overwrite: bool) -> None:

    """Time-series plots: target vs model vs log(VIX) for all splits and per configured split."""

    processed = Path(cfg.processed_dir)
    ensure_processed_readmes(processed)

    fcfg = load_analysis_summarize_config(cfg.source_config_path)

    eval_cfg = load_model_evaluate_config(cfg.source_config_path)

    df = _load_forecasts_with_vix(cfg)

    y_label = "log RV (model target space)"



    out_all = forecast_comparison_path(processed, "all")

    if not _skip_output(out_all, overwrite=overwrite):

        sd = df.sort_values("date")

        save_line_plot(

            sd,

            x_col="date",

            y_cols=_FORECAST_Y_COLS,

            title="Target vs model vs VIX baseline (all splits)",

            y_label=y_label,

            out_path=out_all,

            dpi=fcfg.dpi,

            colors=_FORECAST_COLORS,

            labels=_FORECAST_LABELS,

            vline_dates=_forecast_split_vline_dates(df),

        )



    for sp in eval_cfg.splits:

        if sp not in ("train", "val", "test"):

            continue

        out = forecast_comparison_path(processed, sp)

        if _skip_output(out, overwrite=overwrite):

            continue

        sub = df[df["split"].astype(str) == sp]

        if len(sub) == 0:

            if fcfg.verbose:

                print(

                    f"analysis.summarize: skip forecast figure for split={sp!r} (no rows).",

                    file=sys.stderr,

                )

            continue

        sub = sub.sort_values("date")

        save_line_plot(

            sub,

            x_col="date",

            y_cols=_FORECAST_Y_COLS,

            title=f"Target vs model vs VIX baseline ({sp})",

            y_label=y_label,

            out_path=out,

            dpi=fcfg.dpi,

            colors=_FORECAST_COLORS,

            labels=_FORECAST_LABELS,

        )





def _loss_figure_title(metric: str | None) -> str:
    labels = {
        "mse_log": "MSE(log)",
        "mae_log": "MAE(log)",
        "smooth_l1_log": "SmoothL1(log)",
        "qlike_level": "QLIKE(level)",
    }
    m = (metric or "mse_log").strip().lower()
    lab = labels.get(m, m)
    return f"Training/validation loss with test {lab} reference"


def run_build_loss_figure(cfg: ResolvedConfig, *, overwrite: bool) -> None:

    """Linear- and log-scaled loss-over-epochs figures with horizontal test-loss line."""

    processed = Path(cfg.processed_dir)

    out_linear = loss_figure_path_linear(processed)

    out_log = loss_figure_path_log(processed)

    if _skip_output(out_linear, overwrite=overwrite) and _skip_output(out_log, overwrite=overwrite):

        return



    fcfg = load_analysis_summarize_config(cfg.source_config_path)

    history = _load_training_history(cfg)



    tp = test_loss_path(cfg)

    if not tp.is_file():

        raise FileNotFoundError(

            "analysis.summarize needs processed/test_loss.json. Run model.evaluate first."

        )

    test_payload = json.loads(tp.read_text(encoding="utf-8"))

    test_loss = float(test_payload.get("value"))

    history = history.copy()

    history["test_loss"] = test_loss

    title = _loss_figure_title(test_payload.get("metric"))



    if not _skip_output(out_linear, overwrite=overwrite):

        save_line_plot(

            history,

            x_col="epoch",

            y_cols=["train_loss", "val_loss", "test_loss"],

            title=title,

            y_label="loss",

            out_path=out_linear,

            dpi=fcfg.dpi,

            y_scale="linear",

        )

    if not _skip_output(out_log, overwrite=overwrite):

        save_line_plot(

            history,

            x_col="epoch",

            y_cols=["train_loss", "val_loss", "test_loss"],

            title=title,

            y_label="loss (log scale)",

            out_path=out_log,

            dpi=fcfg.dpi,

            y_scale="log",

        )





def run_build_summary_barplot(cfg: ResolvedConfig, *, overwrite: bool) -> None:

    """Grouped bar chart from processed/metrics/descriptive/summary_table.csv."""

    processed = Path(cfg.processed_dir)

    out = summary_barplot_path(processed)

    if _skip_output(out, overwrite=overwrite):

        return

    fcfg = load_analysis_summarize_config(cfg.source_config_path)

    st = summary_table_path(cfg)

    save_summary_barplot(st, out_path=out, dpi=fcfg.dpi)


def run_build_hypothesis_table_figure(cfg: ResolvedConfig, *, overwrite: bool) -> None:

    """PNG table of key columns from hypothesis_tests.csv (formal inference)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fcfg = load_analysis_summarize_config(cfg.source_config_path)

    if not fcfg.draw_hypothesis_table:

        return

    processed = Path(cfg.processed_dir)

    out = hypothesis_table_figure_path(processed)

    if _skip_output(out, overwrite=overwrite):

        return

    ht = hypothesis_tests_path(cfg)

    if not check_hypothesis_tests(ht).get("passed"):

        raise FileNotFoundError(

            "analysis.summarize hypothesis table needs processed/metrics/formal/hypothesis_tests.csv "

            "(run model.evaluate run_hypothesis_tests)."

        )

    df = pd.read_csv(ht)

    display_cols = [

        c

        for c in (

            "hypothesis_id",

            "inference_procedure",

            "statistic_type",

            "sample",

            "loss_name",

            "tail",

            "better_model",

            "coefficient_tested",

            "p_value_primary",

            "p_value_two_sided",

            "hac_max_lags",

            "n_obs",

            "subsample_operational",

            "effective_sample_start",

            "effective_sample_end",

            "rows_removed_vs_full_test",

        )

        if c in df.columns

    ]

    sub = df[display_cols].copy()

    for c in sub.columns:

        sub[c] = sub[c].astype(str)

    sub = sub.head(50)

    fig_h = min(22, max(4, 0.35 * len(sub) + 2))

    fig, ax = plt.subplots(figsize=(14, fig_h))

    ax.axis("off")

    ax.set_title("Formal hypothesis tests (see hypothesis_tests.csv for full detail)")

    tbl = ax.table(

        cellText=sub.values,

        colLabels=list(sub.columns),

        loc="center",

        cellLoc="left",

    )

    tbl.auto_set_font_size(False)

    tbl.set_fontsize(6)

    tbl.scale(1.0, 1.2)

    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=fcfg.dpi, bbox_inches="tight")

    plt.close(fig)


def _compute_month_boundary_turnover(
    universe_df: pd.DataFrame, feature_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """Turnover on the first feature date of each calendar month vs prior month's last date."""
    permnos_by_date = {pd.Timestamp(d): set(g["permno"].unique()) for d, g in universe_df.groupby("date")}
    rows: list[dict[str, float | int | pd.Timestamp | str]] = []
    for i in range(1, len(feature_dates)):
        t_cur = feature_dates[i]
        t_prev = feature_dates[i - 1]
        if t_cur.year == t_prev.year and t_cur.month == t_prev.month:
            continue
        prev_set = permnos_by_date.get(t_prev, set())
        cur_set = permnos_by_date.get(t_cur, set())
        universe_size = len(cur_set)
        replaced_count = len(cur_set - prev_set)
        turnover_rate = float(replaced_count) / universe_size if universe_size > 0 else float("nan")
        rows.append(
            {
                "date": t_cur,
                "month": f"{t_cur.year:04d}-{t_cur.month:02d}",
                "universe_size": universe_size,
                "replaced_count": replaced_count,
                "turnover_rate": turnover_rate,
            }
        )
    return pd.DataFrame(rows)


def _assert_supported_universe_churn_mode(cfg: ResolvedConfig) -> str:
    from mss.graph.config import load_graph_config

    gc = load_graph_config(cfg.source_config_path)
    mode = gc.universe_mode
    if mode not in ("fixed_replace", "monthly_rebalance"):
        raise ValueError(
            f"Universe churn figures support universe_mode in ('fixed_replace', 'monthly_rebalance'); "
            f"got {mode!r}."
        )
    return mode


def _load_universe_for_churn(universe_path: Path) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    if not universe_path.is_file():
        raise FileNotFoundError(f"universe_churn requires universe.parquet: {universe_path}")

    df = pd.read_parquet(universe_path)
    required = {"date", "permno", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe.parquet missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out = out.dropna(subset=["date", "permno", "rank"])
    out["permno"] = out["permno"].astype("int64")
    out["rank"] = out["rank"].astype("int64")
    out = out.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)

    # `pd.to_datetime(...).unique()` returns values; sort them as actual timestamps.
    feature_dates = pd.to_datetime(out["date"].unique())
    feature_dates = pd.to_datetime(np.sort(feature_dates.to_numpy(dtype="datetime64[ns]")))
    feature_dates = feature_dates.to_list()
    if len(feature_dates) < 2:
        raise ValueError("universe_churn requires at least two feature dates")

    return out, feature_dates


def _compute_turnover_rates(universe_df: pd.DataFrame, feature_dates: list[pd.Timestamp]) -> pd.DataFrame:
    permnos_by_date = {pd.Timestamp(d): set(g["permno"].unique()) for d, g in universe_df.groupby("date")}

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for i in range(1, len(feature_dates)):
        t_prev = feature_dates[i - 1]
        t_cur = feature_dates[i]
        prev_set = permnos_by_date.get(t_prev, set())
        cur_set = permnos_by_date.get(t_cur, set())
        universe_size = len(cur_set)
        replaced_count = len(cur_set - prev_set)
        turnover_rate = float(replaced_count) / universe_size if universe_size > 0 else float("nan")
        rows.append(
            {
                "date": t_cur,
                "universe_size": universe_size,
                "replaced_count": replaced_count,
                "turnover_rate": turnover_rate,
            }
        )
    return pd.DataFrame(rows)


def _compute_rankbucket_replacement_matrix(
    universe_df: pd.DataFrame,
    feature_dates: list[pd.Timestamp],
    *,
    n_buckets: int = 10,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")

    max_rank = int(universe_df["rank"].max())
    bucket_size = max(1, int(math.ceil(max_rank / n_buckets)))

    d = universe_df.copy()
    d["rank_bucket"] = ((d["rank"] - 1) // bucket_size).clip(0, n_buckets - 1).astype("int64")

    transitions = feature_dates[1:]
    n_cols = len(transitions)
    matrix = np.full((n_buckets, n_cols), np.nan, dtype=float)

    for col_idx, t_cur in enumerate(transitions):
        t_prev = feature_dates[col_idx]
        cur_sub = d.loc[d["date"] == t_cur, ["permno", "rank_bucket"]]
        prev_sub = d.loc[d["date"] == t_prev, ["permno", "rank_bucket"]]

        cur_sets = {int(b): set(subb["permno"].unique()) for b, subb in cur_sub.groupby("rank_bucket")}
        prev_sets = {int(b): set(subb["permno"].unique()) for b, subb in prev_sub.groupby("rank_bucket")}

        for b in range(n_buckets):
            cur_set = cur_sets.get(b)
            if not cur_set:
                continue
            prev_set = prev_sets.get(b, set())
            replaced_count = len(cur_set - prev_set)
            matrix[b, col_idx] = float(replaced_count) / float(len(cur_set))

    bucket_ranges: list[tuple[int, int]] = []
    for b in range(n_buckets):
        lo = b * bucket_size + 1
        hi = min((b + 1) * bucket_size, max_rank)
        bucket_ranges.append((lo, hi))

    return matrix, bucket_ranges, bucket_size


def _compute_tenure_streak_lengths(universe_df: pd.DataFrame, feature_dates: list[pd.Timestamp]) -> list[int]:
    date_to_idx = {d: i for i, d in enumerate(feature_dates)}
    d = universe_df.copy()
    d["date_idx"] = d["date"].map(date_to_idx)
    d = d.dropna(subset=["date_idx"])
    d["date_idx"] = d["date_idx"].astype("int64")

    streaks: list[int] = []
    for _permno, sub in d.groupby("permno"):
        idxs = sub["date_idx"].to_numpy()
        if idxs.size == 0:
            continue
        idxs = np.sort(idxs)
        if idxs.size == 1:
            streaks.append(1)
            continue

        breaks = np.where(np.diff(idxs) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(idxs)]))
        lens = (ends - starts).astype(int).tolist()
        streaks.extend(lens)
    return streaks


def run_build_universe_churn_figures(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Universe replacement diagnostics from `interim/graphs/universe.parquet`."""
    universe_mode = _assert_supported_universe_churn_mode(cfg)

    fcfg = load_analysis_summarize_config(cfg.source_config_path)

    processed = Path(cfg.processed_dir)
    out_turnover = universe_turnover_timeseries_path(processed)
    out_heatmap = universe_rankbucket_replacement_heatmap_path(processed)
    out_tenure = universe_tenure_distribution_path(processed)
    out_turnover_csv = universe_turnover_timeseries_csv_path(processed)
    out_heatmap_long_csv = universe_rankbucket_replacement_long_csv_path(processed)
    out_tenure_csv = universe_tenure_distribution_csv_path(processed)
    out_summary_csv = universe_churn_summary_csv_path(processed)

    if (
        _skip_output(out_turnover, overwrite=overwrite)
        and _skip_output(out_heatmap, overwrite=overwrite)
        and _skip_output(out_tenure, overwrite=overwrite)
        and _skip_output(out_turnover_csv, overwrite=overwrite)
        and _skip_output(out_heatmap_long_csv, overwrite=overwrite)
        and _skip_output(out_tenure_csv, overwrite=overwrite)
        and _skip_output(out_summary_csv, overwrite=overwrite)
    ):
        return

    universe_path = Path(cfg.interim_dir) / "graphs" / "universe.parquet"
    universe_df, feature_dates = _load_universe_for_churn(universe_path)

    turnover_df = _compute_turnover_rates(universe_df, feature_dates)
    heat_matrix, bucket_ranges, bucket_size = _compute_rankbucket_replacement_matrix(
        universe_df, feature_dates, n_buckets=10
    )
    tenure_streaks = _compute_tenure_streak_lengths(universe_df, feature_dates)
    tenure_arr = np.asarray(tenure_streaks, dtype=int)
    if tenure_arr.size == 0:
        raise ValueError("universe_churn: no tenure streaks produced from universe data")

    # --- CSV outputs (numeric summaries) ---
    metrics_universe_dir(processed).mkdir(parents=True, exist_ok=True)

    if not _skip_output(out_turnover_csv, overwrite=overwrite):
        turnover_df.to_csv(out_turnover_csv, index=False)

    if not _skip_output(out_heatmap_long_csv, overwrite=overwrite):
        dates_cols = feature_dates[1:]
        rows: list[dict[str, object]] = []
        n_buckets = heat_matrix.shape[0]
        n_cols = heat_matrix.shape[1]
        for col_idx in range(n_cols):
            d_cur = pd.Timestamp(dates_cols[col_idx])
            for b in range(n_buckets):
                lo, hi = bucket_ranges[b]
                rows.append(
                    {
                        "date": d_cur,
                        "rank_bucket": int(b),
                        "rank_lo": int(lo),
                        "rank_hi": int(hi),
                        "bucket_size": int(bucket_size),
                        "replacement_fraction": float(heat_matrix[b, col_idx])
                        if np.isfinite(heat_matrix[b, col_idx])
                        else float("nan"),
                    }
                )
        pd.DataFrame(rows).to_csv(out_heatmap_long_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = figures_universe_churn_dir(processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Turnover timeseries
    if not _skip_output(out_turnover, overwrite=overwrite):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(turnover_df["date"], turnover_df["turnover_rate"], color="tab:blue", linewidth=1.5)
        ax.set_title("Universe replacement: turnover rate over feature dates")
        ax.set_xlabel("Feature date")
        ax.set_ylabel("Turnover rate (replaced / universe size)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_turnover, dpi=fcfg.dpi, bbox_inches="tight")
        plt.close(fig)

    # 2) Rank-bucket heatmap
    if not _skip_output(out_heatmap, overwrite=overwrite):
        n_buckets = heat_matrix.shape[0]
        n_cols = heat_matrix.shape[1]
        dates_cols = feature_dates[1:]

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")

        # Make the figure wider only when needed.
        fig_w = min(24, max(10, 0.03 * n_cols + 10))
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        im = ax.imshow(heat_matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap=cmap)

        y_labels = [f"{lo}-{hi}" for (lo, hi) in bucket_ranges]
        ax.set_yticks(list(range(n_buckets)))
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("Universe rank bucket (inclusive)")
        ax.set_xlabel("Feature date (current)")

        if n_cols > 0:
            step = max(1, n_cols // 8)
            positions = list(range(0, n_cols, step))
            xticklabels = [pd.Timestamp(dates_cols[i]).strftime("%Y-%m-%d") for i in positions]
            ax.set_xticks(positions)
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Fraction replaced within the rank bucket")

        fig.tight_layout()
        fig.savefig(out_heatmap, dpi=fcfg.dpi, bbox_inches="tight")
        plt.close(fig)

    # 3) Tenure / survival distribution
    if not _skip_output(out_tenure, overwrite=overwrite):
        cap = int(min(40, int(tenure_arr.max())))

        counts = [int((tenure_arr == k).sum()) for k in range(1, cap + 1)]
        overflow = int((tenure_arr > cap).sum())

        k_vals = np.arange(1, cap + 1, dtype=int)
        survival = np.asarray([(tenure_arr >= k).mean() for k in k_vals], dtype=float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [1, 1.2]})

        ax1.bar(k_vals, counts, color="tab:blue", alpha=0.85)
        if overflow > 0:
            ax1.bar([cap + 1], [overflow], color="tab:gray", alpha=0.7)
            ax1.set_xticks(list(k_vals) + [cap + 1])
            ax1.set_xticklabels([str(k) for k in k_vals.tolist()] + [f">{cap}"])
        ax1.set_title("Universe tenure (consecutive feature dates present)")
        ax1.set_xlabel("Tenure length (steps)")
        ax1.set_ylabel("Streak count")
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.plot(k_vals, survival, color="tab:orange", linewidth=1.8)
        ax2.set_title("Survival curve: P(tenure >= k)")
        ax2.set_xlabel("k (steps)")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)

        mean_tenure = float(tenure_arr.mean())
        median_tenure = float(np.median(tenure_arr))
        ax2.text(
            0.99,
            0.02,
            f"n_streaks={len(tenure_arr)}\\nmean={mean_tenure:.2f}, median={median_tenure:.2f}",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(out_tenure, dpi=fcfg.dpi, bbox_inches="tight")
        plt.close(fig)

    # Tenure CSV and one-row overall summary CSV.
    cap = int(min(40, int(tenure_arr.max())))
    k_vals = np.arange(1, cap + 1, dtype=int)
    counts = np.asarray([int((tenure_arr == k).sum()) for k in k_vals], dtype=int)
    n_streaks = int(len(tenure_arr))
    survival = np.asarray([(tenure_arr >= k).mean() for k in k_vals], dtype=float)

    if not _skip_output(out_tenure_csv, overwrite=overwrite):
        td = pd.DataFrame(
            {
                "k": k_vals.astype(int),
                "streak_count": counts.astype(int),
                "streak_fraction": (counts / max(1, n_streaks)).astype(float),
                "survival_p_ge_k": survival.astype(float),
            }
        )
        td.to_csv(out_tenure_csv, index=False)

    if not _skip_output(out_summary_csv, overwrite=overwrite):
        from mss.graph.config import load_graph_config

        gc = load_graph_config(cfg.source_config_path)
        sizes_by_date = (
            universe_df.groupby("date", as_index=False)["permno"]
            .nunique()
            .rename(columns={"permno": "universe_size"})
        )
        max_rank = int(universe_df["rank"].max())

        tr = turnover_df["turnover_rate"].to_numpy(dtype=float)
        rc = turnover_df["replaced_count"].to_numpy(dtype=float)

        heat_vals = heat_matrix[np.isfinite(heat_matrix)].astype(float)
        if heat_vals.size == 0:
            heat_vals = np.asarray([float("nan")], dtype=float)

        def _q(x: np.ndarray, q: float) -> float:
            x = x[np.isfinite(x)]
            return float(np.quantile(x, q)) if x.size else float("nan")

        summary_row = {
            "run_id": cfg.run_id,
            "universe_mode": gc.universe_mode,
            "rebalance_freq": gc.rebalance_freq,
            "n_feature_dates": int(len(feature_dates)),
            "n_transitions": int(max(0, len(feature_dates) - 1)),
            "avg_universe_size": float(sizes_by_date["universe_size"].mean()) if len(sizes_by_date) else float("nan"),
            "turnover_rate_mean": float(np.nanmean(tr)),
            "turnover_rate_median": float(np.nanmedian(tr)),
            "turnover_rate_p05": _q(tr, 0.05),
            "turnover_rate_p95": _q(tr, 0.95),
            "turnover_rate_max": float(np.nanmax(tr)) if np.isfinite(tr).any() else float("nan"),
            "replaced_count_mean": float(np.nanmean(rc)),
            "replaced_count_median": float(np.nanmedian(rc)),
            "replaced_count_p95": _q(rc, 0.95),
            "replaced_count_max": float(np.nanmax(rc)) if np.isfinite(rc).any() else float("nan"),
            "n_streaks": int(n_streaks),
            "tenure_mean": float(np.mean(tenure_arr)),
            "tenure_median": float(np.median(tenure_arr)),
            "tenure_p90": float(np.quantile(tenure_arr, 0.90)),
            "tenure_max": int(tenure_arr.max()),
            "tenure_cap_used": int(cap),
            "rankbucket_n_buckets": int(heat_matrix.shape[0]),
            "rankbucket_bucket_size": int(bucket_size),
            "rankbucket_max_rank": int(max_rank),
            "rankbucket_replacement_mean": float(np.nanmean(heat_vals)),
            "rankbucket_replacement_p95": _q(heat_vals, 0.95),
            "rankbucket_replacement_max": float(np.nanmax(heat_vals)) if np.isfinite(heat_vals).any() else float("nan"),
        }
        if universe_mode == "monthly_rebalance":
            month_df = _compute_month_boundary_turnover(universe_df, feature_dates)
            if not month_df.empty:
                mb = month_df["turnover_rate"].to_numpy(dtype=float)
                summary_row["month_boundary_turnover_mean"] = float(np.nanmean(mb))
                summary_row["month_boundary_turnover_median"] = float(np.nanmedian(mb))
                summary_row["month_boundary_turnover_max"] = (
                    float(np.nanmax(mb)) if np.isfinite(mb).any() else float("nan")
                )
                summary_row["n_month_boundaries"] = int(len(month_df))
            else:
                summary_row["month_boundary_turnover_mean"] = float("nan")
                summary_row["month_boundary_turnover_median"] = float("nan")
                summary_row["month_boundary_turnover_max"] = float("nan")
                summary_row["n_month_boundaries"] = 0
        pd.DataFrame([summary_row]).to_csv(out_summary_csv, index=False)


