"""Shared plotting helpers for analysis figures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_line_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_label: str,
    out_path: Path,
    dpi: int,
    y_scale: Literal["linear", "log"] = "linear",
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    vline_dates: list[pd.Timestamp] | None = None,
) -> None:
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in y_cols:
        if c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors="coerce").astype(float)
        if y_scale == "log":
            series = series.clip(lower=1e-30)
        color = colors.get(c) if colors else None
        label = labels.get(c, c) if labels else c
        ax.plot(df[x_col], series, label=label, alpha=0.9, color=color)
    if vline_dates:
        for vd in vline_dates:
            ax.axvline(pd.Timestamp(vd), color="0.5", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[x_col], df[y_col], s=10, alpha=0.4)
    mn = min(float(df[x_col].min()), float(df[y_col].min()))
    mx = max(float(df[x_col].max()), float(df[y_col].max()))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_hist_overlay(
    s1: pd.Series,
    s2: pd.Series,
    *,
    label1: str,
    label2: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s1.dropna(), bins=40, alpha=0.5, label=label1, density=True)
    ax.hist(s2.dropna(), bins=40, alpha=0.5, label=label2, density=True)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str | None,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    if hue_col and hue_col in df.columns:
        piv = df.pivot(index=x_col, columns=hue_col, values=y_col).fillna(0.0)
        piv.plot(kind="bar", ax=ax)
    else:
        ax.bar(df[x_col].astype(str), df[y_col].astype(float))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_forest_plot(
    df: pd.DataFrame,
    *,
    label_col: str,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    _ensure_parent(out_path)
    dd = df.reset_index(drop=True).copy()
    y = range(len(dd))
    fig, ax = plt.subplots(figsize=(9, max(4, len(dd) * 0.3)))
    ax.errorbar(
        dd[mean_col].astype(float),
        y,
        xerr=[
            dd[mean_col].astype(float) - dd[lo_col].astype(float),
            dd[hi_col].astype(float) - dd[mean_col].astype(float),
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(dd[label_col].astype(str).tolist())
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_barplot(
    summary_path: Path,
    *,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Grouped bar chart from processed/summaries/summary_table.csv (all sample slices
    present in the file). Metrics: mse_log, mae_log, mse, mae, qlike.
    """
    from mss.evaluation.summary_table import SUMMARY_SAMPLE_ORDER

    _ensure_parent(out_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary table not found: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {"model", "sample", "mse_log", "mae_log", "mse", "mae", "qlike"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary_table.csv missing columns: {missing}")
    if df.empty:
        raise ValueError("summary_table.csv has no rows")

    metrics = ["mse_log", "mae_log", "mse", "mae", "qlike"]
    metric_labels = ["MSE (log)", "MAE (log)", "MSE", "MAE", "QLIKE"]
    models = sorted(df["model"].astype(str).str.lower().unique().tolist())
    samples_order = [s for s in SUMMARY_SAMPLE_ORDER if s in set(df["sample"].astype(str))]
    if not samples_order:
        raise ValueError("no recognized sample labels in summary table")
    bar_order = [(m, s) for m in models for s in samples_order]
    n_metrics = len(metrics)
    n_bars = len(bar_order)
    raw_values = np.full((n_metrics, n_bars), np.nan, dtype=np.float64)
    heights = np.full_like(raw_values, np.nan, dtype=np.float64)

    sample_labels = {
        "train": "train",
        "val": "val",
        "excl_crisis": "train excl crisis window",
        "full_test": "full test",
        "excl_2020": "test excl 2020",
    }
    base_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(models))))
    colors: dict[tuple[str, str], tuple] = {}
    alphas: dict[tuple[str, str], float] = {}
    for idx, model in enumerate(models):
        base = base_colors[idx % len(base_colors)]
        for si, sample in enumerate(samples_order):
            colors[(model, sample)] = base
            if len(samples_order) <= 1:
                alphas[(model, sample)] = 1.0
            else:
                alphas[(model, sample)] = 0.45 + 0.55 * (si / (len(samples_order) - 1))

    for mi, metric in enumerate(metrics):
        vals: list[float] = []
        for bi, (model, sample) in enumerate(bar_order):
            row = df[(df["model"].astype(str).str.lower() == model) & (df["sample"].astype(str) == sample)]
            val = float(row.iloc[0][metric]) if not row.empty else float("nan")
            raw_values[mi, bi] = val
            vals.append(val)
        arr = np.array(vals, dtype=np.float64)
        if np.all(~np.isfinite(arr)):
            continue
        best = np.nanmin(arr)
        if not np.isfinite(best) or best <= 0:
            norm = np.ones_like(arr)
        else:
            norm = arr / best
        heights[mi, :] = norm

    group_idx = np.arange(n_metrics)
    group_width = 0.8
    bar_width = group_width / max(1, n_bars)
    fig_w = min(22, 8 + n_bars * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    for bi, (model, sample) in enumerate(bar_order):
        offset = (bi - (n_bars - 1) / 2) * bar_width
        x = group_idx + offset
        h = heights[:, bi]
        c = colors[(model, sample)]
        a = alphas[(model, sample)]
        sl = sample_labels.get(sample, sample)
        label = f"{model.upper()} ({sl})"
        bars = ax.bar(
            x,
            h,
            width=bar_width * 0.9,
            color=c,
            alpha=a,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        for mj, bar in enumerate(bars):
            val_raw = raw_values[mj, bi]
            if not np.isfinite(val_raw):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val_raw:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    handles, labels_leg = ax.get_legend_handles_labels()
    seen: set[str] = set()
    u_h, u_l = [], []
    for h, lab in zip(handles, labels_leg):
        if lab not in seen:
            seen.add(lab)
            u_h.append(h)
            u_l.append(lab)
    ax.legend(u_h, u_l, loc="upper right", fontsize=7)
    ax.set_xticks(group_idx)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Metric value (normalized by best per metric group)")
    ax.set_title("Forecast accuracy: GNN vs VIX (summary metrics by sample)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_barplot_test_compare(
    summary_df: pd.DataFrame,
    *,
    out_path: Path,
    dpi: int,
) -> None:
    """Grouped bar chart: model vs vix on test_full vs test_excl_2020 sample rows."""
    _ensure_parent(out_path)
    samples_want = ("test_full", "test_excl_2020")
    df = summary_df[
        summary_df["sample"].isin(samples_want) & summary_df["model"].isin(["model", "vix"])
    ].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no test_full / test_excl_2020 rows", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    metrics = ["mse", "mae", "rmse", "qlike_level"]
    metric_labels = ["MSE (log)", "MAE (log)", "RMSE (log)", "QLIKE (level)"]
    models = sorted(df["model"].unique().tolist())
    samples_order = [s for s in samples_want if s in df["sample"].values]
    bar_order = [(m, s) for m in models for s in samples_order]

    n_metrics = len(metrics)
    n_bars = len(bar_order)
    raw_values = np.full((n_metrics, n_bars), np.nan, dtype=np.float64)
    heights = np.full_like(raw_values, np.nan, dtype=np.float64)

    base_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(models))))
    colors: dict[tuple[str, str], tuple] = {}
    alphas: dict[tuple[str, str], float] = {}
    for idx, model in enumerate(models):
        base = base_colors[idx % len(base_colors)]
        for sample in samples_order:
            colors[(model, sample)] = base
            alphas[(model, sample)] = 0.75 if sample == "test_full" else 1.0

    for mi, metric in enumerate(metrics):
        vals = []
        for bi, (model, sample) in enumerate(bar_order):
            row = df[(df["model"] == model) & (df["sample"] == sample)]
            val = float(row.iloc[0][metric]) if not row.empty and metric in row.columns else np.nan
            raw_values[mi, bi] = val
            vals.append(val)
        arr = np.array(vals, dtype=np.float64)
        if np.all(~np.isfinite(arr)):
            heights[mi, :] = 0.0
            continue
        best = np.nanmin(arr)
        if not np.isfinite(best) or best <= 0:
            norm = np.ones_like(arr)
        else:
            norm = arr / best
        heights[mi, :] = norm

    group_idx = np.arange(n_metrics)
    group_width = 0.8
    bar_width = group_width / n_bars
    fig, ax = plt.subplots(figsize=(10, 5))
    for bi, (model, sample) in enumerate(bar_order):
        offset = (bi - (n_bars - 1) / 2) * bar_width
        x = group_idx + offset
        h = heights[:, bi]
        c = colors[(model, sample)]
        a = alphas[(model, sample)]
        label = f"{model.upper()} ({'excl 2020' if sample == 'test_excl_2020' else 'full test'})"
        bars = ax.bar(
            x,
            h,
            width=bar_width * 0.9,
            color=c,
            alpha=a,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        for mj, bar in enumerate(bars):
            val_raw = raw_values[mj, bi]
            if not np.isfinite(val_raw):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val_raw:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    handles, labels_leg = ax.get_legend_handles_labels()
    seen: set[str] = set()
    u_h, u_l = [], []
    for h, lab in zip(handles, labels_leg):
        if lab not in seen:
            seen.add(lab)
            u_h.append(h)
            u_l.append(lab)
    ax.legend(u_h, u_l, loc="upper right", fontsize=8)
    ax.set_xticks(group_idx)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Metric value (normalized by best per metric group)")
    ax.set_title("Forecast accuracy on test: full vs excluding 2020")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

