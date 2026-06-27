"""Run minimal model evaluation and write forecast + test loss artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mss.evaluation.checks import (
    check_forecast_panel,
    check_forecasts,
    check_hypothesis_tests,
    check_diagnostics_smoothing,
    check_summary_table,
    check_test_loss,
    diagnostics_smoothing_path,
    forecast_panel_path,
    forecasts_path,
    hypothesis_tests_path,
    regression_incremental_path,
    regression_mz_gnn_path,
    summary_table_path,
    test_loss_path,
)
from mss.evaluation.metrics import aggregate_test_metric_value
from mss.evaluation.summary_table import build_summary_table_rows
from mss.evaluation.config import load_model_evaluate_config
from mss.io.config import ResolvedConfig
from mss.model_train.checks import check_final_metrics_complete, final_metrics_path, model_train_dir
from mss.model_train.log_calibration import LOG_CALIBRATION_JSON, load_log_calibration_json
from mss.model_train.config import load_model_train_config
from mss.model_train.device import get_device
from mss.model_train.manifest_io import load_dataset_manifest
from mss.model_train.models import build_model
from mss.model_train.pyg_data import GraphDateDataset


def _require_train_stack() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'model.evaluate requires PyTorch Geometric. Install with: pip install -e ".[train]"'
        ) from e


def run_score_splits(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Step A1: forward pass only; write processed/forecasts.parquet."""
    _require_train_stack()
    from torch_geometric.loader import DataLoader

    ecfg = load_model_evaluate_config(cfg.source_config_path)
    mtrain_cfg = load_model_train_config(cfg.source_config_path)
    manifest_path = Path(cfg.interim_dir) / "dataset" / "manifest.json"
    manifest = load_dataset_manifest(manifest_path)

    fm_chk = check_final_metrics_complete(final_metrics_path(cfg))
    if not fm_chk.get("passed"):
        raise ValueError("model.evaluate requires completed model.train (final_metrics.json).")

    out_forecasts = forecasts_path(cfg)
    if not overwrite and check_forecasts(out_forecasts).get("passed"):
        return
    out_forecasts.parent.mkdir(parents=True, exist_ok=True)

    splits = ecfg.splits
    split_frames: list[pd.DataFrame] = []

    probe_split = splits[0]
    probe_ds = GraphDateDataset(
        manifest,
        probe_split,
        cache_dir=model_train_dir(cfg) / "graph_cache",
        use_parquet_pushdown=mtrain_cfg.parquet_date_pushdown,
    )
    if len(probe_ds) == 0:
        raise ValueError(f"{probe_split} split has no dates with valid labels.")
    g0 = probe_ds[0]
    in_channels = int(g0.x.shape[1])
    model = build_model(
        mtrain_cfg.model_type,
        in_channels,
        mtrain_cfg.hidden_channels,
        mtrain_cfg.num_gnn_layers,
        mtrain_cfg.pooling,
    )
    ck_path = model_train_dir(cfg) / "checkpoint.pt"
    if not ck_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ck_path}")

    device = get_device(ecfg.device if ecfg.device != "auto" else mtrain_cfg.device)
    bundle = torch.load(ck_path, map_location=device, weights_only=False)
    state = bundle.get("best_model_state_dict") or bundle["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for split in splits:
        ds = GraphDateDataset(
            manifest,
            split,
            cache_dir=model_train_dir(cfg) / "graph_cache",
            use_parquet_pushdown=mtrain_cfg.parquet_date_pushdown,
        )
        if len(ds) == 0:
            if ecfg.verbose:
                print(f"[model.evaluate] skipping split {split}: no rows", file=sys.stderr)
            continue
        loader = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False)
        preds: list[float] = []
        preds_placebo: list[float] = []
        truth: list[float] = []
        iterator = tqdm(
            loader,
            desc=f"Evaluate {split}",
            disable=not ecfg.verbose,
            file=sys.stderr,
        )
        with torch.no_grad():
            for batch in iterator:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
                preds.extend(out.detach().cpu().numpy().astype(float).tolist())
                truth.extend(target.detach().cpu().numpy().astype(float).tolist())
        if ecfg.enable_placebo_ablation:
            ds_placebo = GraphDateDataset(
                manifest,
                split,
                cache_dir=None,
                use_parquet_pushdown=mtrain_cfg.parquet_date_pushdown,
                edge_mode="placebo_permute",
                edge_placebo_seed=ecfg.placebo_edge_seed,
            )
            if len(ds_placebo) != len(ds):
                raise RuntimeError(
                    f"placebo dataset size mismatch for split {split}: {len(ds_placebo)} vs {len(ds)}"
                )
            loader_placebo = DataLoader(ds_placebo, batch_size=ecfg.batch_size, shuffle=False)
            iterator_placebo = tqdm(
                loader_placebo,
                desc=f"Evaluate placebo {split}",
                disable=not ecfg.verbose,
                file=sys.stderr,
            )
            with torch.no_grad():
                for batch in iterator_placebo:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    preds_placebo.extend(out.detach().cpu().numpy().astype(float).tolist())
            if len(preds_placebo) != len(preds):
                raise RuntimeError(
                    f"placebo prediction size mismatch for split {split}: {len(preds_placebo)} vs {len(preds)}"
                )
        else:
            preds_placebo = list(preds)
        dates = [pd.Timestamp(d).normalize() for d in ds.dates]
        if not (len(dates) == len(preds) == len(preds_placebo) == len(truth)):
            raise RuntimeError(f"prediction size mismatch for split {split}")
        split_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "split": split,
                    "sample": f"{split}_full",
                    "y_true": truth,
                    "y_pred_model": preds,
                    "y_pred_placebo": preds_placebo,
                }
            )
        )

    if not split_frames:
        raise ValueError("model.evaluate found no non-empty splits to score.")

    forecasts = pd.concat(split_frames, ignore_index=True).sort_values(["date", "split"])

    if ecfg.apply_log_calibration:
        cal_path = model_train_dir(cfg) / LOG_CALIBRATION_JSON
        cal = load_log_calibration_json(cal_path)
        if cal is not None:
            a = float(cal["a"])
            b = float(cal["b"])
            preds_cal = (a + b * pd.to_numeric(forecasts["y_pred_model"], errors="coerce")).to_numpy(
                dtype=float
            )
            preds_placebo_cal = (
                a + b * pd.to_numeric(forecasts["y_pred_placebo"], errors="coerce")
            ).to_numpy(dtype=float)
            forecasts = forecasts.copy()
            forecasts["y_pred_model"] = preds_cal
            forecasts["y_pred_placebo"] = preds_placebo_cal
            if ecfg.verbose:
                print(
                    f"[model.evaluate] Applied log calibration from {cal_path} (a={a:.6g}, b={b:.6g})",
                    file=sys.stderr,
                )
        elif ecfg.verbose:
            print(
                f"[model.evaluate] apply_log_calibration=true but missing {cal_path}; using raw preds.",
                file=sys.stderr,
            )

    forecasts.to_parquet(out_forecasts, index=False)
    fchk = check_forecasts(out_forecasts)
    if not fchk.get("passed"):
        raise RuntimeError(f"forecasts contract failed after score_splits: {fchk['issues']}")


def run_write_forecast_panel(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Write processed/forecast_panel.parquet (VIX-merged, log/level, per-date losses)."""
    from mss.evaluation.forecast_panel import run_write_forecast_panel as _write

    _write(cfg, overwrite=overwrite)


def run_aggregate_test_loss(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Step A2: compute test split scalar loss (aligned with training metric by default) and write test_loss.json."""
    out_forecasts = forecasts_path(cfg)
    out_test_loss = test_loss_path(cfg)
    if not check_forecasts(out_forecasts).get("passed"):
        raise RuntimeError("aggregate_test_loss requires processed/forecasts.parquet from score_splits")
    if not overwrite and check_test_loss(out_test_loss).get("passed"):
        return

    mtrain_cfg = load_model_train_config(cfg.source_config_path)
    ecfg = load_model_evaluate_config(cfg.source_config_path)
    metric = ecfg.test_metric if ecfg.test_metric is not None else mtrain_cfg.loss

    forecasts = pd.read_parquet(out_forecasts)
    test = forecasts[forecasts["split"].astype(str) == "test"].copy()
    if len(test) == 0:
        raise ValueError("aggregate_test_loss found zero test rows in forecasts.parquet")
    y_true = pd.to_numeric(test["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(test["y_pred_model"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("aggregate_test_loss found non-numeric y_true/y_pred_model values")
    value = aggregate_test_metric_value(
        metric,
        y_true,
        y_pred,
        loss_eps=mtrain_cfg.loss_eps,
        smooth_l1_beta=mtrain_cfg.smooth_l1_beta,
        loss_alpha=mtrain_cfg.loss_alpha,
        qlike_pred_transform=mtrain_cfg.qlike_pred_transform,
        qlike_level_floor=mtrain_cfg.qlike_level_floor,
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "metric": metric,
        "split": "test",
        "n_rows": int(len(test)),
        "value": value,
    }
    out_test_loss.parent.mkdir(parents=True, exist_ok=True)
    out_test_loss.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    chk = check_test_loss(out_test_loss)
    if not chk.get("passed"):
        raise RuntimeError(f"test_loss contract failed: {chk['issues']}")


def run_write_summary_table(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Write processed/summaries/summary_table.csv (GNN vs VIX, multiple sample slices)."""
    out = summary_table_path(cfg)
    out_diag = diagnostics_smoothing_path(cfg)
    if (
        not overwrite
        and check_summary_table(out).get("passed")
        and check_diagnostics_smoothing(out_diag).get("passed")
    ):
        return
    if not check_forecasts(forecasts_path(cfg)).get("passed"):
        raise RuntimeError("write_summary_table requires valid processed/forecasts.parquet")
    if not check_test_loss(test_loss_path(cfg)).get("passed"):
        raise RuntimeError("write_summary_table requires valid processed/test_loss.json")
    fpp = forecast_panel_path(cfg)
    if not check_forecast_panel(fpp).get("passed"):
        raise RuntimeError("write_summary_table requires valid processed/forecast_panel.parquet")

    ecfg = load_model_evaluate_config(cfg.source_config_path)
    panel = pd.read_parquet(fpp)
    merged = panel.assign(
        y_true=panel["y_true_log"],
        y_pred_model=panel["y_pred_model_log"],
        y_pred_placebo=panel["y_pred_placebo_log"],
        y_pred_har=panel["y_pred_har_log"],
        y_pred_vix=panel["y_pred_vix_log"],
    )
    rows = build_summary_table_rows(merged, ecfg)
    if not rows:
        raise ValueError(
            "write_summary_table produced no rows; check splits and exclusion-year masks."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    chk = check_summary_table(out)
    if not chk.get("passed"):
        raise RuntimeError(f"summary_table contract failed: {chk['issues']}")
    diag_rows = _build_smoothing_diagnostics_rows(panel, ecfg)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diag_rows).to_csv(out_diag, index=False)
    dchk = check_diagnostics_smoothing(out_diag)
    if not dchk.get("passed"):
        raise RuntimeError(f"diagnostics_smoothing contract failed: {dchk['issues']}")


def _ols_intercept_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _build_smoothing_diagnostics_rows(panel: pd.DataFrame, ecfg: Any) -> list[dict[str, Any]]:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    years = df["date"].dt.year
    test_excl = set(ecfg.summary_test_exclusion_years)
    samples: list[tuple[str, pd.Series]] = [
        ("full_test", df["split"].astype(str) == "test"),
        ("excl_2020", (df["split"].astype(str) == "test") & ~years.isin(test_excl)),
    ]
    model_cols: dict[str, str] = {
        "gnn": "y_pred_model_log",
        "placebo": "y_pred_placebo_log",
        "har": "y_pred_har_log",
        "vix": "y_pred_vix_log",
    }
    out: list[dict[str, Any]] = []
    for sample_name, mask in samples:
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        y = pd.to_numeric(sub["y_true_log"], errors="coerce").to_numpy(dtype=float)
        y_level = pd.to_numeric(sub["y_true_level"], errors="coerce").to_numpy(dtype=float)
        try:
            q_bins = pd.qcut(y_level, q=5, labels=False, duplicates="drop")
        except ValueError:
            q_bins = pd.Series(np.zeros(len(sub), dtype=int), index=sub.index)
        for model, col in model_cols.items():
            x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(x)
            if int(ok.sum()) == 0:
                continue
            y_ok = y[ok]
            x_ok = x[ok]
            if int(ok.sum()) >= 2:
                intercept, slope = _ols_intercept_slope(y_ok, x_ok)
                corr = float(np.corrcoef(y_ok, x_ok)[0, 1])
            else:
                intercept, slope, corr = float(y_ok[0]), 1.0, 1.0
            out.extend(
                [
                    {"sample": sample_name, "diagnostic": "forecast_variance_log", "model": model, "value": float(np.var(x_ok)), "detail": ""},
                    {"sample": sample_name, "diagnostic": "corr_with_target_log", "model": model, "value": corr, "detail": ""},
                    {"sample": sample_name, "diagnostic": "mz_intercept", "model": model, "value": intercept, "detail": ""},
                    {"sample": sample_name, "diagnostic": "mz_slope", "model": model, "value": slope, "detail": ""},
                ]
            )
            # Volatility quantile diagnostics (MSE log by target-vol quintile)
            q_arr = np.asarray(q_bins)
            for q in sorted(pd.Series(q_arr).dropna().unique().tolist()):
                qmask = (q_arr == q) & ok
                if int(qmask.sum()) == 0:
                    continue
                mse_q = float(np.mean(np.square(y[qmask] - x[qmask])))
                out.append(
                    {
                        "sample": sample_name,
                        "diagnostic": "mse_log_by_target_vol_quantile",
                        "model": model,
                        "value": mse_q,
                        "detail": f"q{int(q) + 1}",
                    }
                )
    if not out:
        raise ValueError("No diagnostics rows produced from forecast panel.")
    return out


def run_hypothesis_tests_step(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """HAC inference for thesis hypotheses; reads only forecast_panel.parquet."""
    out_ht = hypothesis_tests_path(cfg)
    out_mz = regression_mz_gnn_path(cfg)
    out_inc = regression_incremental_path(cfg)
    if (
        not overwrite
        and check_hypothesis_tests(out_ht).get("passed")
        and out_mz.is_file()
        and out_inc.is_file()
    ):
        return
    fpp = forecast_panel_path(cfg)
    if not check_forecast_panel(fpp).get("passed"):
        raise RuntimeError("run_hypothesis_tests requires valid processed/forecast_panel.parquet")

    ecfg = load_model_evaluate_config(cfg.source_config_path)
    panel = pd.read_parquet(fpp)
    from mss.evaluation.hypothesis_tests import REGRESSION_EXPORT_COLUMNS, run_hypothesis_tests as _build

    hac = ecfg.hypothesis_hac_max_lags
    rows, mz_r, inc_r = _build(panel, ecfg, hac_max_lags=hac)
    out_ht.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_ht, index=False)
    df_mz = pd.DataFrame(mz_r) if mz_r else pd.DataFrame(columns=REGRESSION_EXPORT_COLUMNS)
    df_inc = pd.DataFrame(inc_r) if inc_r else pd.DataFrame(columns=REGRESSION_EXPORT_COLUMNS)
    df_mz.to_csv(out_mz, index=False)
    df_inc.to_csv(out_inc, index=False)
    chk = check_hypothesis_tests(out_ht)
    if not chk.get("passed"):
        raise RuntimeError(f"hypothesis_tests contract failed: {chk.get('issues')}")


def run_model_evaluate(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    """Run all evaluate steps in order (minimal pipeline)."""
    run_score_splits(cfg, overwrite=overwrite)
    run_write_forecast_panel(cfg, overwrite=overwrite)
    run_aggregate_test_loss(cfg, overwrite=overwrite)
    run_write_summary_table(cfg, overwrite=overwrite)
    run_hypothesis_tests_step(cfg, overwrite=overwrite)
