"""Training loop: cold start, resume, early stopping, final_metrics.json."""

from __future__ import annotations

import json
import random
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mss.io.config import ResolvedConfig
from mss.io.paths import project_root, resolve_under_root
from mss.model_train.checks import (
    CHECKPOINT_FILENAME,
    FINAL_METRICS_FILENAME,
    TRAINING_STATE_FILENAME,
    can_resume_training,
    check_final_metrics_complete,
    checkpoint_path,
    final_metrics_path,
    model_train_dir,
    model_train_step_semantically_complete,
    training_state_path,
)
from mss.model_train.config import (
    ModelTrainConfig,
    current_model_train_fingerprint,
    load_model_train_config,
)
from mss.model_train.device import get_device
from mss.model_train.manifest_io import load_dataset_manifest
from mss.model_train.log_calibration import (
    LOG_CALIBRATION_JSON,
    collect_predictions_log_space,
    fit_affine_log_calibration,
    write_log_calibration_json,
)
from mss.model_train.losses import build_training_loss
from mss.model_train.models import build_model
from mss.model_train.pyg_data import GraphDateDataset


def _require_train_stack() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "model.train requires PyTorch Geometric. Install with: pip install -e \".[train]\""
        ) from e


def clear_model_train_outputs(out_dir: Path) -> None:
    # Preserve the canonical graph cache so `--overwrite model.train`
    # doesn't destroy the `.pt` graphs produced by `model.cache_graphs`.
    cache_dir = out_dir / "graph_cache"
    if out_dir.is_dir():
        for p in out_dir.iterdir():
            if p.name == "graph_cache":
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _save_training_state_json(
    path: Path,
    *,
    fingerprint: str,
    mcfg: ModelTrainConfig,
    last_epoch: int,
    best_epoch: int,
    best_val_loss: float,
    patience_counter: int,
    metrics_history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_fingerprint": fingerprint,
        "seed": mcfg.seed,
        "max_epochs": mcfg.max_epochs,
        "early_stopping_patience": mcfg.early_stopping_patience,
        "lr": mcfg.lr,
        "batch_size": mcfg.batch_size,
        "loss": mcfg.loss,
        "loss_eps": mcfg.loss_eps,
        "smooth_l1_beta": mcfg.smooth_l1_beta,
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "metrics_history": metrics_history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint_pt(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    last_epoch: int,
    best_epoch: int,
    best_val_loss: float,
    patience_counter: int,
    best_model_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    show_progress: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    iterator = tqdm(loader, desc="Train", leave=False, disable=not show_progress, file=sys.stderr)
    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n if n else 0.0


def _maybe_write_log_calibration(
    *,
    mcfg: ModelTrainConfig,
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    out_dir: Path,
    best_state: dict[str, Any],
    verbose: bool,
) -> Path | None:
    if not mcfg.fit_log_calibration:
        return None
    model.load_state_dict(best_state)
    y_t, y_p = collect_predictions_log_space(model, val_loader, device)
    a, b, n = fit_affine_log_calibration(y_t, y_p)
    path = out_dir / LOG_CALIBRATION_JSON
    write_log_calibration_json(path, a=a, b=b, split="val", n_obs=n)
    if verbose:
        print(
            f"[model.train] Wrote val log calibration to {path} (a={a:.6g}, b={b:.6g}, n={n})",
            file=sys.stderr,
        )
    return path


def evaluate_epoch(
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            loss = loss_fn(out, target)
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
    return total_loss / n if n else 0.0


def run_model_train(cfg: ResolvedConfig, *, overwrite: bool) -> None:
    _require_train_stack()
    from torch_geometric.loader import DataLoader

    interim = Path(cfg.interim_dir)
    manifest_path = interim / "dataset" / "manifest.json"
    mcfg = load_model_train_config(cfg.source_config_path)
    fp = current_model_train_fingerprint(cfg.source_config_path)
    out_dir = model_train_dir(cfg)
    ck_path = out_dir / CHECKPOINT_FILENAME
    st_path = out_dir / TRAINING_STATE_FILENAME
    fm_path = out_dir / FINAL_METRICS_FILENAME

    if not overwrite and model_train_step_semantically_complete(cfg):
        return

    if overwrite:
        clear_model_train_outputs(out_dir)
        resume = False
    else:
        resume = can_resume_training(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        if ck_path.is_file() and not st_path.is_file():
            raise ValueError(
                "Found model_train/checkpoint.pt without training_state.json. "
                "Use --overwrite model.train to start clean."
            )

    device = get_device(mcfg.device)
    if mcfg.verbose:
        if device.type == "cuda":
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "GPU"
            print(f"Using device: {device} ({name})", file=sys.stderr)
        else:
            print("Using device: cpu", file=sys.stderr)

    seed = mcfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    manifest = load_dataset_manifest(manifest_path)
    # Canonical cache location populated by `model.cache_graphs`.
    cache_dir: Path | None = out_dir / "graph_cache"

    train_ds = GraphDateDataset(
        manifest,
        "train",
        cache_dir=cache_dir,
        use_parquet_pushdown=mcfg.parquet_date_pushdown,
    )
    val_ds = GraphDateDataset(
        manifest,
        "val",
        cache_dir=cache_dir,
        use_parquet_pushdown=mcfg.parquet_date_pushdown,
    )
    if len(train_ds) == 0:
        raise ValueError("Train split has no dates with valid labels. Check splits and targets.")
    if len(val_ds) == 0:
        raise ValueError("Validation split has no dates with valid labels. Check [dataset] boundaries.")

    g0 = train_ds[0]
    in_channels = int(g0.x.shape[1])

    pin_memory = mcfg.pin_memory and device.type == "cuda"
    persistent_workers = mcfg.dataloader_num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=mcfg.batch_size,
        shuffle=True,
        num_workers=mcfg.dataloader_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=mcfg.batch_size,
        shuffle=False,
        num_workers=mcfg.dataloader_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = build_model(
        mcfg.model_type,
        in_channels,
        mcfg.hidden_channels,
        mcfg.num_gnn_layers,
        mcfg.pooling,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg.lr)
    loss_fn = build_training_loss(mcfg)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    metrics_list: list[dict[str, Any]] = []
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if resume and ck_path.is_file() and st_path.is_file():
        bundle = torch.load(ck_path, map_location=device, weights_only=False)
        prev_state = json.loads(st_path.read_text(encoding="utf-8"))
        if prev_state.get("config_fingerprint") != fp:
            if mcfg.verbose:
                print(
                    "[model.train] Config fingerprint changed since last checkpoint; "
                    "using current TOML max_epochs/patience/lr; optimizer re-initialized if lr changed.",
                    file=sys.stderr,
                )
            old_lr = float(prev_state.get("lr", mcfg.lr))
            model.load_state_dict(bundle["model_state_dict"])
            if abs(old_lr - mcfg.lr) > 1e-15:
                optimizer = torch.optim.Adam(model.parameters(), lr=mcfg.lr)
            else:
                optimizer.load_state_dict(bundle["optimizer_state_dict"])
        else:
            model.load_state_dict(bundle["model_state_dict"])
            optimizer.load_state_dict(bundle["optimizer_state_dict"])

        start_epoch = int(bundle["last_epoch"]) + 1
        best_epoch = int(bundle.get("best_epoch", prev_state.get("best_epoch", 0)))
        best_val = float(bundle.get("best_val_loss", prev_state.get("best_val_loss", float("inf"))))
        patience_counter = int(bundle.get("patience_counter", prev_state.get("patience_counter", 0)))
        metrics_list = list(prev_state.get("metrics_history", []))
        bms = bundle.get("best_model_state_dict")
        if bms is not None:
            best_model_state = {k: v.cpu().clone() if hasattr(v, "cpu") else v for k, v in bms.items()}
        if mcfg.verbose:
            print(f"[model.train] Resuming from epoch {start_epoch}", file=sys.stderr)
    elif mcfg.warm_start_checkpoint:
        w = Path(mcfg.warm_start_checkpoint)
        wp = resolve_under_root(project_root(), str(w)) if not w.is_absolute() else w.resolve()
        if not wp.is_file():
            raise FileNotFoundError(f"warm_start_checkpoint not found: {wp}")
        bundle = torch.load(wp, map_location=device, weights_only=False)
        state = bundle.get("best_model_state_dict") or bundle["model_state_dict"]
        model.load_state_dict(state, strict=True)
        if mcfg.verbose:
            print(f"[model.train] Warm-started weights from {wp}", file=sys.stderr)

    max_epochs = mcfg.max_epochs
    patience = mcfg.early_stopping_patience

    mutable: dict[str, Any] = {
        "last_epoch": start_epoch - 1,
        "metrics": metrics_list,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "patience_counter": patience_counter,
        "best_model_state": best_model_state,
    }

    def _persist(epoch_done: int) -> None:
        _save_checkpoint_pt(
            ck_path,
            model=model,
            optimizer=optimizer,
            last_epoch=epoch_done,
            best_epoch=int(mutable["best_epoch"]),
            best_val_loss=float(mutable["best_val"]),
            patience_counter=int(mutable["patience_counter"]),
            best_model_state=mutable["best_model_state"],
        )
        _save_training_state_json(
            st_path,
            fingerprint=fp,
            mcfg=mcfg,
            last_epoch=epoch_done,
            best_epoch=int(mutable["best_epoch"]),
            best_val_loss=float(mutable["best_val"]),
            patience_counter=int(mutable["patience_counter"]),
            metrics_history=mutable["metrics"],
        )

    def _on_sigint(_sig: int, _frame: object) -> None:
        _persist(int(mutable["last_epoch"]))
        print(f"\n[model.train] Interrupted; state saved under {out_dir}", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    if start_epoch > max_epochs:
        if mcfg.verbose:
            print(
                f"[model.train] last_epoch {start_epoch - 1} >= max_epochs {max_epochs}; "
                "writing completed metrics without further epochs.",
                file=sys.stderr,
            )
        model.load_state_dict(mutable["best_model_state"])
        cal_path = _maybe_write_log_calibration(
            mcfg=mcfg,
            model=model,
            val_loader=val_loader,
            device=device,
            out_dir=out_dir,
            best_state=mutable["best_model_state"],
            verbose=mcfg.verbose,
        )
        final_payload: dict[str, Any] = {
            "status": "completed",
            "seed": mcfg.seed,
            "best_epoch": int(mutable["best_epoch"]),
            "best_val_loss": float(mutable["best_val"]),
            "loss": mcfg.loss,
            "loss_eps": mcfg.loss_eps,
            "smooth_l1_beta": mcfg.smooth_l1_beta,
            "loss_alpha": mcfg.loss_alpha,
            "qlike_pred_transform": mcfg.qlike_pred_transform,
            "manifest_path": str(manifest_path.resolve()),
            "checkpoint_path": str(ck_path.resolve()),
            "training_state_path": str(st_path.resolve()),
            "model_type": mcfg.model_type,
            "config_fingerprint": fp,
            "metrics_history": metrics_list,
            "note": "no new epochs; max_epochs reduced below resume point",
        }
        if cal_path is not None:
            final_payload["log_calibration_path"] = str(cal_path.resolve())
        fm_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        chk = check_final_metrics_complete(fm_path)
        if not chk["passed"]:
            raise RuntimeError(f"final_metrics validation failed: {chk['issues']}")
        return

    epoch_iter = range(start_epoch, max_epochs + 1)
    pbar = tqdm(
        epoch_iter,
        desc="Epoch",
        unit="epoch",
        disable=not mcfg.show_progress,
        file=sys.stderr,
    )
    for epoch in pbar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, show_progress=mcfg.show_progress
        )
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
        metrics_list.append(
            {"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)}
        )
        mutable["metrics"] = metrics_list
        mutable["last_epoch"] = epoch

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            mutable["patience_counter"] = 0
            mutable["best_epoch"] = best_epoch
            mutable["best_val"] = best_val
            mutable["best_model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            mutable["patience_counter"] = int(mutable["patience_counter"]) + 1

        _persist(epoch)

        if mcfg.show_progress:
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_epoch}",
                pat=f"{mutable['patience_counter']}/{patience}",
            )

        if int(mutable["patience_counter"]) >= patience:
            if mcfg.verbose:
                print(
                    f"[model.train] Early stopping at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val:.6f})",
                    file=sys.stderr,
                )
            break

    model.load_state_dict(mutable["best_model_state"])
    cal_path = _maybe_write_log_calibration(
        mcfg=mcfg,
        model=model,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        best_state=mutable["best_model_state"],
        verbose=mcfg.verbose,
    )
    _save_checkpoint_pt(
        ck_path,
        model=model,
        optimizer=optimizer,
        last_epoch=int(mutable["last_epoch"]),
        best_epoch=int(mutable["best_epoch"]),
        best_val_loss=float(mutable["best_val"]),
        patience_counter=int(mutable["patience_counter"]),
        best_model_state=mutable["best_model_state"],
    )
    _save_training_state_json(
        st_path,
        fingerprint=fp,
        mcfg=mcfg,
        last_epoch=int(mutable["last_epoch"]),
        best_epoch=int(mutable["best_epoch"]),
        best_val_loss=float(mutable["best_val"]),
        patience_counter=int(mutable["patience_counter"]),
        metrics_history=metrics_list,
    )

    final_payload = {
        "status": "completed",
        "seed": mcfg.seed,
        "best_epoch": int(mutable["best_epoch"]),
        "best_val_loss": float(mutable["best_val"]),
        "loss": mcfg.loss,
        "loss_eps": mcfg.loss_eps,
        "smooth_l1_beta": mcfg.smooth_l1_beta,
        "loss_alpha": mcfg.loss_alpha,
        "qlike_pred_transform": mcfg.qlike_pred_transform,
        "manifest_path": str(manifest_path.resolve()),
        "checkpoint_path": str(ck_path.resolve()),
        "training_state_path": str(st_path.resolve()),
        "model_type": mcfg.model_type,
        "config_fingerprint": fp,
        "metrics_history": metrics_list,
    }
    if cal_path is not None:
        final_payload["log_calibration_path"] = str(cal_path.resolve())
    fm_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    chk = check_final_metrics_complete(fm_path)
    if not chk["passed"]:
        raise RuntimeError(f"final_metrics validation failed: {chk['issues']}")
