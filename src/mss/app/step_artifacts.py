"""Resolve key artifact paths/previews for a pipeline step (app node detail)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mss.io.config import ResolvedConfig


def _preview_csv(path: Path, *, max_rows: int = 8) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        df = pd.read_csv(path).head(max_rows)
        return {"type": "table", "path": str(path), "columns": list(df.columns), "rows": df.values.tolist()}
    except Exception:
        return {"type": "file", "path": str(path)}


def _preview_image(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return {"type": "file", "path": str(path)}
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else f"image/{path.suffix.lstrip('.')}"
        return {"type": "image", "path": str(path), "data_url": f"data:{mime};base64,{data}"}
    except OSError:
        return {"type": "file", "path": str(path)}


def _preview_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return {"type": "json", "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    except Exception:
        return None


def artifacts_for_step(cfg: ResolvedConfig, pipeline: str, step_id: str) -> list[dict[str, Any]]:
    """Return lightweight previews for known step outputs."""
    from mss.processed.paths import (
        diagnostics_smoothing_path,
        hypothesis_tests_path,
        summary_table_path,
    )

    proc = Path(cfg.processed_dir)
    interim = Path(cfg.run_interim_dir)
    shared = Path(cfg.shared_interim_dir)
    out: list[dict[str, Any]] = []

    if pipeline == "data.prepare":
        if step_id == "ingest":
            for name in ("ingest_manifest.json",):
                p = shared / name
                j = _preview_json(p)
                if j:
                    out.append(j)
        elif step_id == "returns_panel":
            p = shared / "returns_panel.parquet"
            if p.is_file():
                out.append({"type": "file", "path": str(p)})
        elif step_id == "targets":
            p = shared / "targets.parquet"
            if p.is_file():
                out.append({"type": "file", "path": str(p)})

    elif pipeline == "graph.prepare":
        gdir = interim / "graphs"
        if step_id == "feature_dates":
            p = interim / "stage04_dates.parquet"
            if p.is_file():
                out.append({"type": "file", "path": str(p)})
        elif step_id == "universe" and (gdir / "universe.parquet").is_file():
            out.append({"type": "file", "path": str(gdir / "universe.parquet")})
        elif step_id in ("node_features", "edges") and gdir.is_dir():
            out.append({"type": "file", "path": str(gdir)})

    elif pipeline == "dataset.package" and step_id == "manifest":
        p = interim / "dataset" / "dataset_manifest.json"
        j = _preview_json(p)
        if j:
            out.append(j)

    elif pipeline == "model.train" and step_id == "train":
        for name in ("final_metrics.json", "training_state.json"):
            j = _preview_json(interim / "model_train" / name)
            if j:
                out.append(j)

    elif pipeline == "model.evaluate":
        if step_id == "write_summary_table":
            t = _preview_csv(summary_table_path(cfg))
            if t:
                out.append(t)
        elif step_id == "run_hypothesis_tests":
            t = _preview_csv(hypothesis_tests_path(cfg))
            if t:
                out.append(t)
        elif step_id == "aggregate_test_loss":
            j = _preview_json(proc / "scoring" / "test_loss.json")
            if j:
                out.append(j)

    elif pipeline == "analysis.summarize" and step_id == "loss_figure":
        from mss.analysis.figures import expected_paths_for_summarize

        for p in expected_paths_for_summarize(cfg):
            if p.suffix.lower() == ".csv":
                t = _preview_csv(p, max_rows=5)
                if t:
                    out.append(t)
            elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = _preview_image(p)
                if img:
                    out.append(img)

    if step_id == "raw":
        raw = Path(cfg.raw_dir)
        if raw.is_dir():
            files = sorted(f.name for f in raw.glob("*.csv"))[:10]
            out.append({"type": "raw_list", "path": str(raw), "files": files})

    return out


def terminal_artifacts(cfg: ResolvedConfig) -> dict[str, Any]:
    """Bundle for terminal / results node."""
    from mss.analysis.figures import expected_paths_for_summarize
    from mss.processed.paths import hypothesis_tests_path, summary_table_path

    figures: list[dict[str, Any]] = []
    for p in expected_paths_for_summarize(cfg):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img = _preview_image(p)
            if img:
                figures.append(img)

    summary = _preview_csv(summary_table_path(cfg))
    hypo = _preview_csv(hypothesis_tests_path(cfg), max_rows=20)
    return {
        "summary_table": summary,
        "hypothesis_tests": hypo,
        "figures": figures,
        "processed_dir": str(cfg.processed_dir),
    }
