"""Node features: rolling mean and volatility over the estimation window."""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd

from mss.graph.progress_util import have_tqdm

try:
    from tqdm import tqdm as _tqdm_bar
except ImportError:  # pragma: no cover
    _tqdm_bar = None


def compute_node_features_for_date(
    returns_panel: pd.DataFrame,
    permnos: list[int],
    feature_date: pd.Timestamp | str,
    *,
    window_length: int,
    ret_col: str = "ret_used",
    rolling_mean: bool = True,
    rolling_vol: bool = True,
) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame(columns=["date", "permno", "rolling_mean", "rolling_vol"])
    fd = pd.to_datetime(feature_date)
    start = fd - pd.Timedelta(days=window_length * 2)
    window = returns_panel[
        (returns_panel["date"] <= fd) & (returns_panel["date"] >= start)
    ].copy()
    window["date"] = pd.to_datetime(window["date"])
    window = window.sort_values("date")
    uniq_dates = window["date"].drop_duplicates().sort_values(ascending=False)
    if len(uniq_dates) < window_length:
        use_dates = set(uniq_dates)
    else:
        use_dates = set(uniq_dates.iloc[:window_length])
    w = window[window["date"].isin(use_dates) & window["permno"].isin(permnos)]
    w = w.dropna(subset=[ret_col])

    agg = w.groupby("permno")[ret_col].agg(["mean", "std"])
    agg = agg.rename(columns={"mean": "rolling_mean", "std": "rolling_vol"})
    agg = agg.reindex(permnos)
    agg["permno"] = agg.index
    agg["date"] = fd
    agg = agg.reset_index(drop=True)[["date", "permno", "rolling_mean", "rolling_vol"]]
    agg["rolling_vol"] = agg["rolling_vol"].fillna(0.0)
    agg["rolling_mean"] = agg["rolling_mean"].fillna(0.0)
    return agg


def _build_node_features_vectorized(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    rolling_mean: bool,
    rolling_vol: bool,
    min_periods: int,
    verbose: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    use_bar = bool(show_progress and have_tqdm() and _tqdm_bar is not None)
    pbar = (
        _tqdm_bar(
            total=4,
            desc="node_features",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.25,
            unit="stage",
        )
        if use_bar
        else None
    )

    def _tick(postfix: str) -> None:
        if pbar is not None:
            pbar.set_postfix_str(postfix, refresh=False)
            pbar.update(1)
        elif show_progress:
            print(f"[node-features] {postfix}", file=sys.stderr)

    try:
        if verbose:
            print("[node-features] Vectorized build: loading returns panel...")
        ret = pd.read_parquet(returns_panel_path, columns=["date", "permno", ret_col])
        ret["date"] = pd.to_datetime(ret["date"])
        if verbose:
            n_rows = len(ret)
            n_permnos = ret["permno"].nunique()
            print(f"[node-features] Loaded returns panel: {n_rows:,} rows, {n_permnos:,} permnos.")
        _tick("load")
        all_permnos = universe_per_date["permno"].unique()
        ret = ret[ret["permno"].isin(all_permnos)].sort_values(["permno", "date"])
        if verbose:
            n_rows_filt = len(ret)
            n_permnos_filt = ret["permno"].nunique()
            print(
                f"[node-features] Filtered to universe permnos: {n_rows_filt:,} rows, {n_permnos_filt:,} permnos."
            )
        _tick("filter")

        agg_cols = []
        if rolling_mean:
            agg_cols.append("mean")
        if rolling_vol:
            agg_cols.append("std")
        if not agg_cols:
            if pbar is not None:
                pbar.update(2)
            elif show_progress:
                print("[node-features] skip rolling/merge (no mean/vol features)", file=sys.stderr)
            return universe_per_date[["date", "permno"]].assign(rolling_mean=0.0, rolling_vol=0.0)

        if verbose:
            print(
                f"[node-features] Computing rolling stats (window={window_length}, min_periods={min_periods})..."
            )
        rolling = (
            ret.groupby("permno", group_keys=False)[ret_col]
            .rolling(window=window_length, min_periods=min_periods)
            .agg(agg_cols)
        )
        if rolling_mean and rolling_vol:
            rolling.columns = ["rolling_mean", "rolling_vol"]
        elif rolling_mean:
            rolling = rolling.rename(columns={"mean": "rolling_mean"}).assign(rolling_vol=0.0)
        else:
            rolling = rolling.rename(columns={"std": "rolling_vol"}).assign(rolling_mean=0.0)
        rolling = rolling.reset_index(drop=True)
        rolling["permno"] = ret["permno"].values
        rolling["date"] = ret["date"].values
        _tick("rolling")
        if verbose:
            print("[node-features] Joining rolling stats to universe and finalizing table...")
        rolling = rolling.merge(
            universe_per_date[["date", "permno"]], on=["date", "permno"], how="inner"
        )
        rolling["rolling_mean"] = rolling["rolling_mean"].fillna(0.0)
        rolling["rolling_vol"] = rolling["rolling_vol"].fillna(0.0)
        out = rolling[["date", "permno", "rolling_mean", "rolling_vol"]].sort_values(
            ["date", "permno"]
        ).reset_index(drop=True)
        _tick("merge")
        if verbose:
            n_out = len(out)
            n_dates = out["date"].nunique()
            print(f"[node-features] Vectorized build complete: {n_out:,} rows over {n_dates:,} dates.")
        return out
    finally:
        if pbar is not None:
            pbar.close()


def _build_node_features_chunk(
    returns_panel_path: str,
    date_chunk: list[pd.Timestamp],
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    rolling_mean: bool,
    rolling_vol: bool,
    progress_queue: Any | None = None,
    progress_every: int = 200,
    worker_index: int | None = None,
    n_workers: int | None = None,
    debug_progress: bool = False,
) -> pd.DataFrame:
    ret = pd.read_parquet(returns_panel_path, columns=["date", "permno", ret_col])
    ret["date"] = pd.to_datetime(ret["date"])
    rows = []
    total_dates = len(date_chunk)
    for i, d in enumerate(date_chunk, start=1):
        permnos = universe_per_date[universe_per_date["date"] == d]["permno"].tolist()
        nf = compute_node_features_for_date(
            ret,
            permnos,
            d,
            window_length=window_length,
            ret_col=ret_col,
            rolling_mean=rolling_mean,
            rolling_vol=rolling_vol,
        )
        rows.append(nf)
        if i == 1 or i == total_dates or (progress_every > 0 and i % progress_every == 0):
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "worker_index": worker_index,
                        "n_workers": n_workers,
                        "i": i,
                        "total": total_dates,
                        "date": pd.to_datetime(d).date().isoformat(),
                    }
                )
            elif debug_progress:
                prefix = "[node-features][worker]"
                if worker_index is not None and n_workers is not None:
                    prefix = f"[node-features][worker {worker_index + 1}/{n_workers}]"
                print(f"{prefix} {i}/{total_dates} date={d}")
    if not rows:
        return pd.DataFrame(columns=["date", "permno", "rolling_mean", "rolling_vol"])
    return pd.concat(rows, ignore_index=True)


def _build_node_features_parallel(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    rolling_mean: bool,
    rolling_vol: bool,
    progress_every: int,
    n_jobs: int,
    verbose: bool = True,
    debug_progress: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    dates = universe_per_date["date"].drop_duplicates().sort_values().tolist()
    n_dates = len(dates)
    chunk_size = max(1, (n_dates + n_jobs - 1) // n_jobs)
    chunks = [dates[i : i + chunk_size] for i in range(0, n_dates, chunk_size)]
    n_workers = len(chunks)
    chunk_lens = [len(c) for c in chunks]
    if verbose:
        print(
            f"[node-features] Parallel build over {n_dates:,} dates using {n_jobs} jobs ({n_workers} workers)."
        )
    rp_str = str(returns_panel_path.resolve())

    use_queue = bool(show_progress or debug_progress)
    want_tqdm = bool(show_progress and have_tqdm() and _tqdm_bar is not None)
    pbar: Any = None
    lock = threading.Lock()
    last_seen = [0] * n_workers
    stop_event = threading.Event()
    chunk_results: list[pd.DataFrame | None] = [None] * len(chunks)

    with mp.Manager() as manager:
        q = manager.Queue() if use_queue else None
        if want_tqdm:
            pbar = _tqdm_bar(
                total=n_dates,
                desc="node_features",
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.25,
                unit="date",
            )

        def _drain_queue() -> None:
            if q is None:
                return
            while True:
                try:
                    msg = q.get(timeout=0.2)
                except Empty:
                    if stop_event.is_set():
                        break
                    continue
                except (OSError, EOFError, ValueError):
                    if stop_event.is_set():
                        break
                    continue
                wi = int(msg["worker_index"])
                i = int(msg["i"])
                delta = 0
                with lock:
                    if 0 <= wi < n_workers:
                        delta = i - last_seen[wi]
                        if delta > 0:
                            last_seen[wi] = i
                if delta <= 0:
                    continue
                if pbar is not None:
                    pbar.update(delta)
                elif show_progress and not want_tqdm:
                    with lock:
                        done = sum(last_seen)
                    print(f"[node-features] dates {done}/{n_dates}", file=sys.stderr)

        drain_thread: threading.Thread | None = None
        if q is not None:
            drain_thread = threading.Thread(target=_drain_queue, name="node_features_progress", daemon=True)
            drain_thread.start()
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(
                        _build_node_features_chunk,
                        rp_str,
                        chunk,
                        universe_per_date,
                        window_length=window_length,
                        ret_col=ret_col,
                        rolling_mean=rolling_mean,
                        rolling_vol=rolling_vol,
                        progress_queue=q,
                        progress_every=progress_every,
                        worker_index=i,
                        n_workers=n_workers,
                        debug_progress=debug_progress,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                for f in as_completed(futures):
                    idx = futures[f]
                    chunk_results[idx] = f.result()
                    rem = 0
                    with lock:
                        cl = chunk_lens[idx]
                        if last_seen[idx] < cl:
                            rem = cl - last_seen[idx]
                            last_seen[idx] = cl
                    if rem > 0 and pbar is not None:
                        pbar.update(rem)
                    elif rem > 0 and show_progress and not want_tqdm:
                        with lock:
                            done = sum(last_seen)
                        print(f"[node-features] dates {done}/{n_dates}", file=sys.stderr)
        finally:
            stop_event.set()
            if drain_thread is not None:
                drain_thread.join(timeout=5.0)
            if pbar is not None:
                pbar.close()

    rows = [df for df in chunk_results if df is not None and not df.empty]
    if not rows:
        return pd.DataFrame(columns=["date", "permno", "rolling_mean", "rolling_vol"])
    out = pd.concat(rows, ignore_index=True).sort_values(["date", "permno"]).reset_index(drop=True)
    if verbose:
        print(
            f"[node-features] Parallel build complete: {len(out):,} rows over {out['date'].nunique():,} dates."
        )
    return out


def build_node_features(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str = "ret_used",
    rolling_mean: bool = True,
    rolling_vol: bool = True,
    progress_every: int = 50,
    limit_dates: int | None = None,
    n_jobs: int = 1,
    min_obs: int | None = None,
    verbose: bool = True,
    debug_progress: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    if not returns_panel_path.is_file():
        raise FileNotFoundError(f"Returns panel not found: {returns_panel_path}")

    if limit_dates is not None:
        dates = universe_per_date["date"].drop_duplicates().sort_values().tolist()[:limit_dates]
        universe_per_date = universe_per_date[universe_per_date["date"].isin(dates)]

    min_periods = min_obs if min_obs is not None else max(1, int(window_length * 0.8))

    if n_jobs == 1:
        return _build_node_features_vectorized(
            returns_panel_path=returns_panel_path,
            universe_per_date=universe_per_date,
            window_length=window_length,
            ret_col=ret_col,
            rolling_mean=rolling_mean,
            rolling_vol=rolling_vol,
            min_periods=min_periods,
            verbose=verbose,
            show_progress=show_progress,
        )

    return _build_node_features_parallel(
        returns_panel_path=returns_panel_path,
        universe_per_date=universe_per_date,
        window_length=window_length,
        ret_col=ret_col,
        rolling_mean=rolling_mean,
        rolling_vol=rolling_vol,
        progress_every=progress_every,
        n_jobs=n_jobs,
        verbose=verbose,
        debug_progress=debug_progress,
        show_progress=show_progress,
    )
