"""Node features: rolling mean and volatility over the estimation window."""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.progress_util import have_tqdm

try:
    from tqdm import tqdm as _tqdm_bar
except ImportError:  # pragma: no cover
    _tqdm_bar = None


def _output_columns(
    *,
    rolling_mean: bool,
    rolling_vol: bool,
    include_log_dollar_volume: bool,
    include_turnover: bool,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
) -> list[str]:
    cols: list[str] = []
    if rolling_mean:
        cols.append("rolling_mean")
    if rolling_vol:
        cols.append("rolling_vol")
    if include_log_dollar_volume:
        cols.append("log_dollar_volume")
    if include_turnover:
        cols.append("turnover")
    if include_downside_semivol:
        cols.append("downside_semivol")
    if include_skew:
        cols.append("skew")
    return cols


def compute_node_features_for_date(
    returns_panel: pd.DataFrame,
    permnos: list[int],
    feature_date: pd.Timestamp | str,
    *,
    window_length: int,
    ret_col: str = "ret_used",
    rolling_mean: bool = True,
    rolling_vol: bool = True,
    include_log_dollar_volume: bool = False,
    include_turnover: bool = False,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    out_cols = _output_columns(
        rolling_mean=rolling_mean,
        rolling_vol=rolling_vol,
        include_log_dollar_volume=include_log_dollar_volume,
        include_turnover=include_turnover,
        include_downside_semivol=include_downside_semivol,
        include_skew=include_skew,
    )
    base_cols = ["date", "permno"] + out_cols
    if not permnos:
        return pd.DataFrame(columns=base_cols)
    fd = pd.to_datetime(feature_date).normalize()
    if calendar is None:
        uniq = returns_panel["date"].drop_duplicates().sort_values()
        calendar = TradingCalendar.from_sorted_dates(uniq)
    use_dates = set(calendar.window_dates_inclusive(fd, window_length))
    if not use_dates:
        empty = pd.DataFrame(columns=base_cols)
        return empty
    window = returns_panel.copy()
    window["date"] = pd.to_datetime(window["date"]).dt.normalize()
    w = window[window["date"].isin(use_dates) & window["permno"].isin(permnos)]
    if rolling_mean or rolling_vol or include_downside_semivol or include_skew:
        w = w.dropna(subset=[ret_col])

    out = pd.DataFrame({"permno": permnos, "date": fd})
    if rolling_mean:
        out["rolling_mean"] = w.groupby("permno")[ret_col].mean().reindex(permnos).fillna(0.0).values
    if rolling_vol:
        out["rolling_vol"] = w.groupby("permno")[ret_col].std().reindex(permnos).fillna(0.0).values
    if include_downside_semivol:
        neg_sq = w.assign(_neg_sq=np.where(w[ret_col] < 0, w[ret_col] ** 2, 0.0))
        semivar = neg_sq.groupby("permno")["_neg_sq"].mean().reindex(permnos).fillna(0.0)
        out["downside_semivol"] = np.sqrt(semivar.to_numpy(dtype=float))
    if include_skew:
        out["skew"] = (
            w.groupby("permno")[ret_col].skew().reindex(permnos).fillna(0.0).to_numpy(dtype=float)
        )
    if include_log_dollar_volume and {"prc", "vol"}.issubset(w.columns):
        log_dv = np.log1p(w["prc"].abs() * w["vol"])
        out["log_dollar_volume"] = (
            w.assign(_log_dv=log_dv).groupby("permno")["_log_dv"].mean().reindex(permnos).fillna(0.0).values
        )
    if include_turnover and {"vol", "shrout"}.issubset(w.columns):
        turnover = w["vol"] / w["shrout"].replace(0, pd.NA)
        out["turnover"] = (
            w.assign(_turnover=turnover).groupby("permno")["_turnover"].mean().reindex(permnos).fillna(0.0).values
        )
    for c in out_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[base_cols]


def _build_node_features_vectorized(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    rolling_mean: bool,
    rolling_vol: bool,
    include_log_dollar_volume: bool,
    include_turnover: bool,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
    min_periods: int = 1,
    verbose: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    out_cols = _output_columns(
        rolling_mean=rolling_mean,
        rolling_vol=rolling_vol,
        include_log_dollar_volume=include_log_dollar_volume,
        include_turnover=include_turnover,
        include_downside_semivol=include_downside_semivol,
        include_skew=include_skew,
    )
    base_cols = ["date", "permno"] + out_cols
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
        read_cols = ["date", "permno", ret_col]
        if include_log_dollar_volume:
            read_cols.extend(["prc", "vol"])
        if include_turnover:
            read_cols.extend(["vol", "shrout"])
        read_cols = list(dict.fromkeys(read_cols))
        ret = pd.read_parquet(returns_panel_path, columns=read_cols)
        ret["date"] = pd.to_datetime(ret["date"]).dt.normalize()
        if verbose:
            n_rows = len(ret)
            n_permnos = ret["permno"].nunique()
            print(f"[node-features] Loaded returns panel: {n_rows:,} rows, {n_permnos:,} permnos.")
        _tick("load")
        all_permnos = universe_per_date["permno"].unique()
        ret = ret[ret["permno"].isin(all_permnos)]
        if verbose:
            n_rows_filt = len(ret)
            n_permnos_filt = ret["permno"].nunique()
            print(
                f"[node-features] Filtered to universe permnos: {n_rows_filt:,} rows, {n_permnos_filt:,} permnos."
            )
        _tick("filter")

        if not out_cols:
            if pbar is not None:
                pbar.update(2)
            elif show_progress:
                print("[node-features] skip rolling/merge (no features enabled)", file=sys.stderr)
            out = universe_per_date[["date", "permno"]].copy()
            for c in out_cols:
                out[c] = 0.0
            return out

        cal = TradingCalendar.from_parquet_date_column(returns_panel_path)
        spine = pd.DataFrame({"date": list(cal.dates), "pos": range(1, len(cal) + 1)})
        ret = ret.merge(spine, on="date", how="inner")
        feat_dates = pd.to_datetime(universe_per_date["date"].drop_duplicates())
        anchor = feat_dates.to_frame("date").merge(spine, on="date")
        anchor["anchor_pos"] = anchor["pos"]
        anchor["start_pos"] = anchor["anchor_pos"] - window_length + 1

        if include_log_dollar_volume:
            ret["_log_dv"] = np.log1p(ret["prc"].abs() * ret["vol"])
        if include_turnover:
            ret["_turnover"] = ret["vol"] / ret["shrout"].replace(0, pd.NA)

        select_parts = ["a.date", "r.permno", f"COUNT(r.{ret_col}) AS n_obs"]
        if rolling_mean:
            select_parts.append(f"AVG(r.{ret_col}) AS rolling_mean")
        if rolling_vol:
            select_parts.append(f"STDDEV_SAMP(r.{ret_col}) AS rolling_vol")
        if include_log_dollar_volume:
            select_parts.append("AVG(r._log_dv) AS log_dollar_volume")
        if include_turnover:
            select_parts.append("AVG(r._turnover) AS turnover")
        if include_downside_semivol:
            select_parts.append(
                f"SQRT(AVG(CASE WHEN r.{ret_col} < 0 THEN r.{ret_col} * r.{ret_col} ELSE 0 END)) "
                "AS downside_semivol"
            )
        if include_skew:
            select_parts.append(f"COALESCE(skewness(r.{ret_col}), 0.0) AS skew")

        if verbose:
            print(
                f"[node-features] Computing calendar-window stats (window={window_length}, min_periods={min_periods})..."
            )
        con = duckdb.connect(database=":memory:")
        con.register("ret", ret)
        con.register("anchor", anchor)
        q = f"""
            SELECT {", ".join(select_parts)}
            FROM anchor a
            JOIN ret r ON r.pos BETWEEN a.start_pos AND a.anchor_pos
            GROUP BY a.date, r.permno
            HAVING COUNT(r.{ret_col}) >= {int(min_periods)}
        """
        rolling = con.execute(q).df()
        con.close()
        rolling["date"] = pd.to_datetime(rolling["date"])
        _tick("rolling")
        if verbose:
            print("[node-features] Joining rolling stats to universe and finalizing table...")
        rolling = rolling.merge(
            universe_per_date[["date", "permno"]], on=["date", "permno"], how="inner"
        )
        for c in out_cols:
            if c not in rolling.columns:
                rolling[c] = 0.0
            else:
                rolling[c] = rolling[c].fillna(0.0)
        out = rolling[base_cols].sort_values(["date", "permno"]).reset_index(drop=True)
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
    include_log_dollar_volume: bool,
    include_turnover: bool,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
    calendar_dates: tuple[pd.Timestamp, ...] = (),
    progress_queue: Any | None = None,
    progress_every: int = 200,
    worker_index: int | None = None,
    n_workers: int | None = None,
    debug_progress: bool = False,
) -> pd.DataFrame:
    out_cols = _output_columns(
        rolling_mean=rolling_mean,
        rolling_vol=rolling_vol,
        include_log_dollar_volume=include_log_dollar_volume,
        include_turnover=include_turnover,
        include_downside_semivol=include_downside_semivol,
        include_skew=include_skew,
    )
    read_cols = ["date", "permno", ret_col]
    if include_log_dollar_volume:
        read_cols.extend(["prc", "vol"])
    if include_turnover:
        read_cols.extend(["vol", "shrout"])
    read_cols = list(dict.fromkeys(read_cols))
    ret = pd.read_parquet(returns_panel_path, columns=read_cols)
    ret["date"] = pd.to_datetime(ret["date"])
    calendar = TradingCalendar(calendar_dates)
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
            include_log_dollar_volume=include_log_dollar_volume,
            include_turnover=include_turnover,
            include_downside_semivol=include_downside_semivol,
            include_skew=include_skew,
            calendar=calendar,
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
        return pd.DataFrame(columns=["date", "permno"] + out_cols)
    return pd.concat(rows, ignore_index=True)


def _build_node_features_parallel(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    rolling_mean: bool,
    rolling_vol: bool,
    include_log_dollar_volume: bool,
    include_turnover: bool,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
    progress_every: int = 50,
    n_jobs: int = 1,
    verbose: bool = True,
    debug_progress: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    out_cols = _output_columns(
        rolling_mean=rolling_mean,
        rolling_vol=rolling_vol,
        include_log_dollar_volume=include_log_dollar_volume,
        include_turnover=include_turnover,
        include_downside_semivol=include_downside_semivol,
        include_skew=include_skew,
    )
    dates = universe_per_date["date"].drop_duplicates().sort_values().tolist()
    n_dates = len(dates)
    chunk_size = max(1, (n_dates + n_jobs - 1) // n_jobs)
    chunks = [dates[i : i + chunk_size] for i in range(0, n_dates, chunk_size)]
    n_workers = len(chunks)
    chunk_lens = [len(c) for c in chunks]
    calendar = TradingCalendar.from_parquet_date_column(returns_panel_path)
    calendar_dates = calendar.dates
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
                        include_log_dollar_volume=include_log_dollar_volume,
                        include_turnover=include_turnover,
                        include_downside_semivol=include_downside_semivol,
                        include_skew=include_skew,
                        calendar_dates=calendar_dates,
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
        return pd.DataFrame(columns=["date", "permno"] + out_cols)
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
    include_log_dollar_volume: bool = False,
    include_turnover: bool = False,
    include_downside_semivol: bool = False,
    include_skew: bool = False,
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
            include_log_dollar_volume=include_log_dollar_volume,
            include_turnover=include_turnover,
            include_downside_semivol=include_downside_semivol,
            include_skew=include_skew,
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
        include_log_dollar_volume=include_log_dollar_volume,
        include_turnover=include_turnover,
        include_downside_semivol=include_downside_semivol,
        include_skew=include_skew,
        progress_every=progress_every,
        n_jobs=n_jobs,
        verbose=verbose,
        debug_progress=debug_progress,
        show_progress=show_progress,
    )
