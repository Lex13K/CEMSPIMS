"""Sparse edges from return correlations (top-k per node)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mss.calendar.trading_windows import TradingCalendar
from mss.graph.progress_util import have_tqdm, maybe_tqdm


def _read_returns_for_chunk(
    returns_panel_path: str,
    date_chunk: list,
    *,
    ret_col: str,
    window_length: int,
    calendar: TradingCalendar,
) -> pd.DataFrame:
    """Load returns for chunk dates plus each date's rolling window (not the full panel)."""
    needed: set[pd.Timestamp] = set()
    for d in date_chunk:
        fd = pd.to_datetime(d).normalize()
        needed.update(calendar.window_dates_inclusive(fd, window_length))
    if not needed:
        return pd.DataFrame(columns=["date", "permno", ret_col])
    date_literals = ", ".join(
        f"DATE '{pd.Timestamp(x).date().isoformat()}'" for x in sorted(needed)
    )
    pq = str(Path(returns_panel_path).resolve()).replace("\\", "/").replace("'", "''")
    con = duckdb.connect(database=":memory:")
    df = con.execute(
        f"""
        SELECT date, permno, {ret_col}
        FROM read_parquet('{pq}')
        WHERE date IN ({date_literals})
        """
    ).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_edges_for_date(
    returns_panel: pd.DataFrame,
    permnos: list[int],
    feature_date: pd.Timestamp | str,
    *,
    window_length: int,
    ret_col: str = "ret_used",
    dependence: str = "spearman",
    top_k: int = 10,
    symmetrize: bool = True,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    if len(permnos) < 2:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    fd = pd.to_datetime(feature_date).normalize()
    if calendar is None:
        uniq = returns_panel["date"].drop_duplicates().sort_values()
        calendar = TradingCalendar.from_sorted_dates(uniq)
    use_dates = set(calendar.window_dates_inclusive(fd, window_length))
    if not use_dates:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    window = returns_panel.copy()
    window["date"] = pd.to_datetime(window["date"]).dt.normalize()
    w = window[window["date"].isin(use_dates) & window["permno"].isin(permnos)]
    w = w.dropna(subset=[ret_col])
    pivot = w.pivot_table(index="date", columns="permno", values=ret_col)
    pivot = pivot.reindex(columns=permnos).dropna(axis=1, how="all")
    pivot.columns = pivot.columns.astype(int)
    permnos_avail = pivot.columns.tolist()
    if len(permnos_avail) < 2:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    if dependence.lower() == "spearman":
        corr = pivot.rank().corr(method="pearson")
    else:
        corr = pivot.corr()
    corr.index = corr.index.astype(int)
    corr.columns = corr.columns.astype(int)
    corr_np = corr.values.astype(np.float64)
    n = len(permnos_avail)
    np.fill_diagonal(corr_np, np.nan)
    abs_corr = np.abs(corr_np)
    np.fill_diagonal(abs_corr, -np.inf)
    k_use = min(top_k, n - 1)
    if k_use < 1:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    top_inds = np.argpartition(-abs_corr, k_use - 1, axis=1)[:, :k_use]
    permnos_arr = np.array(permnos_avail, dtype=np.int64)
    src_arr = np.repeat(permnos_arr, k_use)
    dst_arr = permnos_arr[top_inds.ravel()]
    weights = corr_np[np.repeat(np.arange(n), k_use), top_inds.ravel()]
    valid = ~np.isnan(weights)
    if not np.any(valid):
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    edf = pd.DataFrame(
        {
            "src": src_arr[valid].astype(int),
            "dst": dst_arr[valid].astype(int),
            "weight": weights[valid].astype(np.float64),
        }
    )
    if symmetrize:
        rev = edf.rename(columns={"src": "dst", "dst": "src"})
        edf = pd.concat([edf, rev], ignore_index=True).drop_duplicates(subset=["src", "dst"])
    edf["date"] = fd
    return edf[["date", "src", "dst", "weight"]]


def _build_edges_chunk(
    returns_panel_path: str,
    date_chunk: list,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str,
    dependence: str,
    top_k: int,
    symmetrize: bool,
    calendar_dates: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    calendar = TradingCalendar(calendar_dates)
    ret = _read_returns_for_chunk(
        returns_panel_path,
        date_chunk,
        ret_col=ret_col,
        window_length=window_length,
        calendar=calendar,
    )
    chunk_dates = {pd.to_datetime(d).normalize() for d in date_chunk}
    universe_slice = universe_per_date[
        pd.to_datetime(universe_per_date["date"]).dt.normalize().isin(chunk_dates)
    ]
    rows = []
    for d in date_chunk:
        permnos = universe_slice[pd.to_datetime(universe_slice["date"]).dt.normalize() == pd.to_datetime(d).normalize()][
            "permno"
        ].tolist()
        edf = compute_edges_for_date(
            ret,
            permnos,
            d,
            window_length=window_length,
            ret_col=ret_col,
            dependence=dependence,
            top_k=top_k,
            symmetrize=symmetrize,
            calendar=calendar,
        )
        rows.append(edf)
    if not rows:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    return pd.concat(rows, ignore_index=True)


def build_edge_lists(
    returns_panel_path: Path,
    universe_per_date: pd.DataFrame,
    *,
    window_length: int,
    ret_col: str = "ret_used",
    dependence: str = "spearman",
    top_k: int = 10,
    symmetrize: bool = True,
    update_every: int = 100,
    limit_dates: int | None = None,
    out_path: Path | None = None,
    overwrite: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if not returns_panel_path.is_file():
        raise FileNotFoundError(f"Returns panel not found: {returns_panel_path}")

    all_dates = [
        pd.to_datetime(d) for d in universe_per_date["date"].drop_duplicates().sort_values().tolist()
    ]
    if limit_dates is not None:
        all_dates = all_dates[:limit_dates]

    if update_every <= 0:
        raise ValueError("update_every must be >= 1.")

    existing_edges: pd.DataFrame | None = None
    if out_path is not None and out_path.exists() and not overwrite:
        existing_edges = pd.read_parquet(out_path)
        existing_edges["date"] = pd.to_datetime(existing_edges["date"])
        done_dates = {pd.to_datetime(d) for d in existing_edges["date"].unique()}
        dates = [d for d in all_dates if d not in done_dates]
        if not dates:
            if verbose:
                print("  All dates already in output; nothing to do.")
            return existing_edges
        if verbose:
            print(f"  Resuming: {len(done_dates)} dates already in output, {len(dates)} remaining.")
    else:
        dates = all_dates

    def _checkpoint_merge(
        existing: pd.DataFrame | None, pending_rows: list[pd.DataFrame]
    ) -> pd.DataFrame:
        if existing is not None and pending_rows:
            return pd.concat([existing] + pending_rows, ignore_index=True)
        if existing is not None:
            return existing
        if pending_rows:
            return pd.concat(pending_rows, ignore_index=True)
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])

    def _write_checkpoint(df: pd.DataFrame) -> None:
        if out_path is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df = df.sort_values("date")
        df.to_parquet(out_path, index=False)

    n_dates = len(dates)
    calendar = TradingCalendar.from_parquet_date_column(returns_panel_path)
    calendar_dates = calendar.dates

    if n_jobs > 1:
        chunk_size = update_every
        chunks = [dates[i : i + chunk_size] for i in range(0, n_dates, chunk_size)]
        n_chunks = len(chunks)
        rp_str = str(returns_panel_path.resolve())
        if n_chunks == 0:
            return (
                existing_edges
                if existing_edges is not None
                else pd.DataFrame(columns=["date", "src", "dst", "weight"])
            )
        if verbose:
            print(
                f"  Parallel edges: {n_dates} dates, {n_chunks} chunks, "
                f"n_jobs={n_jobs}, update_every={update_every}"
            )
        completed_dates = 0
        since_last_checkpoint = 0
        pending_rows: list[pd.DataFrame] = []
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(
                        _build_edges_chunk,
                        rp_str,
                        chunk,
                        universe_per_date[
                            pd.to_datetime(universe_per_date["date"]).isin(
                                [pd.to_datetime(d) for d in chunk]
                            )
                        ],
                        window_length=window_length,
                        ret_col=ret_col,
                        dependence=dependence,
                        top_k=top_k,
                        symmetrize=symmetrize,
                        calendar_dates=calendar_dates,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                done_iter = as_completed(futures)
                done_iter = maybe_tqdm(
                    done_iter,
                    desc="edges",
                    total=n_chunks,
                    show=show_progress,
                )
                for f in done_iter:
                    idx = futures[f]
                    chunk_df = f.result()
                    chunk_len = len(chunks[idx])
                    completed_dates += chunk_len
                    since_last_checkpoint += chunk_len
                    if chunk_df is not None and not chunk_df.empty:
                        pending_rows.append(chunk_df)
                    latest_date = pd.to_datetime(chunks[idx][-1]) if chunks[idx] else pd.NaT
                    # tqdm when show_progress + available; else periodic prints (incl. no-tqdm fallback)
                    if verbose and (not show_progress or not have_tqdm()):
                        print(
                            f"  [{completed_dates:5d}/{n_dates}] edges completed "
                            f"(chunk {idx + 1}/{n_chunks}, latest date {latest_date})"
                        )
                    if since_last_checkpoint >= update_every:
                        combined = _checkpoint_merge(existing_edges, pending_rows)
                        _write_checkpoint(combined)
                        existing_edges = combined
                        pending_rows = []
                        since_last_checkpoint = since_last_checkpoint % update_every
        except KeyboardInterrupt:
            combined = _checkpoint_merge(existing_edges, pending_rows)
            if len(combined) > 0:
                _write_checkpoint(combined)
                if verbose:
                    print(
                        f"\n  Interrupted (Ctrl+C). Progress saved to {out_path} "
                        f"({combined['date'].nunique()} dates). Run again without --overwrite to continue."
                    )
            raise
        combined = _checkpoint_merge(existing_edges, pending_rows)
        _write_checkpoint(combined)
        return combined

    ret = pd.read_parquet(returns_panel_path, columns=["date", "permno", ret_col])
    ret["date"] = pd.to_datetime(ret["date"])
    rows: list[pd.DataFrame] = []
    since_last_checkpoint = 0
    try:
        idx_loop = range(len(dates))
        idx_loop = maybe_tqdm(idx_loop, desc="edges", total=len(dates), show=show_progress)
        for j in idx_loop:
            d = dates[j]
            i = j + 1
            since_last_checkpoint += 1
            if not show_progress and verbose and (i == 1 or i % update_every == 0 or i == n_dates):
                print(f"  [{i:5d}/{n_dates}] edges for {d}")
            permnos = universe_per_date[universe_per_date["date"] == d]["permno"].tolist()
            edf = compute_edges_for_date(
                ret,
                permnos,
                d,
                window_length=window_length,
                ret_col=ret_col,
                dependence=dependence,
                top_k=top_k,
                symmetrize=symmetrize,
                calendar=calendar,
            )
            rows.append(edf)
            if since_last_checkpoint >= update_every:
                combined = _checkpoint_merge(existing_edges, rows)
                _write_checkpoint(combined)
                existing_edges = combined
                rows = []
                since_last_checkpoint = 0
    except KeyboardInterrupt:
        combined = _checkpoint_merge(existing_edges, rows)
        if len(combined) > 0:
            _write_checkpoint(combined)
            if verbose:
                print(
                    f"\n  Interrupted (Ctrl+C). Progress saved to {out_path} "
                    f"({combined['date'].nunique()} dates). Run again without --overwrite to continue."
                )
        raise

    if not rows and existing_edges is not None:
        return existing_edges
    if not rows:
        return pd.DataFrame(columns=["date", "src", "dst", "weight"])
    combined = _checkpoint_merge(existing_edges, rows)
    _write_checkpoint(combined)
    return combined
