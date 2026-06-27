"""Optional tqdm wrappers for graph steps (stderr; matches ingest pattern)."""

from __future__ import annotations

import sys
from typing import Any, Iterable, Iterator, TypeVar

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

T = TypeVar("T")


def have_tqdm() -> bool:
    """True if tqdm imported successfully (for fallback messaging when bars are unavailable)."""
    return _tqdm is not None


def maybe_tqdm(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None,
    show: bool,
    **kwargs: Any,
) -> Iterable[T]:
    """Wrap iterable with tqdm when show is True and tqdm is installed; else passthrough."""
    if not show or _tqdm is None:
        return iterable
    return _tqdm(
        iterable,
        desc=desc,
        total=total,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.25,
        **kwargs,
    )


def iter_tqdm(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None,
    show: bool,
    **kwargs: Any,
) -> Iterator[T]:
    """Iterator variant of maybe_tqdm."""
    return iter(maybe_tqdm(iterable, desc=desc, total=total, show=show, **kwargs))
