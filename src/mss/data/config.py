"""Data-prepare parameters loaded from project TOML `[data.returns_panel]`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_VALID_USE_RET = frozenset({"ret", "retx"})


@dataclass(frozen=True)
class ReturnsPanelConfig:
    use_ret: str
    apply_delisting_adjustment: bool
    common_shares_only: bool
    major_exchanges_only: bool


def load_returns_panel_config(config_path: Path) -> ReturnsPanelConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    rp = (data.get("data") or {}).get("returns_panel") or {}
    use_ret = str(rp.get("use_ret", "retx")).lower()
    if use_ret not in _VALID_USE_RET:
        raise ValueError(f"[data.returns_panel].use_ret must be one of {sorted(_VALID_USE_RET)}; got {use_ret!r}")
    return ReturnsPanelConfig(
        use_ret=use_ret,
        apply_delisting_adjustment=bool(rp.get("apply_delisting_adjustment", False)),
        common_shares_only=bool(rp.get("common_shares_only", True)),
        major_exchanges_only=bool(rp.get("major_exchanges_only", False)),
    )
