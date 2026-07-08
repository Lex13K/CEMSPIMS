"""Named H4 stress-window presets for formal inference (Phase 6, #20)."""

from __future__ import annotations

from typing import Final

STRESS_PRESETS: Final[dict[str, tuple[str, str]]] = {
    "covid_2020": ("2020-02-01", "2020-05-31"),
    "rate_shock_2022": ("2022-01-01", "2022-10-31"),
}

DEFAULT_STRESS_PRESET_ID = "rate_shock_2022"


def preset_bounds(preset_id: str) -> tuple[str, str]:
    key = str(preset_id).strip()
    if key not in STRESS_PRESETS:
        allowed = ", ".join(sorted(STRESS_PRESETS))
        raise ValueError(f"Unknown hypothesis_stress_preset {preset_id!r}; expected one of: {allowed}")
    return STRESS_PRESETS[key]


def resolve_stress_window(
  *,
  preset_id: str,
  explicit_start: str | None,
  explicit_end: str | None,
) -> tuple[str, str, str]:
    """Return (start, end, active_preset_id).

    When both explicit bounds are provided in config, they override the preset
    and active_preset_id is empty.
    """
    if explicit_start is not None and explicit_end is not None:
        start = str(explicit_start).strip()
        end = str(explicit_end).strip()
        if not start or not end:
            raise ValueError("summary_test_stress_excl_start/end must be non-empty when set")
        return start, end, ""
    pid = str(preset_id).strip() or DEFAULT_STRESS_PRESET_ID
    start, end = preset_bounds(pid)
    return start, end, pid
