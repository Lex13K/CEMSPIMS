"""Unit tests for v2 graph config: selection rule validation and node-feature naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from mss.graph.config import load_graph_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_node_feature_column_names_v2_default(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "[graph.node_features]\n"
        "rolling_mean = true\nrolling_vol = true\n"
        "include_log_dollar_volume = true\ninclude_turnover = true\n"
        "include_downside_semivol = true\ninclude_skew = true\n",
    )
    gc = load_graph_config(p)
    assert gc.node_feature_column_names() == (
        "rolling_mean",
        "rolling_vol",
        "log_dollar_volume",
        "turnover",
        "downside_semivol",
        "skew",
    )


def test_selection_rule_dollar_volume_accepted(tmp_path: Path) -> None:
    p = _write(tmp_path, "[graph.universe]\nselection_rule = \"dollar_volume\"\n")
    gc = load_graph_config(p)
    assert gc.selection_rule == "dollar_volume"


def test_selection_rule_invalid_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "[graph.universe]\nselection_rule = \"nonsense\"\n")
    with pytest.raises(ValueError, match="selection_rule"):
        load_graph_config(p)


def test_restrict_to_sp500_defaults_and_parse(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "[graph.universe]\nrestrict_to_sp500 = true\n"
        "sp500_membership_path = \"data/shared/interim/sp500_membership.parquet\"\n",
    )
    gc = load_graph_config(p)
    assert gc.restrict_to_sp500 is True
    assert gc.sp500_membership_path.endswith("sp500_membership.parquet")


def test_target_column_v2_default(tmp_path: Path) -> None:
    p = _write(tmp_path, "[graph.output]\ntarget_column = \"log_rv_fwd_30cal\"\n")
    gc = load_graph_config(p)
    assert gc.target_column == "log_rv_fwd_30cal"


def test_target_column_fallback_without_graph_output(tmp_path: Path) -> None:
    """Minimal TOML without [graph.output] resolves to v2 calendar target."""
    p = _write(tmp_path, "[graph]\nret_col = \"ret_used\"\n")
    gc = load_graph_config(p)
    assert gc.target_column == "log_rv_fwd_30cal"

    from mss.dataset.config import load_dataset_config

    dc = load_dataset_config(p)
    assert dc.target_column == "log_rv_fwd_30cal"
