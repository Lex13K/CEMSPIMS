import math

import pytest

from mss.data.ingest import is_close_numeric, normalize_columns_lower


def test_normalize_columns_lower_strips_and_lowers() -> None:
    assert normalize_columns_lower(["Date", " PERMNO ", "Ret"]) == ["date", "permno", "ret"]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 1, True),
        (1.0, 1.0, True),
        (None, None, True),
        (1.0, 1.000000001, True),
        ("x", "x", True),
        (float("nan"), float("nan"), True),
        (None, float("nan"), True),
        (1, 2, False),
    ],
)
def test_is_close_numeric(a, b, expected) -> None:
    atol, rtol = 1e-8, 1e-8
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        assert is_close_numeric(a, b, atol=atol, rtol=rtol) is True
    else:
        assert is_close_numeric(a, b, atol=atol, rtol=rtol) is expected
