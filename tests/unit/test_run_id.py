import pytest

from mss.io.config import sanitize_run_id


@pytest.mark.parametrize(
    "value",
    ["default", "attempt1", "run_v2", "a.b", "x-y", "_z"],
)
def test_sanitize_run_id_accepts(value: str) -> None:
    assert sanitize_run_id(value) == value


@pytest.mark.parametrize(
    "value",
    ["", " ", ".", "..", "a/b", "a\\b", "run id", "café"],
)
def test_sanitize_run_id_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        sanitize_run_id(value)
