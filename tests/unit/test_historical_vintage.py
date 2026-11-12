"""A historical validation reads its FRED indices as of a fixed vintage (ADR 032).

Written before the change. FRED's annual PCE revision of September 2026 moved
the 2022 mean from the declared 6.545% to 6.562%, and validate_2022 began
refusing to run with no change to the code. The scripts live in scripts/, so
the module is imported from there; the network is replaced by a fake.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import historical_validation as hv  # noqa: E402


class FakeFred:
    """Records the options of each request and returns two years of levels."""

    calls: list[dict[str, Any]] = []

    def __enter__(self) -> "FakeFred":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def fetch_series(self, code: str, **options: Any) -> str:
        FakeFred.calls.append({"code": code, **options})
        return code


def levels(_series: object) -> tuple[list[date], list[float]]:
    periods = [date(y, m, 1) for y in (2021, 2022) for m in range(1, 13)]
    return periods, [100.0 + i for i in range(24)]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFred.calls = []
    monkeypatch.setattr(hv, "FredAdapter", FakeFred)
    monkeypatch.setattr(hv, "to_array", levels)


def test_a_fred_index_reads_the_current_vintage_unless_told_otherwise() -> None:
    assert hv.FredIndex("PCEPI").vintage is None


def test_a_fred_index_can_name_its_vintage() -> None:
    assert hv.FredIndex("PCEPI", date(2026, 7, 1)).vintage == date(2026, 7, 1)


def test_a_vintage_is_requested_as_a_one_day_alfred_window() -> None:
    hv.fetch("PCEPI", "2021-01-01", "2022-12-31", date(2026, 7, 1))
    call = FakeFred.calls[-1]
    assert call["realtime_start"] == "2026-07-01"
    assert call["realtime_end"] == "2026-07-01"


def test_without_a_vintage_no_alfred_window_is_requested() -> None:
    hv.fetch("PCEPI", "2021-01-01", "2022-12-31")
    call = FakeFred.calls[-1]
    assert "realtime_start" not in call
    assert "realtime_end" not in call


def test_average_inflation_passes_the_vintage_through() -> None:
    hv.average_inflation("PCEPI", 2022, date(2026, 7, 1))
    assert FakeFred.calls[-1]["realtime_start"] == "2026-07-01"
