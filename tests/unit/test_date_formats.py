"""Day and month order must be declared, never guessed (ADR 034).

Written before the fix. ISO_DATE tried day-first before month-first, so
03/04/2024 was always 3 April. A US file of month-first dates was read with
day and month swapped for every date up to the 12th, and silently: nothing
fails until the 13th, and by then the earlier rows are already wrong.
"""

from datetime import date

import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.manual_csv import PeriodFormat, parse_period

ISO = PeriodFormat.ISO_DATE


@pytest.mark.parametrize(
    ("raw", "expected"), [("2024-03-15", date(2024, 3, 1)), ("2024-03", date(2024, 3, 1))]
)
def test_iso_dates_are_read_without_a_declaration(raw: str, expected: date) -> None:
    assert parse_period(raw, ISO) == expected


@pytest.mark.parametrize("raw", ["03/04/2024", "03-04-2024", "13/04/2024"])
def test_iso_refuses_a_slashed_date_and_says_what_to_declare(raw: str) -> None:
    with pytest.raises(IngestionError, match="DAY_MONTH_YEAR"):
        parse_period(raw, ISO)


@pytest.mark.parametrize("raw", ["03/04/2024", "03-04-2024"])
def test_day_first_reads_3_april(raw: str) -> None:
    assert parse_period(raw, PeriodFormat.DAY_MONTH_YEAR) == date(2024, 4, 1)


@pytest.mark.parametrize("raw", ["03/04/2024", "03-04-2024"])
def test_month_first_reads_4_march(raw: str) -> None:
    assert parse_period(raw, PeriodFormat.MONTH_DAY_YEAR) == date(2024, 3, 1)


def test_day_first_refuses_a_thirteenth_month() -> None:
    with pytest.raises(IngestionError, match="day-first"):
        parse_period("04/13/2024", PeriodFormat.DAY_MONTH_YEAR)


def test_month_first_refuses_a_thirteenth_month() -> None:
    with pytest.raises(IngestionError, match="month-first"):
        parse_period("13/04/2024", PeriodFormat.MONTH_DAY_YEAR)


@pytest.mark.parametrize("name", ["DAY_MONTH_YEAR", "MONTH_DAY_YEAR"])
def test_a_two_digit_year_is_refused(name: str) -> None:
    with pytest.raises(IngestionError):
        parse_period("03/04/24", PeriodFormat[name])
