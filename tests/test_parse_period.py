"""parse_period: a label must mean exactly one period, or be rejected.

Written before the fix. Three bugs, each a label that should be rejected but
is read silently as a valid period:

  * Indian fiscal year: the end year is never read, so "2019-25" is FY2019-20.
  * Indian fiscal quarter: the same, so "Q1 2024-30" is Q1 FY2024-25.
  * Month name: only the first three letters are read, so "Junk-24" is June.

The acceptance tests pass on the code as it stands and must keep passing.
The rejection tests fail until the fix.
"""

from datetime import date

import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.manual_csv import PeriodFormat, parse_period

FY = PeriodFormat.INDIAN_FISCAL_YEAR
FQ = PeriodFormat.INDIAN_FISCAL_QUARTER
MONTH = PeriodFormat.MONTH_NAME_YEAR


# ---- Indian fiscal year ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2019-20", date(2019, 4, 1)),
        ("2019-2020", date(2019, 4, 1)),
        ("2019/20", date(2019, 4, 1)),
        ("2019 - 20", date(2019, 4, 1)),
        ("1999-00", date(1999, 4, 1)),  # end year rolls over the century
        ("1999-2000", date(1999, 4, 1)),
    ],
)
def test_fiscal_year_accepts_consecutive_years(label: str, expected: date) -> None:
    assert parse_period(label, FY) == expected


@pytest.mark.parametrize(
    "label",
    [
        "2019-25",  # end year six years on
        "2019-19",  # end year equal to start
        "2019-2025",
        "2019-2019",
        "1999-01",  # wrong across the century rollover
        "2019-202",  # three-digit end year
    ],
)
def test_fiscal_year_rejects_an_end_year_that_is_not_the_next(label: str) -> None:
    with pytest.raises(IngestionError):
        parse_period(label, FY)


# ---- Indian fiscal quarter ---------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Q1 2024-25", date(2024, 4, 1)),
        ("Q3 2024-25", date(2024, 10, 1)),
        ("Q4 2024-25", date(2025, 1, 1)),  # January to March of the end year
        ("q2 2024-2025", date(2024, 7, 1)),
    ],
)
def test_fiscal_quarter_accepts_consecutive_years(label: str, expected: date) -> None:
    assert parse_period(label, FQ) == expected


@pytest.mark.parametrize("label", ["Q1 2024-30", "Q4 2024-2024", "Q2 2024-202"])
def test_fiscal_quarter_rejects_an_end_year_that_is_not_the_next(label: str) -> None:
    with pytest.raises(IngestionError):
        parse_period(label, FQ)


# ---- Month name and year -----------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Mar-24", date(2024, 3, 1)),
        ("March 2024", date(2024, 3, 1)),
        ("Sep-2024", date(2024, 9, 1)),
        ("Sept 2024", date(2024, 9, 1)),  # common four-letter abbreviation
        ("mar/24", date(2024, 3, 1)),
    ],
)
def test_month_name_accepts_names_and_abbreviations(label: str, expected: date) -> None:
    assert parse_period(label, MONTH) == expected


@pytest.mark.parametrize("label", ["Junk-24", "Marvel 2024", "Mayday-24"])
def test_month_name_rejects_words_that_only_start_like_a_month(label: str) -> None:
    with pytest.raises(IngestionError):
        parse_period(label, MONTH)
