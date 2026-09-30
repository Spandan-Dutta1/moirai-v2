"""Tests for the spreadsheet adapter.

Every Indian series in the project enters through this module, and until
now nothing tested it: the suite passed in an environment where openpyxl
was not even installed, because no test imported the adapter. The India
pipeline was protected only by running it.

Each test builds a small workbook in a temporary directory, laid out the
way RBI lays out its tables: headers above the data, year separator rows
carrying a bare integer, a notes footer below, and sentinel strings where
a figure is unavailable. The layout is the thing under test, so it is
reproduced rather than simplified away.

Nothing here writes to data/raw. Every fetch passes archive=False, and the
one test of archiving writes to a temporary directory.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import openpyxl
import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.manual_excel import (
    MISSING_MARKERS,
    RBI_CPI_INFLATION,
    RBI_GSEC_10Y,
    RBI_IIP,
    RBI_MONTHLY_SPECS,
    RBI_REPO_RATE,
    ExcelSpec,
    ManualExcelAdapter,
    _to_period,
    _to_value,
)
from moirai.engine.data_fabric.series.models import Frequency, Source, Unit

SHEET = "Monthly"


def make_spec(**overrides) -> ExcelSpec:
    defaults = dict(
        series_id="test_rate",
        title="A test rate",
        sheet=SHEET,
        date_column=1,
        value_column=2,
        first_data_row=3,
        expected_header="Policy Rate",
        header_row=2,
        frequency=Frequency.MONTHLY,
        unit=Unit.PERCENT,
        publisher="Test Publisher",
    )
    return ExcelSpec(**{**defaults, **overrides})


def write_workbook(path: Path, rows: list[list], sheet: str = SHEET) -> Path:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def rbi_like_rows() -> list[list]:
    """Title, header, then data broken by a year separator, then notes."""
    return [
        ["Select Economic Indicators", None],
        ["Period", "4.1 Policy Rate"],
        [2023, None],  # year separator
        [datetime(2023, 11, 30), 6.50],  # RBI dates months to their last day
        [datetime(2023, 12, 31), "6.50"],  # numbers sometimes arrive as text
        [2024, None],  # year separator
        [datetime(2024, 1, 31), "-"],  # sentinel for unavailable
        [datetime(2024, 2, 29), "8.35/9.90"],  # a band, not a point
        [datetime(2024, 3, 31), "1,234.5%"],
        ["Note: figures are provisional.", None],  # footer
    ]


def ingest(tmp_path: Path, rows: list[list], **spec_overrides):
    path = write_workbook(tmp_path / "table.xlsx", rows)
    adapter = ManualExcelAdapter(make_spec(**spec_overrides))
    return adapter.fetch(adapter.spec.series_id, path=path, archive=False)


# --- cell interpretation ---------------------------------------------------


@pytest.mark.parametrize("marker", sorted(MISSING_MARKERS))
def test_every_publisher_sentinel_reads_as_missing(marker):
    assert _to_value(marker) is None


@pytest.mark.parametrize(
    ("cell", "expected"),
    [(6.5, 6.5), (6, 6.0), ("6.50", 6.5), ("1,234.5", 1234.5), ("4.4%", 4.4), (" 7.1 ", 7.1)],
)
def test_numeric_cells_are_read(cell, expected):
    assert _to_value(cell) == expected


def test_a_band_is_missing_rather_than_one_end_of_it():
    """'8.35/9.90' is the range across banks. Reducing it to either end
    would invent a point observation that was never published."""
    assert _to_value("8.35/9.90") is None


@pytest.mark.parametrize("cell", [None, float("nan"), True, False, "provisional"])
def test_non_numbers_are_missing(cell):
    assert _to_value(cell) is None


def test_a_year_separator_is_not_a_period():
    assert _to_period(2024, anchor_to_start=True) is None


def test_footnote_text_is_not_a_period():
    assert _to_period("Note: provisional", anchor_to_start=True) is None


def test_month_end_dates_are_anchored_to_the_first():
    assert _to_period(datetime(2024, 2, 29), anchor_to_start=True) == date(2024, 2, 1)


def test_anchoring_can_be_switched_off():
    assert _to_period(date(2024, 2, 29), anchor_to_start=False) == date(2024, 2, 29)


# --- reading a laid-out table ----------------------------------------------


def test_the_rbi_layout_parses_to_one_row_per_month(tmp_path):
    series = ingest(tmp_path, rbi_like_rows())
    assert [o.period for o in series.observations] == [
        date(2023, 11, 1),
        date(2023, 12, 1),
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 3, 1),
    ]


def test_values_follow_the_cell_rules(tmp_path):
    values = [o.value for o in ingest(tmp_path, rbi_like_rows()).observations]
    assert values == [6.5, 6.5, None, None, 1234.5]


def test_missing_months_are_kept_as_missing_rather_than_dropped(tmp_path):
    """A gap stays visible as a missing observation. Dropping it here would
    hide it from preparation, which is where the decision belongs."""
    series = ingest(tmp_path, rbi_like_rows())
    assert sum(1 for o in series.observations if o.is_missing) == 2
    assert len(series) == 5


def test_metadata_comes_from_the_spec(tmp_path):
    metadata = ingest(tmp_path, rbi_like_rows()).metadata
    assert metadata.series_id == "test_rate"
    assert metadata.unit is Unit.PERCENT
    assert metadata.frequency is Frequency.MONTHLY
    assert metadata.source is Source.RBI


def test_last_data_row_stops_reading(tmp_path):
    series = ingest(tmp_path, rbi_like_rows(), last_data_row=5)
    assert [o.period for o in series.observations] == [date(2023, 11, 1), date(2023, 12, 1)]


def test_a_rebased_series_carries_a_warning_in_its_notes(tmp_path):
    series = ingest(tmp_path, rbi_like_rows(), rebase_dates=(date(2024, 1, 1),))
    assert "REBASED at 2024-01-01" in series.metadata.notes


def test_known_at_is_the_file_modification_time(tmp_path):
    """Re-ingesting the same download must not manufacture a new vintage."""
    path = write_workbook(tmp_path / "table.xlsx", rbi_like_rows())
    stamp = datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp()
    os.utime(path, (stamp, stamp))
    adapter = ManualExcelAdapter(make_spec())
    first = adapter.fetch("test_rate", path=path, archive=False)
    second = adapter.fetch("test_rate", path=path, archive=False)
    known = {o.known_at for o in first.observations} | {o.known_at for o in second.observations}
    assert known == {datetime(2026, 5, 1, 12, 0, tzinfo=UTC)}


# --- loud failures ---------------------------------------------------------


def test_a_moved_column_is_an_error_not_a_different_series(tmp_path):
    """The failure the header check exists for: a republished file with a
    column inserted would otherwise ingest the wrong series by name."""
    rows = rbi_like_rows()
    rows[1] = ["Period", "4.2 Reverse Repo Rate"]
    with pytest.raises(IngestionError, match="publisher may have changed the layout"):
        ingest(tmp_path, rows)


def test_the_header_check_is_case_insensitive(tmp_path):
    rows = rbi_like_rows()
    rows[1] = ["Period", "4.1 POLICY RATE"]
    assert len(ingest(tmp_path, rows)) == 5


def test_a_duplicated_period_is_an_error(tmp_path):
    rows = rbi_like_rows()
    rows.insert(5, [datetime(2023, 12, 31), 6.75])
    with pytest.raises(IngestionError, match="appears twice"):
        ingest(tmp_path, rows)


def test_a_missing_sheet_is_an_error(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        ingest(tmp_path, rbi_like_rows(), sheet="Quarterly")


def test_a_sheet_with_no_observations_is_an_error(tmp_path):
    with pytest.raises(IngestionError, match="no observations found"):
        ingest(tmp_path, rbi_like_rows(), first_data_row=10)


def test_a_missing_file_is_an_error(tmp_path):
    adapter = ManualExcelAdapter(make_spec())
    with pytest.raises(IngestionError, match="no file at"):
        adapter.fetch("test_rate", path=tmp_path / "absent.xlsx", archive=False)


def test_an_empty_file_is_an_error(tmp_path):
    path = tmp_path / "empty.xlsx"
    path.write_bytes(b"")
    adapter = ManualExcelAdapter(make_spec())
    with pytest.raises(IngestionError, match="is empty"):
        adapter.fetch("test_rate", path=path, archive=False)


def test_a_file_that_is_not_a_workbook_is_an_error(tmp_path):
    path = tmp_path / "table.xlsx"
    path.write_bytes(b"this is not a spreadsheet")
    adapter = ManualExcelAdapter(make_spec())
    with pytest.raises(IngestionError, match="could not open the workbook"):
        adapter.fetch("test_rate", path=path, archive=False)


def test_a_path_is_required():
    adapter = ManualExcelAdapter(make_spec())
    with pytest.raises(IngestionError, match="requires a path"):
        adapter.fetch("test_rate", archive=False)


# --- provenance ------------------------------------------------------------


def test_archiving_writes_the_exact_bytes_and_a_sidecar(tmp_path):
    path = write_workbook(tmp_path / "table.xlsx", rbi_like_rows())
    adapter = ManualExcelAdapter(make_spec())
    result = adapter.fetch_raw("test_rate", path=path)
    payload = result.archive(tmp_path / "raw")
    assert payload.read_bytes() == path.read_bytes()
    assert list((tmp_path / "raw").glob("*.provenance.json"))


def test_parsing_is_a_function_of_the_bytes(tmp_path):
    path = write_workbook(tmp_path / "table.xlsx", rbi_like_rows())
    adapter = ManualExcelAdapter(make_spec())
    result = adapter.fetch_raw("test_rate", path=path)
    first, second = adapter.parse(result), adapter.parse(result)
    assert [(o.period, o.value) for o in first.observations] == [
        (o.period, o.value) for o in second.observations
    ]


# --- the declared RBI specs ------------------------------------------------


def test_the_growth_rates_are_declared_as_percent_change():
    """This declaration is what stops preparation differencing a series
    RBI already publishes as a growth rate."""
    assert RBI_IIP.unit is Unit.PERCENT_CHANGE
    assert RBI_CPI_INFLATION.unit is Unit.PERCENT_CHANGE


def test_the_rates_are_declared_as_percent_levels():
    assert RBI_REPO_RATE.unit is Unit.PERCENT
    assert RBI_GSEC_10Y.unit is Unit.PERCENT


def test_every_rbi_spec_checks_its_header():
    for spec in RBI_MONTHLY_SPECS:
        assert spec.expected_header, f"{spec.series_id} has no header check"


def test_rbi_specs_point_at_distinct_columns():
    columns = [spec.value_column for spec in RBI_MONTHLY_SPECS]
    assert len(set(columns)) == len(columns)


def test_rbi_series_ids_are_unique():
    ids = [spec.series_id for spec in RBI_MONTHLY_SPECS]
    assert len(set(ids)) == len(ids)
