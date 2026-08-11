"""
Adapter for published spreadsheets.

Statistical agencies publish tables designed to be read by people: merged
header cells spanning several rows, year separators breaking the data into
blocks, footnotes below the last observation, and a sentinel string where
a figure is unavailable. None of that is a defect, it is a layout, and a
parser that assumes a tidy rectangle will read it wrongly rather than fail.

Columns are addressed by index rather than by name. RBI's headers are
hierarchical, so a single cell reads "4.1   Policy Repo Rate" while its
meaning depends on the group header two rows above. Matching on that text
is brittle in a way that index selection is not, and the spec carries an
`expected_header` string that is checked on read, so a column moving is an
error rather than a silently different series.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import openpyxl
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import IngestionError
from moirai.core.logging import get_logger
from moirai.engine.data_fabric.ingestion.base import FetchResult, SourceAdapter
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)

log = get_logger(__name__)

#: Strings publishers use where a figure is unavailable.
MISSING_MARKERS = frozenset({"", "-", "--", "..", ".", "n/a", "na", "nil", "*", "#"})


class ExcelSpec(BaseModel):
    """How to read one column of a published spreadsheet into one series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1)
    title: str = Field(min_length=1)

    sheet: str = Field(min_length=1)
    date_column: int = Field(ge=1, description="1-based column index holding the period.")
    value_column: int = Field(ge=1, description="1-based column index holding the value.")
    first_data_row: int = Field(ge=1)
    last_data_row: int | None = Field(
        default=None, description="Omit to read to the end of the sheet."
    )

    expected_header: str = Field(
        default="",
        description=(
            "Substring the value column's header must contain. Checked on "
            "read so a column moving fails loudly rather than silently "
            "yielding a different series."
        ),
    )
    header_row: int = Field(default=0, ge=0, description="Row holding expected_header.")

    frequency: Frequency
    unit: Unit = Unit.UNKNOWN
    unit_label: str = ""
    seasonal_adjustment: SeasonalAdjustment = SeasonalAdjustment.UNKNOWN
    geography: str = "IN"
    publisher: str = ""
    source_url: str = ""
    license: str = ""
    notes: str = ""

    anchor_to_period_start: bool = Field(
        default=True,
        description=(
            "RBI dates a monthly observation to the last day of the month. "
            "Moirai anchors every period to its first day so series of "
            "different frequencies align without a per-frequency convention."
        ),
    )
    rebase_dates: tuple[date, ...] = ()

    def to_ledger_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["rebase_dates"] = [d.isoformat() for d in self.rebase_dates]
        return payload


def _to_period(value: Any, *, anchor_to_start: bool) -> date | None:
    """Interpret a date cell, or return None if the row is not an observation.

    Year separator rows carry a bare integer, footnote rows carry text, and
    both must be skipped rather than treated as failures: they are part of
    the layout, not corruption.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        return None  # a year separator or a footnote
    return parsed.replace(day=1) if anchor_to_start else parsed


def _to_value(cell: Any) -> float | None:
    """Read a numeric cell, treating publisher sentinels as missing.

    Rate cells sometimes carry a range such as "8.35/9.90", meaning the
    prevailing band across banks. That is not a single observation and is
    recorded as missing rather than silently reduced to one end of it.
    """
    if cell is None:
        return None
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        numeric = float(cell)
        if numeric != numeric:  # NaN
            return None
        return numeric

    text = str(cell).strip()
    if text.lower() in MISSING_MARKERS:
        return None
    if "/" in text:
        return None  # a band, not a point observation

    try:
        return float(text.replace(",", "").replace("%", ""))
    except ValueError:
        return None


class ManualExcelAdapter(SourceAdapter):
    """Reads one column of a downloaded spreadsheet into a canonical series.

    The file is read from disk rather than fetched, but the provenance
    guarantees are identical: bytes hashed, archived with a sidecar, parse
    a pure function of the archived bytes. `known_at` is the file's
    modification time, so reingesting the same download does not
    manufacture a new vintage.
    """

    supports_revisions = False

    def __init__(self, spec: ExcelSpec, *, source: Source = Source.RBI) -> None:
        self.spec = spec
        self._source = source

    @property
    def source(self) -> Source:
        return self._source

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        raw_path = options.get("path")
        if raw_path is None:
            raise IngestionError("spreadsheet ingestion requires a path= option")

        path = Path(raw_path)
        if not path.is_file():
            raise IngestionError(f"no file at {path}")

        content = path.read_bytes()
        if not content:
            raise IngestionError(f"{path} is empty")

        return FetchResult(
            source=self._source,
            source_series_id=source_series_id,
            url=self.spec.source_url or f"file://{path.as_posix()}",
            content=content,
            fetched_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            request_params={
                "path": str(path),
                "sheet": self.spec.sheet,
                "value_column": self.spec.value_column,
                "publisher": self.spec.publisher,
            },
        )

    def parse(self, result: FetchResult) -> TimeSeries:
        try:
            workbook = openpyxl.load_workbook(
                io.BytesIO(result.content), data_only=True, read_only=True
            )
        except Exception as err:
            raise IngestionError(f"could not open the workbook: {err}") from err

        try:
            if self.spec.sheet not in workbook.sheetnames:
                raise IngestionError(
                    f"sheet {self.spec.sheet!r} not found; "
                    f"have {workbook.sheetnames}"
                )
            sheet = workbook[self.spec.sheet]
            rows = list(sheet.iter_rows(values_only=True))
        finally:
            workbook.close()

        if self.spec.expected_header:
            self._verify_header(rows)

        last = self.spec.last_data_row or len(rows)
        observations: list[Observation] = []
        seen: set[date] = set()
        skipped = 0

        for index in range(self.spec.first_data_row - 1, min(last, len(rows))):
            row = rows[index]
            if len(row) < max(self.spec.date_column, self.spec.value_column):
                skipped += 1
                continue

            period = _to_period(
                row[self.spec.date_column - 1],
                anchor_to_start=self.spec.anchor_to_period_start,
            )
            if period is None:
                skipped += 1  # year separator, footnote, or blank
                continue

            if period in seen:
                raise IngestionError(
                    f"row {index + 1}: period {period} appears twice in "
                    f"{self.spec.series_id!r}"
                )
            seen.add(period)

            observations.append(
                Observation(
                    period=period,
                    value=_to_value(row[self.spec.value_column - 1]),
                    known_at=result.fetched_at,
                )
            )

        if not observations:
            raise IngestionError(
                f"no observations found for {self.spec.series_id!r}; check "
                f"first_data_row ({self.spec.first_data_row}) and the column "
                f"indices"
            )

        note = self.spec.notes
        if self.spec.rebase_dates:
            dates = ", ".join(d.isoformat() for d in self.spec.rebase_dates)
            note = (
                f"{note}\nREBASED at {dates}. A level shift there is not an "
                f"economic event."
            ).strip()

        metadata = SeriesMetadata(
            series_id=self.spec.series_id,
            title=self.spec.title,
            source=self._source,
            source_series_id=result.source_series_id,
            frequency=self.spec.frequency,
            unit=self.spec.unit,
            unit_label=self.spec.unit_label[:128],
            seasonal_adjustment=self.spec.seasonal_adjustment,
            geography=self.spec.geography,
            notes=note[:4096],
            license=self.spec.license[:256],
            retrieved_at=result.fetched_at,
        )

        series = TimeSeries.from_observations(metadata, observations)
        present = sum(1 for o in series.observations if not o.is_missing)

        log.info(
            "excel_series_parsed",
            series=self.spec.series_id,
            observations=len(series),
            with_values=present,
            rows_skipped=skipped,
            publisher=self.spec.publisher,
        )
        return series

    def _verify_header(self, rows: list[tuple]) -> None:
        """Confirm the value column still holds the expected series.

        Column indices are stable in practice but not guaranteed. Without
        this check a republished file with one inserted column would ingest
        a different series under the same name, and nothing in the numbers
        would reveal it.
        """
        row_index = (self.spec.header_row or 1) - 1
        if row_index >= len(rows):
            raise IngestionError(f"header row {self.spec.header_row} is beyond the sheet")

        row = rows[row_index]
        if len(row) < self.spec.value_column:
            raise IngestionError(
                f"column {self.spec.value_column} is beyond the header row"
            )

        found = str(row[self.spec.value_column - 1] or "")
        if self.spec.expected_header.lower() not in found.lower():
            raise IngestionError(
                f"column {self.spec.value_column} header is {found.strip()!r}, "
                f"expected it to contain {self.spec.expected_header!r}. The "
                f"publisher may have changed the layout."
            )

    def ingest(self, path: Path | str, **options: Any) -> TimeSeries:
        return self.fetch(self.spec.series_id, path=path, **options)


# ---- declared specs for the RBI monthly indicators table -------------------

RBI_INDICATORS_URL = "https://data.rbi.org.in/"
RBI_LICENSE = "Reserve Bank of India, terms at rbi.org.in"

#: Layout of RBIB Table No. 1, Select Economic Indicators, monthly sheet.
#: Headers span rows 4 to 6; data begins at row 8; year separators and a
#: notes footer are skipped by the date-cell check rather than by counting.
_RBI_SHEET = "Monthly"
_RBI_FIRST_DATA_ROW = 8
_RBI_DATE_COLUMN = 2


def _rbi_spec(
    series_id: str,
    title: str,
    column: int,
    expected_header: str,
    unit: Unit,
    unit_label: str,
    notes: str = "",
) -> ExcelSpec:
    return ExcelSpec(
        series_id=series_id,
        title=title,
        sheet=_RBI_SHEET,
        date_column=_RBI_DATE_COLUMN,
        value_column=column,
        first_data_row=_RBI_FIRST_DATA_ROW,
        expected_header=expected_header,
        header_row=5,
        frequency=Frequency.MONTHLY,
        unit=unit,
        unit_label=unit_label,
        seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
        geography="IN",
        publisher="Reserve Bank of India",
        source_url=RBI_INDICATORS_URL,
        license=RBI_LICENSE,
        notes=notes,
    )


#: Industrial production, published as a year on year percent change rather
#: than as an index level. Declaring the unit correctly is what stops the
#: preparation layer differencing an already differenced series.
RBI_IIP = _rbi_spec(
    series_id="in_iip_yoy",
    title="India Index of Industrial Production, year on year growth",
    column=3,
    expected_header="Index of Industrial Production",
    unit=Unit.PERCENT_CHANGE,
    unit_label="Percent change, year on year",
    notes=(
        "Published under the group header 'Real Sector (% Change)'. This is "
        "a growth rate, not an index level."
    ),
)

#: The policy rate is a level in percent and does need differencing.
RBI_REPO_RATE = _rbi_spec(
    series_id="in_repo_rate",
    title="RBI Policy Repo Rate",
    column=17,
    expected_header="Policy Repo Rate",
    unit=Unit.PERCENT,
    unit_label="Percent per annum",
    notes=(
        "The operating target changed over this sample: the repo rate became "
        "the single policy rate under the 2016 framework, and the corridor "
        "was restructured with the Standing Deposit Facility in 2022."
    ),
)

#: CPI inflation, again a year on year rate. Because it is a growth rate
#: rather than a level, the 2024 rebasing does not introduce a level shift,
#: though a small discontinuity remains where the series are spliced.
RBI_CPI_INFLATION = _rbi_spec(
    series_id="in_cpi_inflation",
    title="India All India Consumer Price Index inflation",
    column=36,
    expected_header="Consumer Price Index",
    unit=Unit.PERCENT_CHANGE,
    unit_label="Percent change, year on year",
    notes=(
        "Published under the group header 'Inflation (%)'. Coverage begins "
        "around 2012; earlier rows are blank because the CPI Combined series "
        "did not exist. The index was rebased to 2024 base; expressing it as "
        "a growth rate avoids the level shift but leaves a discontinuity at "
        "the splice."
    ),
)

#: Not part of the monetary VAR, but the series most relevant to Indian
#: government bond market research.
RBI_GSEC_10Y = _rbi_spec(
    series_id="in_gsec_10y",
    title="India 10 Year Government Security Par Yield (FBIL)",
    column=30,
    expected_header="G-Sec Par Yield",
    unit=Unit.PERCENT,
    unit_label="Percent per annum",
    notes="Financial Benchmarks India par yield for the ten year tenor.",
)

RBI_MONTHLY_SPECS: tuple[ExcelSpec, ...] = (
    RBI_IIP,
    RBI_REPO_RATE,
    RBI_CPI_INFLATION,
    RBI_GSEC_10Y,
)