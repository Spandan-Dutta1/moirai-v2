"""
Adapter for sources that publish downloads rather than APIs.

RBI's data portal, MoSPI's catalogue and most Indian statistical releases
offer spreadsheets and CSV files, not endpoints. Scraping them is fragile:
a page redesign breaks the parser, and the failure is usually silent
because the scraper finds *something* and parses it wrongly. Downloading
the file by hand once a month and ingesting it here is less elegant and
far more robust, and it loses nothing that matters, since the provenance
guarantees are identical: the bytes are hashed, archived, and replayable.

Column names and date formats vary by publisher and by file, so the
mapping is declared per file rather than inferred. A CsvSpec says which
column holds the period, which holds the value, how the period is
written, and what frequency it represents.

Two India-specific problems this exists to handle:

  * Fiscal years. "2019-20" means April 2019 to March 2020. Read as a
    calendar year it misaligns against calendar-year data by up to nine
    months, and nothing about the numbers reveals the error.

  * Rebasing. India's CPI moved to a 2024 base year, and the level shifts
    for reasons unrelated to prices. Differencing across that boundary
    produces a phantom shock. The spec records known rebase dates so the
    break travels with the data instead of being discovered later by
    someone puzzled at an impulse response.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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

#: Values that publishers use to mean "no observation".
MISSING_MARKERS = frozenset({"", "-", "--", "n/a", "na", "nil", ".", "..", "*"})


class PeriodFormat(StrEnum):
    """How a period is written in the source file."""

    #: 2024-03-01, 2024-03, or 01/03/2024 handled by ISO-ish parsing.
    ISO_DATE = "iso_date"
    #: Mar-24, Mar-2024, March 2024.
    MONTH_NAME_YEAR = "month_name_year"
    #: 2024 as a calendar year.
    CALENDAR_YEAR = "calendar_year"
    #: 2019-20 meaning April 2019 to March 2020. Anchored to the April start.
    INDIAN_FISCAL_YEAR = "indian_fiscal_year"
    #: 2024Q1 or Q1 2024, calendar quarters.
    CALENDAR_QUARTER = "calendar_quarter"
    #: Q1 2024-25, the Indian fiscal quarter starting April.
    INDIAN_FISCAL_QUARTER = "indian_fiscal_quarter"


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class CsvSpec(BaseModel):
    """How to read one downloaded file into one series.

    Declared rather than inferred. Inference would work for most files and
    fail silently for the rest, which in a provenance system is worse than
    failing loudly for all of them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    period_column: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    period_format: PeriodFormat
    frequency: Frequency

    unit: Unit = Unit.UNKNOWN
    unit_label: str = ""
    seasonal_adjustment: SeasonalAdjustment = SeasonalAdjustment.UNKNOWN
    geography: str = "IN"
    publisher: str = Field(default="", description="Who published the file.")
    source_url: str = Field(default="", description="Where it was downloaded from.")
    license: str = ""
    notes: str = ""

    skip_rows: int = Field(default=0, ge=0, description="Preamble rows before the header.")
    thousands_separator: str = Field(default=",", max_length=1)
    rebase_dates: tuple[date, ...] = Field(
        default=(),
        description=(
            "Periods at which the index was rebased. A level shift here is "
            "not an economic event and must not be differenced through."
        ),
    )

    def to_ledger_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["rebase_dates"] = [d.isoformat() for d in self.rebase_dates]
        return payload


# ---- period parsing -------------------------------------------------------


def _fail(raw: str, expected: str) -> None:
    raise IngestionError(f"could not read period {raw!r}; expected {expected}")


def parse_period(raw: str, period_format: PeriodFormat) -> date:
    """Turn a period label into the date the period begins.

    Every period is anchored to its first day, so a monthly, quarterly and
    annual series can sit in the same warehouse and be compared without a
    separate convention for each.
    """
    text = raw.strip()
    if not text:
        raise IngestionError("period cell is empty")

    if period_format is PeriodFormat.ISO_DATE:
        for pattern in ("%Y-%m-%d", "%Y-%m", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, pattern).date().replace(day=1)
            except ValueError:
                continue
        _fail(text, "an ISO-like date")

    if period_format is PeriodFormat.MONTH_NAME_YEAR:
        match = re.match(r"^([A-Za-z]{3,9})[\s\-/]+(\d{2,4})$", text)
        if not match:
            _fail(text, "a month name and year such as Mar-24")
        month = MONTHS.get(match.group(1)[:3].lower())
        if month is None:
            _fail(text, "a recognisable month name")
        year = int(match.group(2))
        if year < 100:
            year += 2000 if year < 70 else 1900
        return date(year, month, 1)

    if period_format is PeriodFormat.CALENDAR_YEAR:
        if not re.fullmatch(r"\d{4}", text):
            _fail(text, "a four digit year")
        return date(int(text), 1, 1)

    if period_format is PeriodFormat.INDIAN_FISCAL_YEAR:
        # 2019-20 or 2019-2020, both meaning April 2019 to March 2020.
        match = re.fullmatch(r"(\d{4})\s*[-/]\s*(\d{2,4})", text)
        if not match:
            _fail(text, "an Indian fiscal year such as 2019-20")
        return date(int(match.group(1)), 4, 1)

    if period_format is PeriodFormat.CALENDAR_QUARTER:
        match = re.fullmatch(r"(?:(\d{4})\s*[-\s]?Q([1-4])|Q([1-4])\s*[-\s]?(\d{4}))", text, re.I)
        if not match:
            _fail(text, "a calendar quarter such as 2024Q1")
        year = int(match.group(1) or match.group(4))
        quarter = int(match.group(2) or match.group(3))
        return date(year, 3 * (quarter - 1) + 1, 1)

    if period_format is PeriodFormat.INDIAN_FISCAL_QUARTER:
        # Q1 2024-25 is April to June 2024.
        match = re.fullmatch(r"Q([1-4])\s+(\d{4})\s*[-/]\s*(\d{2,4})", text, re.I)
        if not match:
            _fail(text, "an Indian fiscal quarter such as Q1 2024-25")
        quarter = int(match.group(1))
        start_year = int(match.group(2))
        month = 4 + 3 * (quarter - 1)
        if month > 12:
            month -= 12
            start_year += 1
        return date(start_year, month, 1)

    raise IngestionError(f"unhandled period format: {period_format}")


def parse_value(raw: str, thousands_separator: str) -> float | None:
    """Read a numeric cell, treating publisher sentinels as missing.

    A missing observation is information: the publisher printed a dash
    because the figure was not available, and turning that into zero would
    invent data rather than record its absence.
    """
    text = raw.strip()
    if text.lower() in MISSING_MARKERS:
        return None

    cleaned = text.replace(thousands_separator, "").replace("%", "").strip()
    # Parenthesised negatives, an accounting convention some tables use.
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    try:
        return float(cleaned)
    except ValueError as err:
        raise IngestionError(f"value {raw!r} is not numeric") from err


# ---- adapter --------------------------------------------------------------


class ManualCsvAdapter(SourceAdapter):
    """Ingests a downloaded file according to a declared spec.

    The file is read from disk rather than fetched, but everything after
    that is identical to an API adapter: the bytes are hashed, archived
    with a provenance sidecar, and the parse is a pure function of them.
    A file ingested today can be reparsed in five years from the archive.

    `known_at` is the file's modification time rather than now, so
    reingesting the same download does not silently create a new vintage.
    """

    supports_revisions = False

    def __init__(self, spec: CsvSpec, *, source: Source = Source.MANUAL) -> None:
        self.spec = spec
        self._source = source

    @property
    def source(self) -> Source:
        return self._source

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        """Read a downloaded file from disk.

        Options
        -------
        path
            Path to the file. Required.
        """
        raw_path = options.get("path")
        if raw_path is None:
            raise IngestionError("manual ingestion requires a path= option")

        path = Path(raw_path)
        if not path.is_file():
            raise IngestionError(f"no file at {path}")

        content = path.read_bytes()
        if not content:
            raise IngestionError(f"{path} is empty")

        # File mtime, not now: reingesting a download is not a new vintage.
        known_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

        return FetchResult(
            source=self._source,
            source_series_id=source_series_id,
            url=self.spec.source_url or f"file://{path.as_posix()}",
            content=content,
            fetched_at=known_at,
            request_params={
                "path": str(path),
                "spec": self.spec.series_id,
                "publisher": self.spec.publisher,
            },
        )

    def parse(self, result: FetchResult) -> TimeSeries:
        """Read the archived bytes into a canonical series."""
        # utf-8-sig strips a BOM if the publisher wrote one, which Windows
        # tooling frequently does and which would corrupt the first header.
        try:
            text = result.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = result.content.decode("latin-1")
            except UnicodeDecodeError as err:
                raise IngestionError("file is not readable as text") from err

        lines = text.splitlines()
        if self.spec.skip_rows:
            lines = lines[self.spec.skip_rows :]
        if not lines:
            raise IngestionError("no rows left after skipping the preamble")

        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise IngestionError("file has no header row")

        headers = {name.strip(): name for name in reader.fieldnames if name}
        for column in (self.spec.period_column, self.spec.value_column):
            if column not in headers:
                raise IngestionError(
                    f"column {column!r} is not in the file; found {sorted(headers)}"
                )

        observations: list[Observation] = []
        seen: set[date] = set()

        for number, row in enumerate(reader, start=2 + self.spec.skip_rows):
            raw_period = (row.get(headers[self.spec.period_column]) or "").strip()
            if not raw_period:
                continue  # trailing blank or a footnote row

            try:
                period = parse_period(raw_period, self.spec.period_format)
                value = parse_value(
                    row.get(headers[self.spec.value_column]) or "",
                    self.spec.thousands_separator,
                )
            except IngestionError as err:
                raise IngestionError(f"row {number}: {err}") from err

            if period in seen:
                raise IngestionError(f"row {number}: period {period} appears twice")
            seen.add(period)

            observations.append(
                Observation(period=period, value=value, known_at=result.fetched_at)
            )

        if not observations:
            raise IngestionError("file contained no usable rows")

        note = self.spec.notes
        if self.spec.rebase_dates:
            dates = ", ".join(d.isoformat() for d in self.spec.rebase_dates)
            note = (
                f"{note}\nREBASED at {dates}. The level shift there is not an "
                f"economic event; do not difference across it without handling "
                f"the break."
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

        log.info(
            "manual_csv_parsed",
            series=self.spec.series_id,
            observations=len(series),
            publisher=self.spec.publisher,
            rebased=bool(self.spec.rebase_dates),
        )
        return series

    def ingest(self, path: Path | str, **options: Any) -> TimeSeries:
        """Read, archive and parse a downloaded file."""
        return self.fetch(self.spec.series_id, path=path, **options)