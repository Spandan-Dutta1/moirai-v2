"""Round-trip a synthetic file through the manual adapter."""

from pathlib import Path

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.data_fabric.ingestion.manual_csv import (
    CsvSpec,
    ManualCsvAdapter,
    PeriodFormat,
    parse_period,
)
from moirai.engine.data_fabric.series.models import Frequency, Unit

configure_logging("WARNING")

print("PERIOD FORMATS")
cases = [
    ("2024-03-01", PeriodFormat.ISO_DATE),
    ("Mar-24", PeriodFormat.MONTH_NAME_YEAR),
    ("March 2024", PeriodFormat.MONTH_NAME_YEAR),
    ("2024", PeriodFormat.CALENDAR_YEAR),
    ("2019-20", PeriodFormat.INDIAN_FISCAL_YEAR),
    ("2024Q1", PeriodFormat.CALENDAR_QUARTER),
    ("Q1 2024-25", PeriodFormat.INDIAN_FISCAL_QUARTER),
]
for raw, fmt in cases:
    print(f"  {raw:<14} [{fmt.value:<22}] -> {parse_period(raw, fmt)}")
print()
print("  note 2019-20 anchors to April, not January. A calendar reading")
print("  would misalign it against calendar-year data by nine months.")
print()

# A synthetic file in the shape RBI downloads take.
sample = get_paths().raw / "manual"
sample.mkdir(parents=True, exist_ok=True)
path = sample / "sample_repo_rate.csv"
path.write_text(
    "Month,Repo Rate\n"
    "Jan-24,6.50\n"
    "Feb-24,6.50\n"
    "Mar-24,6.50\n"
    "Apr-24,-\n"
    "May-24,6.50\n",
    encoding="utf-8",
)

spec = CsvSpec(
    series_id="in_repo_rate",
    title="RBI Policy Repo Rate",
    period_column="Month",
    value_column="Repo Rate",
    period_format=PeriodFormat.MONTH_NAME_YEAR,
    frequency=Frequency.MONTHLY,
    unit=Unit.PERCENT,
    unit_label="Percent per annum",
    publisher="Reserve Bank of India",
    source_url="https://data.rbi.org.in/",
)

series = ManualCsvAdapter(spec).ingest(path)

print("INGESTED")
print(f"  series     : {series.metadata.series_id}")
print(f"  title      : {series.metadata.title}")
print(f"  frequency  : {series.metadata.frequency.value}")
print(f"  unit       : {series.metadata.unit.value}")
print(f"  observations: {len(series)}")
print()
for observation in series.observations:
    shown = "missing" if observation.value is None else observation.value
    print(f"    {observation.period}  {shown}")
print()
print("  the dash became None, not zero: an absent figure is not a zero one")