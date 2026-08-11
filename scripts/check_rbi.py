"""Ingest the RBI monthly indicators table."""

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_MONTHLY_SPECS,
    ManualExcelAdapter,
)
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

configure_logging("WARNING")

path = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"
if not path.exists():
    raise SystemExit(f"file not found: {path}")

print(f"reading {path.name}\n")

series_list = []
for spec in RBI_MONTHLY_SPECS:
    series = ManualExcelAdapter(spec).ingest(path)
    series_list.append(series)

    present = [o for o in series.observations if not o.is_missing]
    print(f"{spec.series_id}")
    print(f"  {series.metadata.title}")
    print(f"  unit        : {series.metadata.unit.value}  ({series.metadata.unit_label})")
    print(f"  observations: {len(series)}  ({len(present)} with values)")
    if present:
        print(f"  coverage    : {present[0].period} to {present[-1].period}")
        print(f"  latest      : {present[-1].value}")
    print()

print("LAST SIX MONTHS")
labels = [s.metadata.series_id for s in series_list]
print(f"  {'period':<12}" + "".join(f"{n:>18}" for n in labels))
recent = [o.period for o in series_list[0].observations][-6:]
for period in recent:
    row = ""
    for series in series_list:
        match = next((o for o in series.observations if o.period == period), None)
        value = "-" if match is None or match.is_missing else f"{match.value:.2f}"
        row += f"{value:>18}"
    print(f"  {str(period):<12}{row}")
print()

with Warehouse() as warehouse:
    for series in series_list:
        warehouse.upsert_series(
            series, provenance={"publisher": "RBI", "file": path.name}
        )
    print("warehouse:", warehouse.list_series())