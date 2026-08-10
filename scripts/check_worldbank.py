"""Smoke test against the live World Bank API, then into the warehouse."""

from moirai.core.logging import configure_logging
from moirai.engine.data_fabric.ingestion.worldbank import WorldBankAdapter
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

configure_logging("INFO")

with WorldBankAdapter() as wb:
    gdp_growth = wb.fetch_series("IND/NY.GDP.MKTP.KD.ZG", date_range="2000:2024")

print()
print("series_id :", gdp_growth.metadata.series_id)
print("title     :", gdp_growth.metadata.title)
print("frequency :", gdp_growth.metadata.frequency)
print("unit      :", gdp_growth.metadata.unit)
print("geography :", gdp_growth.metadata.geography)
print("count     :", len(gdp_growth))
print()
for observation in gdp_growth.observations[-6:]:
    print(f"  {observation.period.year}  {observation.value}")

# The abstraction test: identical warehouse code, a completely different source.
with Warehouse() as warehouse:
    warehouse.upsert_series(gdp_growth)
    print()
    print("series in warehouse:", warehouse.list_series())