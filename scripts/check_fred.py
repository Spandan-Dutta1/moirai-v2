"""Manual smoke test against the live FRED API."""

from moirai.core.logging import configure_logging
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("INFO")

with FredAdapter() as fred:
    series = fred.fetch_series("INDCPIALLMINMEI", observation_start="2024-01-01")

    print()
    print("series_id :", series.metadata.series_id)
    print("title     :", series.metadata.title)
    print("frequency :", series.metadata.frequency)
    print("units     :", series.metadata.unit, "|", series.metadata.unit_label)
    print("adjustment:", series.metadata.seasonal_adjustment)
    print("count     :", len(series))
    print()
    for observation in series.observations[:6]:
        print(f"  {observation.period}  {observation.value}")