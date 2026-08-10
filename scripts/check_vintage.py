"""Demonstrate ALFRED vintages: what did we know, and when?"""

from moirai.core.logging import configure_logging
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.series.temporal import revision_history, vintage_dates

configure_logging("WARNING")

# US real GDP is heavily revised, so it shows the machinery clearly.
with FredAdapter() as fred:
    series = fred.fetch(
        "GDPC1",
        realtime_start="2020-01-01",
        realtime_end="2023-12-31",
        observation_start="2020-01-01",
        observation_end="2020-12-31",
    )

print("observations :", len(series))
print("distinct periods:", len(series.periods))
print("has revisions:", series.has_revisions)
print("vintage dates:", len(vintage_dates(series)))
print()

from datetime import date

target = date(2020, 4, 1)  # Q2 2020, the pandemic collapse
print(f"Revision history for {target}:")
for observation in revision_history(series, target):
    print(f"  known {observation.known_at:%Y-%m-%d}  ->  {observation.value}")