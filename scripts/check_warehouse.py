"""End-to-end: FRED -> warehouse -> as-of query."""

from datetime import UTC, date, datetime

from moirai.core.logging import configure_logging
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

configure_logging("INFO")

with FredAdapter() as fred:
    series = fred.fetch_series(
        "GDPC1",
        realtime_start="2020-01-01",
        realtime_end="2023-12-31",
        observation_start="2020-01-01",
        observation_end="2020-12-31",
    )

with Warehouse() as warehouse:
    written = warehouse.upsert_series(series, provenance={"script": "check_warehouse"})
    print(f"\nwrote {written} observations")

    # Idempotency: writing the same data again must not duplicate it.
    warehouse.upsert_series(series)
    print("after re-ingest:", warehouse.count_observations("gdpc1"), "rows")

    print("series in warehouse:", warehouse.list_series())
    print()

    for when in ("2020-08-01", "2021-01-01", "2024-01-01"):
        moment = datetime.fromisoformat(when).replace(tzinfo=UTC)
        snapshot = warehouse.as_of("gdpc1", moment)
        q2 = next(
            (o.value for o in snapshot.observations if o.period == date(2020, 4, 1)), None
        )
        print(f"Q2 2020 GDP as known on {when}: {q2}")