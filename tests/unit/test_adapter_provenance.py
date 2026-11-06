"""Adapters hand the provenance of what they fetched to whoever stores it.

Written before the fix. SourceAdapter.fetch() built a FetchResult (content
hash, URL, fetch time), archived it and parsed it, then returned only the
TimeSeries. Nothing that reached the warehouse knew where the numbers came
from, so scripts stored labels such as {"pipeline": "run_pipeline"} instead.

provenance_for(series) returns the record of the latest fetch that produced
the series, so it can be stored next to it.
"""

from datetime import UTC, date, datetime
from typing import Any

import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.base import FetchResult, SourceAdapter, compute_hash
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

MAY = datetime(2025, 5, 30, tzinfo=UTC)
AUGUST = datetime(2025, 8, 30, tzinfo=UTC)
Q1 = date(2025, 1, 1)


class Stub(SourceAdapter):
    """Serves fixed bytes; the payload and fetch time can be changed per test."""

    def __init__(self) -> None:
        self.payload = b"7.8"
        self.fetched_at = MAY

    @property
    def source(self) -> Source:
        return Source.MANUAL

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        return FetchResult(
            source=Source.MANUAL,
            source_series_id=source_series_id,
            url=f"file://downloads/{source_series_id}.csv",
            content=self.payload,
            fetched_at=self.fetched_at,
        )

    def parse(self, result: FetchResult) -> TimeSeries:
        metadata = SeriesMetadata(
            series_id=result.source_series_id.lower(),
            title="Stub series",
            source=Source.MANUAL,
            source_series_id=result.source_series_id,
            frequency=Frequency.QUARTERLY,
            unit=Unit.PERCENT,
            unit_label="Percent",
            seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
            geography="IN",
            start=Q1,
            end=Q1,
        )
        value = float(result.content.decode())
        observation = Observation(period=Q1, value=value, known_at=result.fetched_at)
        return TimeSeries.from_observations(metadata, [observation])


@pytest.fixture
def stub() -> Stub:
    return Stub()


# archive=False throughout: the tests must not write into the real data/raw.


def test_provenance_for_describes_the_fetch(stub: Stub) -> None:
    series = stub.fetch("GDP", archive=False)
    record = stub.provenance_for(series)

    assert record["content_hash"] == compute_hash(b"7.8")
    assert record["url"] == "file://downloads/GDP.csv"
    assert record["fetched_at"] == MAY.isoformat()
    assert record["source_series_id"] == "GDP"
    assert record["archived"] is False


def test_a_series_this_adapter_did_not_fetch_has_no_provenance(stub: Stub) -> None:
    other = Stub().fetch("CPI", archive=False)
    with pytest.raises(IngestionError):
        stub.provenance_for(other)


def test_each_series_keeps_its_own_record(stub: Stub) -> None:
    gdp = stub.fetch("GDP", archive=False)
    stub.payload = b"4.4"
    cpi = stub.fetch("CPI", archive=False)

    assert stub.provenance_for(gdp)["content_hash"] == compute_hash(b"7.8")
    assert stub.provenance_for(cpi)["content_hash"] == compute_hash(b"4.4")


def test_a_refetch_replaces_that_series_record(stub: Stub) -> None:
    stub.fetch("GDP", archive=False)
    stub.payload, stub.fetched_at = b"7.4", AUGUST
    revised = stub.fetch("GDP", archive=False)

    record = stub.provenance_for(revised)
    assert record["content_hash"] == compute_hash(b"7.4")
    assert record["fetched_at"] == AUGUST.isoformat()


def test_the_returned_record_is_a_copy(stub: Stub) -> None:
    series = stub.fetch("GDP", archive=False)
    stub.provenance_for(series)["content_hash"] = "tampered"

    assert stub.provenance_for(series)["content_hash"] == compute_hash(b"7.8")


def test_two_vintages_reach_the_warehouse_traceable_to_their_bytes(stub: Stub) -> None:
    """The point of Part A and Part B together."""
    with Warehouse(":memory:") as warehouse:
        may = stub.fetch("GDP", archive=False)
        warehouse.upsert_series(may, provenance=stub.provenance_for(may))

        stub.payload, stub.fetched_at = b"7.4", AUGUST
        august = stub.fetch("GDP", archive=False)
        warehouse.upsert_series(august, provenance=stub.provenance_for(august))

        hashes = [r["content_hash"] for r in warehouse.provenance_history("gdp")]
        assert hashes == [compute_hash(b"7.8"), compute_hash(b"7.4")]
