"""Provenance survives re-ingestion.

Written before the fix. The warehouse kept provenance in series_catalog, one
row per series, and every upsert replaced it: a second vintage overwrote the
first vintage's record, and an upsert without provenance wrote '{}' over it.
The observations of both vintages survived, but the earlier numbers could no
longer be traced to the bytes they came from.

The fix adds an append-only provenance_log. series_catalog.provenance stays
as the latest record, which get_provenance() returns, but an upsert without
provenance no longer erases it.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from moirai.core.exceptions import DataFabricError
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)
from moirai.engine.data_fabric.warehouse.duckdb_store import SCHEMA_VERSION, Warehouse

MAY = datetime(2025, 5, 30, tzinfo=UTC)
AUGUST = datetime(2025, 8, 30, tzinfo=UTC)
Q1 = date(2025, 1, 1)

MAY_FETCH = {"content_hash": "sha256:aaaa", "fetched_at": MAY.isoformat(), "url": "x"}
AUGUST_FETCH = {"content_hash": "sha256:bbbb", "fetched_at": AUGUST.isoformat(), "url": "x"}


def vintage(known_at: datetime, value: float) -> TimeSeries:
    """One vintage of Q1 Indian GDP growth, as published at `known_at`."""
    metadata = SeriesMetadata(
        series_id="in_gdp",
        title="India Real GDP Growth",
        source=Source.RBI,
        source_series_id="GDP_YOY",
        frequency=Frequency.QUARTERLY,
        unit=Unit.PERCENT,
        unit_label="Percent, year on year",
        seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
        geography="IN",
        start=Q1,
        end=Q1,
    )
    return TimeSeries.from_observations(
        metadata, [Observation(period=Q1, value=value, known_at=known_at)]
    )


@pytest.fixture
def warehouse() -> Iterator[Warehouse]:
    with Warehouse(":memory:") as store:
        yield store


# ---- the bug -----------------------------------------------------------------


def test_a_second_vintage_keeps_the_first_vintages_provenance(warehouse: Warehouse) -> None:
    warehouse.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)
    warehouse.upsert_series(vintage(AUGUST, 7.4), provenance=AUGUST_FETCH)

    assert warehouse.provenance_history("in_gdp") == (MAY_FETCH, AUGUST_FETCH)
    assert warehouse.get_provenance("in_gdp") == AUGUST_FETCH  # latest, as before


def test_an_upsert_without_provenance_does_not_erase_it(warehouse: Warehouse) -> None:
    warehouse.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)
    warehouse.upsert_series(vintage(MAY, 7.8))

    assert warehouse.get_provenance("in_gdp") == MAY_FETCH
    assert warehouse.provenance_history("in_gdp") == (MAY_FETCH,)


# ---- the log -----------------------------------------------------------------


def test_identical_provenance_is_logged_once(warehouse: Warehouse) -> None:
    warehouse.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)
    warehouse.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)

    assert warehouse.provenance_history("in_gdp") == (MAY_FETCH,)


def test_history_is_empty_when_no_provenance_was_given(warehouse: Warehouse) -> None:
    warehouse.upsert_series(vintage(MAY, 7.8))

    assert warehouse.provenance_history("in_gdp") == ()
    assert warehouse.get_provenance("in_gdp") == {}


def test_history_of_an_unknown_series_raises(warehouse: Warehouse) -> None:
    with pytest.raises(DataFabricError):
        warehouse.provenance_history("nope")


def test_deleting_a_series_deletes_its_history(warehouse: Warehouse) -> None:
    warehouse.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)
    warehouse.delete_series("in_gdp")

    assert warehouse.query("SELECT count(*) FROM provenance_log") == [(0,)]


# ---- upgrading a warehouse file written before the log existed -----------------


def test_an_old_warehouse_file_is_upgraded_and_keeps_its_provenance(tmp_path: Path) -> None:
    path = tmp_path / "w.duckdb"
    with Warehouse(path) as store:
        store.upsert_series(vintage(MAY, 7.8), provenance=MAY_FETCH)
        # Make the file look like schema version 1: no log, version 1.
        store.query("DROP TABLE provenance_log")
        store.query("UPDATE schema_meta SET value = '1' WHERE key = 'version'")

    with Warehouse(path) as store:
        assert store.schema_version == SCHEMA_VERSION == 2
        assert store.provenance_history("in_gdp") == (MAY_FETCH,)
