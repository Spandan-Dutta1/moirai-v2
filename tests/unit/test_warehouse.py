"""Tests for the DuckDB bitemporal warehouse.

Every test runs against an in-memory database. That keeps them fast and
guarantees no test can see another's data or touch the real warehouse file.

The fixture scenario mirrors a real release pattern: Q1 published in May,
revised in August; Q2 arrives in August.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from moirai.core.exceptions import DataFabricError, VintageError
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)
from moirai.engine.data_fabric.warehouse.duckdb_store import (
    SCHEMA_VERSION,
    Warehouse,
)

MAY = datetime(2025, 5, 30, tzinfo=UTC)
JUNE = datetime(2025, 6, 15, tzinfo=UTC)
AUGUST = datetime(2025, 8, 30, tzinfo=UTC)
NOVEMBER = datetime(2025, 11, 30, tzinfo=UTC)

Q1 = date(2025, 1, 1)
Q2 = date(2025, 4, 1)


def metadata(**overrides) -> SeriesMetadata:
    defaults = dict(
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
        end=Q2,
    )
    return SeriesMetadata(**{**defaults, **overrides})


def obs(period, value, known_at, revision=0) -> Observation:
    return Observation(period=period, value=value, known_at=known_at, revision=revision)


@pytest.fixture
def warehouse() -> Warehouse:
    with Warehouse(":memory:") as store:
        yield store


@pytest.fixture
def gdp() -> TimeSeries:
    """Q1 published in May at 7.8, revised in August to 7.4. Q2 arrives in August."""
    return TimeSeries.from_observations(
        metadata(),
        [
            obs(Q1, 7.8, MAY),
            obs(Q1, 7.4, AUGUST, revision=1),
            obs(Q2, 6.5, AUGUST),
        ],
    )


def values(series: TimeSeries) -> dict:
    return {o.period: o.value for o in series.observations}


# --- schema ----------------------------------------------------------------

def test_schema_version_is_recorded(warehouse):
    assert warehouse.schema_version == SCHEMA_VERSION


def test_schema_creation_is_idempotent(tmp_path: Path):
    path = tmp_path / "w.duckdb"
    Warehouse(path).close()
    second = Warehouse(path)  # must not fail on existing tables
    assert second.schema_version == SCHEMA_VERSION
    second.close()


def test_a_new_warehouse_is_empty(warehouse):
    assert warehouse.list_series() == ()
    assert warehouse.count_observations() == 0


def test_parent_directory_is_created(tmp_path: Path):
    path = tmp_path / "nested" / "deeper" / "w.duckdb"
    Warehouse(path).close()
    assert path.exists()


# --- writing ---------------------------------------------------------------

def test_upsert_returns_the_row_count(warehouse, gdp):
    assert warehouse.upsert_series(gdp) == 3


def test_upsert_registers_the_series(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.list_series() == ("in_gdp",)
    assert warehouse.has_series("in_gdp") is True


def test_upsert_is_idempotent(warehouse, gdp):
    """Re-ingesting identical data must not duplicate rows."""
    warehouse.upsert_series(gdp)
    warehouse.upsert_series(gdp)
    assert warehouse.count_observations("in_gdp") == 3


def test_upsert_updates_a_changed_value(warehouse, gdp):
    """Same vintage, corrected value: the row is replaced, not duplicated."""
    warehouse.upsert_series(gdp)
    corrected = TimeSeries.from_observations(metadata(), [obs(Q1, 7.9, MAY)])
    warehouse.upsert_series(corrected)

    assert warehouse.count_observations("in_gdp") == 3
    assert values(warehouse.as_of("in_gdp", JUNE))[Q1] == 7.9


def test_upsert_refreshes_metadata(warehouse, gdp):
    warehouse.upsert_series(gdp)
    renamed = TimeSeries(metadata=metadata(title="Revised Title"), observations=gdp.observations)
    warehouse.upsert_series(renamed)
    assert warehouse.get_metadata("in_gdp").title == "Revised Title"


def test_empty_series_stores_metadata_only(warehouse):
    warehouse.upsert_series(TimeSeries(metadata=metadata()))
    assert warehouse.has_series("in_gdp") is True
    assert warehouse.count_observations("in_gdp") == 0


def test_missing_values_round_trip(warehouse):
    series = TimeSeries.from_observations(metadata(), [obs(Q1, None, MAY)])
    warehouse.upsert_series(series)
    assert warehouse.get_series("in_gdp").observations[0].value is None


def test_provenance_is_stored(warehouse, gdp):
    warehouse.upsert_series(gdp, provenance={"content_hash": "sha256:abc", "url": "x"})
    assert warehouse.get_provenance("in_gdp")["content_hash"] == "sha256:abc"


def test_provenance_defaults_to_empty(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.get_provenance("in_gdp") == {}


# --- metadata round trip ---------------------------------------------------

def test_metadata_round_trips_exactly(warehouse, gdp):
    warehouse.upsert_series(gdp)
    restored = warehouse.get_metadata("in_gdp")

    assert restored.frequency is Frequency.QUARTERLY
    assert restored.unit is Unit.PERCENT
    assert restored.unit_label == "Percent, year on year"
    assert restored.seasonal_adjustment is SeasonalAdjustment.NOT_ADJUSTED
    assert restored.source is Source.RBI
    assert restored.geography == "IN"
    assert restored.start == Q1


def test_metadata_for_an_unknown_series_raises(warehouse):
    with pytest.raises(DataFabricError, match="not in the warehouse"):
        warehouse.get_metadata("nope")


def test_provenance_for_an_unknown_series_raises(warehouse):
    with pytest.raises(DataFabricError, match="not in the warehouse"):
        warehouse.get_provenance("nope")


# --- get_series ------------------------------------------------------------

def test_get_series_returns_every_vintage(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert len(warehouse.get_series("in_gdp")) == 3


def test_get_series_preserves_revisions(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.get_series("in_gdp").has_revisions is True


def test_get_series_is_ordered(warehouse, gdp):
    warehouse.upsert_series(gdp)
    observations = warehouse.get_series("in_gdp").observations
    keys = [(o.period, o.known_at, o.revision) for o in observations]
    assert keys == sorted(keys)


def test_get_series_respects_a_period_window(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.get_series("in_gdp", start=Q2).periods == (Q2,)


# --- as_of -----------------------------------------------------------------

def test_as_of_returns_the_vintage_available_then(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert values(warehouse.as_of("in_gdp", JUNE)) == {Q1: 7.8}


def test_as_of_excludes_periods_published_later(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert Q2 not in values(warehouse.as_of("in_gdp", JUNE))


def test_as_of_picks_the_revision_once_visible(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert values(warehouse.as_of("in_gdp", NOVEMBER))[Q1] == 7.4


def test_as_of_boundary_is_inclusive(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert values(warehouse.as_of("in_gdp", MAY)) == {Q1: 7.8}


def test_as_of_before_publication_is_empty(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.as_of("in_gdp", datetime(2025, 1, 1, tzinfo=UTC)).is_empty


def test_as_of_returns_one_row_per_period(warehouse, gdp):
    warehouse.upsert_series(gdp)
    result = warehouse.as_of("in_gdp", NOVEMBER)
    assert len(result) == len(result.periods)


def test_as_of_rejects_naive_datetimes(warehouse, gdp):
    warehouse.upsert_series(gdp)
    with pytest.raises(VintageError, match="timezone-aware"):
        warehouse.as_of("in_gdp", datetime(2025, 6, 15))


def test_as_of_respects_a_period_window(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.as_of("in_gdp", NOVEMBER, start=Q2).periods == (Q2,)


# --- latest ----------------------------------------------------------------

def test_latest_uses_the_most_recent_vintage(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert values(warehouse.latest("in_gdp")) == {Q1: 7.4, Q2: 6.5}


def test_latest_collapses_revisions(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.latest("in_gdp").has_revisions is False


def test_latest_matches_a_far_future_as_of(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert values(warehouse.latest("in_gdp")) == values(
        warehouse.as_of("in_gdp", datetime(2030, 1, 1, tzinfo=UTC))
    )


# --- vintages and counts ---------------------------------------------------

def test_vintage_dates_are_distinct_and_ordered(warehouse, gdp):
    warehouse.upsert_series(gdp)
    assert warehouse.vintage_dates("in_gdp") == (MAY, AUGUST)


def test_count_observations_across_all_series(warehouse, gdp):
    warehouse.upsert_series(gdp)
    warehouse.upsert_series(
        TimeSeries.from_observations(metadata(series_id="in_cpi"), [obs(Q1, 4.2, MAY)])
    )
    assert warehouse.count_observations() == 4
    assert warehouse.count_observations("in_cpi") == 1


def test_series_are_isolated_from_each_other(warehouse, gdp):
    warehouse.upsert_series(gdp)
    warehouse.upsert_series(
        TimeSeries.from_observations(metadata(series_id="in_cpi"), [obs(Q1, 4.2, MAY)])
    )
    assert values(warehouse.latest("in_cpi")) == {Q1: 4.2}


# --- deletion --------------------------------------------------------------

def test_delete_removes_the_series_and_its_data(warehouse, gdp):
    warehouse.upsert_series(gdp)
    warehouse.delete_series("in_gdp")
    assert warehouse.has_series("in_gdp") is False
    assert warehouse.count_observations("in_gdp") == 0


def test_delete_leaves_other_series_intact(warehouse, gdp):
    warehouse.upsert_series(gdp)
    warehouse.upsert_series(
        TimeSeries.from_observations(metadata(series_id="in_cpi"), [obs(Q1, 4.2, MAY)])
    )
    warehouse.delete_series("in_gdp")
    assert warehouse.list_series() == ("in_cpi",)


# --- persistence -----------------------------------------------------------

def test_data_survives_reopening(tmp_path: Path, gdp):
    path = tmp_path / "persist.duckdb"
    with Warehouse(path) as store:
        store.upsert_series(gdp)

    with Warehouse(path) as reopened:
        assert values(reopened.latest("in_gdp")) == {Q1: 7.4, Q2: 6.5}


def test_read_only_mode_rejects_writes(tmp_path: Path, gdp):
    path = tmp_path / "ro.duckdb"
    with Warehouse(path) as store:
        store.upsert_series(gdp)

    with Warehouse(path, read_only=True) as reader:
        assert len(reader.get_series("in_gdp")) == 3
        with pytest.raises(DataFabricError, match="read-only"):
            reader.upsert_series(gdp)


# --- transactions ----------------------------------------------------------

def test_a_failed_transaction_leaves_no_partial_data(warehouse, gdp):
    warehouse.upsert_series(gdp)
    with pytest.raises(RuntimeError):  # noqa: SIM117
        with warehouse.transaction():
            warehouse.query("DELETE FROM observations WHERE series_id = 'in_gdp'")
            raise RuntimeError("simulated failure mid-write")
    assert warehouse.count_observations("in_gdp") == 3


# --- ad-hoc SQL ------------------------------------------------------------

def test_query_escape_hatch(warehouse, gdp):
    warehouse.upsert_series(gdp)
    rows = warehouse.query(
        "SELECT count(*) FROM observations WHERE series_id = ?", ["in_gdp"]
    )
    assert rows[0][0] == 3


def test_repr_shows_the_path(warehouse):
    assert ":memory:" in repr(warehouse)