"""Tests for the canonical time series model.

These tests are mostly about invariants: things that must be true of every
TimeSeries in Moirai, regardless of which source produced it. If an invariant
can be violated, some layer above will silently compute the wrong answer.
"""

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)

APRIL = datetime(2025, 4, 12, tzinfo=UTC)
MAY = datetime(2025, 5, 12, tzinfo=UTC)


def make_metadata(**overrides) -> SeriesMetadata:
    defaults = dict(
        series_id="in_cpi",
        title="India Consumer Price Index",
        source=Source.FRED,
        source_series_id="INDCPIALLMINMEI",
        frequency=Frequency.MONTHLY,
    )
    return SeriesMetadata(**{**defaults, **overrides})


def obs(period: date, value: float | None, known_at: datetime, revision: int = 0) -> Observation:
    return Observation(period=period, value=value, known_at=known_at, revision=revision)


# --- Frequency -------------------------------------------------------------

def test_frequency_rank_orders_coarse_to_fine():
    assert Frequency.ANNUAL.rank < Frequency.QUARTERLY.rank < Frequency.DAILY.rank


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        (Frequency.ANNUAL, 1.0),
        (Frequency.QUARTERLY, 4.0),
        (Frequency.MONTHLY, 12.0),
    ],
)
def test_periods_per_year(frequency, expected):
    assert frequency.periods_per_year == expected


def test_every_frequency_has_a_rank_and_a_period_count():
    """Guard: adding a Frequency member without updating both maps fails here."""
    for member in Frequency:
        assert isinstance(member.rank, int)
        assert member.periods_per_year > 0


# --- Observation -----------------------------------------------------------

def test_observation_carries_both_time_axes():
    o = obs(date(2025, 3, 1), 4.2, APRIL)
    assert o.period == date(2025, 3, 1)
    assert o.known_at == APRIL


def test_observation_is_frozen():
    o = obs(date(2025, 3, 1), 4.2, APRIL)
    with pytest.raises(ValidationError):
        o.value = 9.9  # type: ignore[misc]


def test_missing_value_is_representable():
    o = obs(date(2025, 3, 1), None, APRIL)
    assert o.is_missing is True
    assert o.value is None


def test_present_value_is_not_missing():
    assert obs(date(2025, 3, 1), 0.0, APRIL).is_missing is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(bad):
    """NaN in the fabric propagates silently through every downstream model."""
    with pytest.raises(ValidationError):
        obs(date(2025, 3, 1), bad, APRIL)


def test_negative_revision_is_rejected():
    with pytest.raises(ValidationError):
        obs(date(2025, 3, 1), 4.2, APRIL, revision=-1)


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Observation(period=date(2025, 3, 1), value=1.0, known_at=APRIL, typo=True)


# --- SeriesMetadata --------------------------------------------------------

def test_metadata_defaults_are_explicit_unknowns():
    m = make_metadata()
    assert m.unit is Unit.UNKNOWN
    assert m.seasonal_adjustment is SeasonalAdjustment.UNKNOWN
    assert m.geography == ""


@pytest.mark.parametrize("valid", ["in_cpi", "IN.CPI.2011", "gdp-real", "a1"])
def test_valid_series_ids_are_accepted(valid):
    assert make_metadata(series_id=valid).series_id == valid


@pytest.mark.parametrize(
    "invalid", ["", "   ", "in cpi", "in/cpi", "in\\cpi", "cpi:2011", "../escape"]
)
def test_unsafe_series_ids_are_rejected(invalid):
    """series_id becomes a filename and a table name."""
    with pytest.raises(ValidationError):
        make_metadata(series_id=invalid)


def test_series_id_is_stripped():
    assert make_metadata(series_id="  in_cpi  ").series_id == "in_cpi"


def test_coverage_must_be_ordered():
    with pytest.raises(ValidationError):
        make_metadata(start=date(2025, 1, 1), end=date(2024, 1, 1))


def test_equal_start_and_end_is_allowed():
    m = make_metadata(start=date(2025, 1, 1), end=date(2025, 1, 1))
    assert m.start == m.end


def test_metadata_is_frozen():
    m = make_metadata()
    with pytest.raises(ValidationError):
        m.title = "changed"  # type: ignore[misc]


# --- TimeSeries invariants -------------------------------------------------

def test_empty_series_is_valid():
    ts = TimeSeries(metadata=make_metadata())
    assert ts.is_empty is True
    assert len(ts) == 0
    assert ts.periods == ()


def test_from_observations_sorts_unordered_input():
    """Adapters return data in arbitrary order; sorting happens at the boundary."""
    ts = TimeSeries.from_observations(
        make_metadata(),
        [
            obs(date(2025, 3, 1), 4.2, APRIL),
            obs(date(2025, 1, 1), 4.0, APRIL),
            obs(date(2025, 2, 1), 4.1, APRIL),
        ],
    )
    assert [o.period.month for o in ts.observations] == [1, 2, 3]


def test_unsorted_observations_are_rejected_by_the_constructor():
    with pytest.raises(ValidationError, match="sorted"):
        TimeSeries(
            metadata=make_metadata(),
            observations=(
                obs(date(2025, 3, 1), 4.2, APRIL),
                obs(date(2025, 1, 1), 4.0, APRIL),
            ),
        )


def test_duplicate_observations_are_rejected():
    """Same period, same known_at, same revision cannot mean two values."""
    with pytest.raises(ValidationError, match="duplicate"):
        TimeSeries.from_observations(
            make_metadata(),
            [
                obs(date(2025, 3, 1), 4.2, APRIL),
                obs(date(2025, 3, 1), 4.4, APRIL),
            ],
        )


def test_same_period_with_different_known_at_is_allowed():
    """This is the whole point of bitemporality: revisions coexist."""
    ts = TimeSeries.from_observations(
        make_metadata(),
        [
            obs(date(2025, 3, 1), 4.2, APRIL),
            obs(date(2025, 3, 1), 4.4, MAY, revision=1),
        ],
    )
    assert len(ts) == 2
    assert ts.periods == (date(2025, 3, 1),)


def test_time_series_is_frozen():
    ts = TimeSeries(metadata=make_metadata())
    with pytest.raises(ValidationError):
        ts.observations = ()  # type: ignore[misc]


# --- derived properties ----------------------------------------------------

def test_has_revisions_is_false_for_single_vintage_data():
    ts = TimeSeries.from_observations(
        make_metadata(),
        [obs(date(2025, 1, 1), 4.0, APRIL), obs(date(2025, 2, 1), 4.1, APRIL)],
    )
    assert ts.has_revisions is False


def test_has_revisions_is_true_when_a_period_repeats():
    ts = TimeSeries.from_observations(
        make_metadata(),
        [
            obs(date(2025, 3, 1), 4.2, APRIL),
            obs(date(2025, 3, 1), 4.4, MAY, revision=1),
        ],
    )
    assert ts.has_revisions is True


def test_periods_are_distinct_and_ordered():
    ts = TimeSeries.from_observations(
        make_metadata(),
        [
            obs(date(2025, 2, 1), 4.1, APRIL),
            obs(date(2025, 1, 1), 4.0, APRIL),
            obs(date(2025, 1, 1), 4.05, MAY, revision=1),
        ],
    )
    assert ts.periods == (date(2025, 1, 1), date(2025, 2, 1))


def test_series_id_is_exposed_from_metadata():
    assert TimeSeries(metadata=make_metadata()).series_id == "in_cpi"


# --- records ---------------------------------------------------------------

def test_to_records_shape():
    ts = TimeSeries.from_observations(
        make_metadata(), [obs(date(2025, 3, 1), 4.2, APRIL)]
    )
    (record,) = ts.to_records()
    assert record == {
        "series_id": "in_cpi",
        "period": date(2025, 3, 1),
        "value": 4.2,
        "known_at": APRIL,
        "revision": 0,
    }


def test_to_records_preserves_missing_values():
    ts = TimeSeries.from_observations(
        make_metadata(), [obs(date(2025, 3, 1), None, APRIL)]
    )
    assert ts.to_records()[0]["value"] is None


def test_to_records_is_empty_for_an_empty_series():
    assert TimeSeries(metadata=make_metadata()).to_records() == []