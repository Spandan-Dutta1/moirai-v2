"""Tests for bitemporal query operations.

The scenario used throughout mirrors a real Indian GDP release pattern:
Q1 is first published in May, revised in August; Q2 arrives in August.
That is enough structure to distinguish every operation in this module.
"""

from datetime import UTC, date, datetime

import pytest

from moirai.core.exceptions import VintageError
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeriesMetadata,
    Source,
    TimeSeries,
)
from moirai.engine.data_fabric.series.temporal import (
    as_of,
    drop_missing,
    first_release,
    latest,
    restrict,
    revision_history,
    total_revision,
    vintage_dates,
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
    )
    return SeriesMetadata(**{**defaults, **overrides})


def obs(period, value, known_at, revision=0) -> Observation:
    return Observation(period=period, value=value, known_at=known_at, revision=revision)


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


@pytest.fixture
def empty() -> TimeSeries:
    return TimeSeries(metadata=metadata())


def values(series: TimeSeries) -> dict:
    return {o.period: o.value for o in series.observations}


# --- as_of -----------------------------------------------------------------

def test_as_of_returns_the_vintage_available_at_that_time(gdp):
    assert values(as_of(gdp, JUNE)) == {Q1: 7.8}


def test_as_of_excludes_periods_published_later(gdp):
    """Q2 did not exist in June. It must not appear."""
    assert Q2 not in values(as_of(gdp, JUNE))


def test_as_of_after_all_revisions_matches_latest(gdp):
    assert values(as_of(gdp, NOVEMBER)) == values(latest(gdp))


def test_as_of_before_any_publication_is_empty(gdp):
    assert as_of(gdp, datetime(2025, 1, 1, tzinfo=UTC)).is_empty


def test_as_of_boundary_is_inclusive(gdp):
    """An observation known_at exactly `when` was known at `when`."""
    assert values(as_of(gdp, MAY)) == {Q1: 7.8}


def test_as_of_picks_the_revision_when_both_are_visible(gdp):
    assert values(as_of(gdp, AUGUST))[Q1] == 7.4


def test_as_of_rejects_naive_datetimes(gdp):
    """Timezone ambiguity at a vintage boundary changes which data you see."""
    with pytest.raises(VintageError, match="timezone-aware"):
        as_of(gdp, datetime(2025, 6, 15))


def test_as_of_result_has_one_observation_per_period(gdp):
    result = as_of(gdp, NOVEMBER)
    assert len(result) == len(result.periods)


def test_as_of_preserves_metadata(gdp):
    assert as_of(gdp, JUNE).metadata == gdp.metadata


def test_as_of_on_empty_series(empty):
    assert as_of(empty, JUNE).is_empty


# --- latest ----------------------------------------------------------------

def test_latest_uses_the_most_recent_vintage(gdp):
    assert values(latest(gdp)) == {Q1: 7.4, Q2: 6.5}


def test_latest_collapses_revisions(gdp):
    assert latest(gdp).has_revisions is False


def test_latest_on_empty_series(empty):
    assert latest(empty).is_empty


# --- first_release ---------------------------------------------------------

def test_first_release_uses_the_initial_publication(gdp):
    assert values(first_release(gdp)) == {Q1: 7.8, Q2: 6.5}


def test_first_release_differs_from_latest_when_revised(gdp):
    assert values(first_release(gdp))[Q1] != values(latest(gdp))[Q1]


def test_first_release_on_empty_series(empty):
    assert first_release(empty).is_empty


# --- revision_history ------------------------------------------------------

def test_revision_history_is_ordered_oldest_first(gdp):
    history = revision_history(gdp, Q1)
    assert [o.value for o in history] == [7.8, 7.4]


def test_revision_history_for_an_unrevised_period(gdp):
    assert len(revision_history(gdp, Q2)) == 1


def test_revision_history_rejects_an_unknown_period(gdp):
    with pytest.raises(VintageError, match="not present"):
        revision_history(gdp, date(2024, 1, 1))


# --- total_revision --------------------------------------------------------

def test_total_revision_is_latest_minus_first(gdp):
    assert total_revision(gdp, Q1) == pytest.approx(-0.4)


def test_total_revision_is_zero_for_unrevised_periods(gdp):
    assert total_revision(gdp, Q2) == pytest.approx(0.0)


def test_total_revision_is_none_when_a_value_is_missing():
    series = TimeSeries.from_observations(
        metadata(),
        [obs(Q1, None, MAY), obs(Q1, 7.4, AUGUST, revision=1)],
    )
    assert total_revision(series, Q1) is None


# --- vintage_dates ---------------------------------------------------------

def test_vintage_dates_are_distinct_and_ascending(gdp):
    assert vintage_dates(gdp) == (MAY, AUGUST)


def test_vintage_dates_on_empty_series(empty):
    assert vintage_dates(empty) == ()


# --- restrict --------------------------------------------------------------

def test_restrict_keeps_only_the_window(gdp):
    assert restrict(gdp, start=Q2).periods == (Q2,)


def test_restrict_preserves_every_vintage_in_the_window(gdp):
    """Restricting periods must not silently collapse revisions."""
    assert len(restrict(gdp, end=Q1)) == 2


def test_restrict_bounds_are_inclusive(gdp):
    assert restrict(gdp, start=Q1, end=Q1).periods == (Q1,)


def test_restrict_with_no_bounds_is_a_no_op(gdp):
    assert restrict(gdp).observations == gdp.observations


def test_restrict_rejects_an_inverted_window(gdp):
    with pytest.raises(VintageError, match="after"):
        restrict(gdp, start=Q2, end=Q1)


def test_restrict_to_an_empty_window(gdp):
    assert restrict(gdp, start=date(2030, 1, 1)).is_empty


# --- drop_missing ----------------------------------------------------------

def test_drop_missing_removes_gaps():
    series = TimeSeries.from_observations(
        metadata(), [obs(Q1, 7.8, MAY), obs(Q2, None, AUGUST)]
    )
    assert drop_missing(series).periods == (Q1,)


def test_drop_missing_is_a_no_op_without_gaps(gdp):
    assert drop_missing(gdp).observations == gdp.observations


# --- composition -----------------------------------------------------------

def test_as_of_then_restrict_composes(gdp):
    result = restrict(as_of(gdp, NOVEMBER), start=Q2)
    assert values(result) == {Q2: 6.5}


def test_operations_do_not_mutate_the_input(gdp):
    before = gdp.observations
    as_of(gdp, JUNE)
    latest(gdp)
    first_release(gdp)
    drop_missing(gdp)
    assert gdp.observations == before