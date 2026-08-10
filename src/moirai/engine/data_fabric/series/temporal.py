"""
Bitemporal query operations over a TimeSeries.

A TimeSeries can hold several vintages of the same period. These functions
collapse that two-dimensional structure into the one-dimensional view a
given question needs.

    as_of(ts, when)         what we knew on `when`      -> backtesting
    latest(ts)              current best estimate        -> reporting
    revision_history(ts, p) how one figure evolved       -> revision analysis
    first_release(ts)       initial prints only          -> nowcasting

The distinction that matters:

    latest()   uses today's revised numbers. Fine for description.
    as_of(t)   uses only what existed at t. Required for evaluation.

Scoring a June 2025 forecast against today's revised GDP is lookahead bias:
the model is credited with information it did not have. `as_of` is the
mechanism that makes that error impossible rather than merely discouraged.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from moirai.core.exceptions import VintageError
from moirai.engine.data_fabric.series.models import (
    Observation,
    SeriesMetadata,
    TimeSeries,
)


def _require_aware(when: datetime, argument: str) -> datetime:
    """Reject naive datetimes.

    A naive timestamp is ambiguous across timezones, and vintage boundaries
    are exactly where an hour of ambiguity changes which data you see.
    """
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise VintageError(f"{argument} must be timezone-aware, got {when!r}")
    return when


def _rebuild(metadata: SeriesMetadata, observations: Iterable[Observation]) -> TimeSeries:
    return TimeSeries.from_observations(metadata, list(observations))


def as_of(series: TimeSeries, when: datetime) -> TimeSeries:
    """Return the series as it was known at `when`.

    For each period, selects the most recent observation with
    `known_at <= when`. Periods first published after `when` are absent,
    exactly as they would have been at the time.

    Raises
    ------
    VintageError
        If `when` is naive.
    """
    _require_aware(when, "when")

    newest: dict[date, Observation] = {}
    for observation in series.observations:
        if observation.known_at > when:
            continue
        current = newest.get(observation.period)
        if current is None or (observation.known_at, observation.revision) > (
            current.known_at,
            current.revision,
        ):
            newest[observation.period] = observation

    return _rebuild(series.metadata, newest.values())


def latest(series: TimeSeries) -> TimeSeries:
    """Return the most recent vintage of each period: today's best estimate."""
    newest: dict[date, Observation] = {}
    for observation in series.observations:
        current = newest.get(observation.period)
        if current is None or (observation.known_at, observation.revision) > (
            current.known_at,
            current.revision,
        ):
            newest[observation.period] = observation
    return _rebuild(series.metadata, newest.values())


def first_release(series: TimeSeries) -> TimeSeries:
    """Return the initial publication of each period.

    Real-time nowcasting research is evaluated against first releases,
    because that is what a forecaster actually saw.
    """
    earliest: dict[date, Observation] = {}
    for observation in series.observations:
        current = earliest.get(observation.period)
        if current is None or (observation.known_at, observation.revision) < (
            current.known_at,
            current.revision,
        ):
            earliest[observation.period] = observation
    return _rebuild(series.metadata, earliest.values())


def revision_history(series: TimeSeries, period: date) -> tuple[Observation, ...]:
    """Every vintage of one period, oldest first.

    Raises
    ------
    VintageError
        If the period is not covered by the series.
    """
    matches = tuple(o for o in series.observations if o.period == period)
    if not matches:
        raise VintageError(
            f"period {period} is not present in series {series.metadata.series_id!r}"
        )
    return matches


def vintage_dates(series: TimeSeries) -> tuple[datetime, ...]:
    """Distinct `known_at` timestamps, ascending.

    These are the only dates at which the series changed, so they are the
    only interesting points to run an as-of comparison.
    """
    return tuple(sorted({o.known_at for o in series.observations}))


def total_revision(series: TimeSeries, period: date) -> float | None:
    """Latest value minus first release for one period.

    Returns None if either end is missing, since a revision from an
    unpublished value is not a number.
    """
    history = revision_history(series, period)
    initial, final = history[0].value, history[-1].value
    if initial is None or final is None:
        return None
    return final - initial


def restrict(
    series: TimeSeries,
    *,
    start: date | None = None,
    end: date | None = None,
) -> TimeSeries:
    """Restrict to a closed period window, preserving every vintage."""
    if start is not None and end is not None and start > end:
        raise VintageError(f"start {start} is after end {end}")

    kept = [
        o
        for o in series.observations
        if (start is None or o.period >= start) and (end is None or o.period <= end)
    ]
    return _rebuild(series.metadata, kept)


def drop_missing(series: TimeSeries) -> TimeSeries:
    """Remove observations with no value.

    Deliberately explicit rather than automatic: a gap is information, and
    discarding it should be a decision the caller makes and can be seen
    making in a diff.
    """
    return _rebuild(series.metadata, [o for o in series.observations if not o.is_missing])