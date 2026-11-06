"""Stationarity tests: the KPSS table edge, and the level test kept on record.

Written before the fix. Two findings from the study of preparation.py:

  * KPSS p-values are interpolated from a table that runs from 0.01 to 0.10,
    so statsmodels clips anything outside it. A reported 0.10 means "0.10 or
    more", but the verdict compared it with `p > significance`, so at a 10%
    significance level a clearly stationary series failed KPSS. A level above
    0.10 or below 0.01 cannot be judged at all, because the table does not
    reach it, and was accepted anyway.

  * prepare() tests the series before transforming it and chooses the
    transformation from that test, then discards the test. The record kept
    the verdict as text but not the statistics, so the choice could not be
    audited from the record.
"""

from datetime import UTC, date, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from moirai.engine.causal.preparation import StationarityResult, Transformation, prepare
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)

KNOWN = datetime(2025, 1, 1, tzinfo=UTC)


def result(
    kpss_pvalue: float, significance: float, adf_pvalue: float = 0.01
) -> StationarityResult:
    return StationarityResult(
        series_id="x",
        n_observations=200,
        adf_statistic=-4.0,
        adf_pvalue=adf_pvalue,
        kpss_statistic=0.1,
        kpss_pvalue=kpss_pvalue,
        significance=significance,
    )


# ---- the KPSS table edge -------------------------------------------------------


def test_a_clipped_kpss_pvalue_counts_as_stationary_at_ten_percent() -> None:
    """0.10 is the top of the table: it means 0.10 or more."""
    assert result(kpss_pvalue=0.10, significance=0.10).kpss_says_stationary


@pytest.mark.parametrize("significance", [0.15, 0.20, 0.005])
def test_a_significance_level_the_kpss_table_cannot_reach_is_rejected(
    significance: float,
) -> None:
    with pytest.raises(ValidationError):
        result(kpss_pvalue=0.05, significance=significance)


@pytest.mark.parametrize(
    ("kpss_pvalue", "significance", "stationary"),
    [
        (0.10, 0.05, True),  # top of the table, usual level
        (0.04, 0.05, False),
        (0.01, 0.01, False),  # bottom of the table means 0.01 or less
        (0.06, 0.05, True),
    ],
)
def test_kpss_verdicts_away_from_the_edge_are_unchanged(
    kpss_pvalue: float, significance: float, stationary: bool
) -> None:
    assert result(kpss_pvalue, significance).kpss_says_stationary is stationary


# ---- the level test is kept ----------------------------------------------------


def random_walk(n: int = 240, seed: int = 7) -> TimeSeries:
    """A price index that wanders: non-stationary in levels, so it is differenced."""
    rng = np.random.default_rng(seed)
    levels = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n)))
    periods = [date(2000 + i // 12, i % 12 + 1, 1) for i in range(n)]
    metadata = SeriesMetadata(
        series_id="walk",
        title="Random walk index",
        source=Source.MANUAL,
        source_series_id="WALK",
        frequency=Frequency.MONTHLY,
        unit=Unit.INDEX,
        unit_label="Index",
        seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
        geography="IN",
        start=periods[0],
        end=periods[-1],
    )
    observations = [
        Observation(period=p, value=float(v), known_at=KNOWN)
        for p, v in zip(periods, levels, strict=True)
    ]
    return TimeSeries.from_observations(metadata, observations)


def test_prepare_keeps_the_test_that_chose_the_transformation() -> None:
    prepared = prepare(random_walk())

    assert prepared.record.steps == (Transformation.LOG_DIFFERENCE,)
    initial = prepared.initial_stationarity
    assert initial is not None
    assert initial.n_observations == 240  # the untransformed series
    assert not initial.is_stationary  # which is why it was differenced


def test_the_level_test_and_the_final_test_are_both_on_record() -> None:
    prepared = prepare(random_walk())

    assert prepared.initial_stationarity is not None
    assert prepared.stationarity is not None
    assert prepared.stationarity.n_observations == 239  # one lost to differencing
