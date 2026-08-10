"""Tests for causal data preparation.

Synthetic series with known properties are used throughout: a random walk
is non-stationary by construction, white noise is stationary by
construction. Testing against real data would test the data, not the code.
"""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.preparation import (
    MIN_OBSERVATIONS,
    PreparedSeries,
    StationarityVerdict,
    Transformation,
    align,
    apply_transformation,
    check_stationarity,
    drop_leading_and_trailing_nans,
    prepare,
    suggest_transformation,
    to_array,
)
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)

KNOWN_AT = datetime(2026, 1, 1, tzinfo=UTC)
RNG = np.random.default_rng(42)


def monthly_periods(n: int, start_year: int = 2000) -> list[date]:
    return [date(start_year + i // 12, i % 12 + 1, 1) for i in range(n)]


def make_series(values, *, unit=Unit.UNKNOWN, series_id="test_series") -> TimeSeries:
    periods = monthly_periods(len(values))
    metadata = SeriesMetadata(
        series_id=series_id,
        title="Test Series",
        source=Source.MANUAL,
        source_series_id="TEST",
        frequency=Frequency.MONTHLY,
        unit=unit,
    )
    observations = [
        Observation(
            period=period,
            value=None if value is None else float(value),
            known_at=KNOWN_AT,
        )
        for period, value in zip(periods, values, strict=True)
    ]
    return TimeSeries.from_observations(metadata, observations)


def white_noise(n: int = 200, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, n)


def random_walk(n: int = 200, seed: int = 1) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, n)) + 100


def trending_index(n: int = 200, seed: int = 1) -> np.ndarray:
    """A price index: multiplicative growth plus noise, strictly positive."""
    shocks = np.random.default_rng(seed).normal(0.004, 0.006, n)
    return 100 * np.exp(np.cumsum(shocks))


# --- to_array --------------------------------------------------------------

def test_to_array_extracts_periods_and_values():
    series = make_series([1.0, 2.0, 3.0])
    periods, values = to_array(series)
    assert len(periods) == 3
    assert values.tolist() == [1.0, 2.0, 3.0]


def test_to_array_maps_missing_values_to_nan():
    _, values = to_array(make_series([1.0, None, 3.0]))
    assert np.isnan(values[1])


def test_to_array_refuses_multiple_vintages():
    """Collapsing silently would hide which vintage produced a result."""
    metadata = SeriesMetadata(
        series_id="revised",
        title="Revised",
        source=Source.FRED,
        source_series_id="X",
        frequency=Frequency.QUARTERLY,
    )
    series = TimeSeries.from_observations(
        metadata,
        [
            Observation(period=date(2025, 1, 1), value=1.0, known_at=KNOWN_AT),
            Observation(
                period=date(2025, 1, 1),
                value=1.2,
                known_at=datetime(2026, 6, 1, tzinfo=UTC),
                revision=1,
            ),
        ],
    )
    with pytest.raises(EngineError, match="multiple vintages"):
        to_array(series)


# --- trimming --------------------------------------------------------------

def test_leading_and_trailing_nans_are_trimmed():
    periods = monthly_periods(5)
    values = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    kept_periods, kept_values = drop_leading_and_trailing_nans(periods, values)
    assert len(kept_periods) == 3
    assert kept_values.tolist() == [1.0, 2.0, 3.0]


def test_interior_nans_are_preserved_by_trimming():
    """Only the edges are trimmed; an interior gap must survive to be caught."""
    periods = monthly_periods(3)
    values = np.array([1.0, np.nan, 3.0])
    _, kept = drop_leading_and_trailing_nans(periods, values)
    assert np.isnan(kept[1])


def test_all_nan_input_yields_empty_output():
    periods, values = drop_leading_and_trailing_nans(
        monthly_periods(3), np.array([np.nan] * 3)
    )
    assert periods == () and values.size == 0


# --- stationarity ----------------------------------------------------------

def test_white_noise_is_stationary():
    result = check_stationarity("wn", white_noise())
    assert result.verdict is StationarityVerdict.STATIONARY
    assert result.is_stationary is True


def test_random_walk_is_non_stationary():
    result = check_stationarity("rw", random_walk())
    assert result.verdict is StationarityVerdict.NON_STATIONARY


def test_differenced_random_walk_becomes_stationary():
    differenced = np.diff(random_walk())
    assert check_stationarity("drw", differenced).is_stationary is True


def test_adf_and_kpss_have_opposite_conventions():
    """Guard against inverting a test: they disagree by design."""
    result = check_stationarity("wn", white_noise())
    assert result.adf_says_stationary is True   # low ADF p
    assert result.kpss_says_stationary is True  # high KPSS p


def test_disagreement_is_reported_not_resolved():
    result = check_stationarity("wn", white_noise())
    forced = result.model_copy(update={"adf_pvalue": 0.9, "kpss_pvalue": 0.9})
    assert forced.verdict is StationarityVerdict.INCONCLUSIVE
    assert forced.is_stationary is False


def test_short_series_is_rejected():
    with pytest.raises(EngineError, match="at least"):
        check_stationarity("short", white_noise(MIN_OBSERVATIONS - 1))


def test_constant_series_is_rejected():
    with pytest.raises(EngineError, match="constant"):
        check_stationarity("flat", np.full(50, 3.0))


def test_significance_level_is_configurable():
    result = check_stationarity("wn", white_noise(), significance=0.01)
    assert result.significance == 0.01


def test_result_serialises_for_the_ledger():
    payload = check_stationarity("wn", white_noise()).to_ledger_dict()
    assert payload["verdict"] == "stationary"
    assert "adf_pvalue" in payload


# --- transformations -------------------------------------------------------

def test_none_transformation_is_identity():
    values = np.array([1.0, 2.0, 3.0])
    result, consumed = apply_transformation(values, Transformation.NONE)
    assert result.tolist() == [1.0, 2.0, 3.0]
    assert consumed == 0


def test_difference_consumes_one_observation():
    result, consumed = apply_transformation(
        np.array([1.0, 3.0, 6.0]), Transformation.DIFFERENCE
    )
    assert result.tolist() == [2.0, 3.0]
    assert consumed == 1


def test_log_difference_approximates_growth_rate():
    values = np.array([100.0, 110.0])
    result, _ = apply_transformation(values, Transformation.LOG_DIFFERENCE)
    assert result[0] == pytest.approx(np.log(1.1))


def test_percent_change_is_in_percent_units():
    result, _ = apply_transformation(
        np.array([100.0, 110.0]), Transformation.PERCENT_CHANGE
    )
    assert result[0] == pytest.approx(10.0)


def test_demean_removes_the_mean():
    result, _ = apply_transformation(np.array([1.0, 2.0, 3.0]), Transformation.DEMEAN)
    assert result.mean() == pytest.approx(0.0)


def test_standardise_gives_unit_variance():
    result, _ = apply_transformation(white_noise(100), Transformation.STANDARDISE)
    assert np.std(result, ddof=1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "transformation", [Transformation.LOG, Transformation.LOG_DIFFERENCE]
)
def test_log_rejects_non_positive_values(transformation):
    with pytest.raises(EngineError, match="positive"):
        apply_transformation(np.array([1.0, 0.0, 2.0]), transformation)


def test_percent_change_rejects_zero_denominators():
    with pytest.raises(EngineError, match="non-zero"):
        apply_transformation(np.array([0.0, 1.0]), Transformation.PERCENT_CHANGE)


def test_standardise_rejects_a_constant_series():
    with pytest.raises(EngineError, match="constant"):
        apply_transformation(np.full(10, 5.0), Transformation.STANDARDISE)


# --- suggestion logic ------------------------------------------------------

def test_percent_change_series_is_left_alone():
    """Metadata is authoritative: a rate must not be differenced again."""
    series = make_series(random_walk(), unit=Unit.PERCENT_CHANGE)
    result = check_stationarity("x", random_walk())
    assert suggest_transformation(series, result) is Transformation.NONE


def test_stationary_series_needs_no_transformation():
    series = make_series(white_noise(), unit=Unit.PERCENT)
    result = check_stationarity("x", white_noise())
    assert suggest_transformation(series, result) is Transformation.NONE


@pytest.mark.parametrize("unit", [Unit.INDEX, Unit.CURRENCY, Unit.COUNT])
def test_multiplicative_levels_get_log_differences(unit):
    series = make_series(trending_index(), unit=unit)
    result = check_stationarity("x", random_walk())
    assert suggest_transformation(series, result) is Transformation.LOG_DIFFERENCE


@pytest.mark.parametrize("unit", [Unit.PERCENT, Unit.RATIO, Unit.UNKNOWN])
def test_scale_free_series_get_plain_differences(unit):
    series = make_series(random_walk(), unit=unit)
    result = check_stationarity("x", random_walk())
    assert suggest_transformation(series, result) is Transformation.DIFFERENCE


# --- prepare ---------------------------------------------------------------

def test_prepare_transforms_a_random_walk():
    result = prepare(make_series(random_walk(), unit=Unit.PERCENT))
    assert result.record.steps == (Transformation.DIFFERENCE,)
    assert result.stationarity is not None
    assert result.stationarity.is_stationary is True


def test_prepare_leaves_stationary_data_alone():
    result = prepare(make_series(white_noise(), unit=Unit.PERCENT))
    assert result.record.steps == (Transformation.NONE,)
    assert len(result) == 200


def test_prepare_records_observations_lost():
    result = prepare(make_series(random_walk(), unit=Unit.PERCENT))
    assert result.record.observations_lost == 1


def test_prepare_keeps_periods_aligned_with_values():
    result = prepare(make_series(random_walk(), unit=Unit.PERCENT))
    assert len(result.periods) == len(result.values)


def test_explicit_transformation_overrides_the_suggestion():
    result = prepare(
        make_series(white_noise(), unit=Unit.PERCENT),
        transformation=Transformation.STANDARDISE,
    )
    assert result.record.steps == (Transformation.STANDARDISE,)
    assert "explicit" in result.record.reason


def test_prepare_records_the_original_unit():
    result = prepare(make_series(random_walk(), unit=Unit.INDEX))
    assert result.record.original_unit is Unit.INDEX


def test_prepare_rejects_interior_gaps():
    values = list(random_walk(60))
    values[30] = None
    with pytest.raises(EngineError, match="interior gaps"):
        prepare(make_series(values, unit=Unit.PERCENT))


def test_prepare_can_skip_verification():
    result = prepare(make_series(random_walk(), unit=Unit.PERCENT), verify=False)
    assert result.stationarity is None


def test_record_serialises_for_the_ledger():
    payload = prepare(make_series(random_walk(), unit=Unit.PERCENT)).record.to_ledger_dict()
    assert payload["steps"] == ["difference"]
    assert payload["original_unit"] == "percent"


# --- align -----------------------------------------------------------------

def _prepared(series_id: str, values, offset: int = 0) -> PreparedSeries:
    periods = monthly_periods(len(values) + offset)[offset:]
    from moirai.engine.causal.preparation import TransformationRecord

    return PreparedSeries(
        series_id=series_id,
        periods=tuple(periods),
        values=np.asarray(values, dtype=float),
        record=TransformationRecord(
            series_id=series_id,
            steps=(Transformation.NONE,),
            reason="test",
            observations_before=len(values),
            observations_after=len(values),
            original_unit=Unit.UNKNOWN,
        ),
    )


def test_align_produces_a_matrix_with_one_column_per_series():
    periods, matrix = align([_prepared("a", [1, 2, 3]), _prepared("b", [4, 5, 6])])
    assert matrix.shape == (3, 2)
    assert len(periods) == 3


def test_align_preserves_column_order():
    _, matrix = align([_prepared("a", [1, 2, 3]), _prepared("b", [4, 5, 6])])
    assert matrix[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert matrix[:, 1].tolist() == [4.0, 5.0, 6.0]


def test_align_intersects_differing_samples():
    """A VAR needs every variable observed at every date."""
    periods, matrix = align(
        [_prepared("a", [1, 2, 3, 4]), _prepared("b", [5, 6, 7], offset=1)]
    )
    assert matrix.shape == (3, 2)
    assert periods[0] == monthly_periods(4)[1]


def test_align_output_is_sorted_by_period():
    periods, _ = align([_prepared("a", [1, 2, 3])])
    assert list(periods) == sorted(periods)


def test_align_rejects_an_empty_input():
    with pytest.raises(EngineError, match="at least one"):
        align([])


def test_align_rejects_disjoint_samples():
    with pytest.raises(EngineError, match="no common periods"):
        align([_prepared("a", [1, 2, 3]), _prepared("b", [4, 5, 6], offset=50)])