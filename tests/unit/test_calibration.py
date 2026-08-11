"""Tests for moment matching.

The point of this module is that a parameter cannot be chosen quietly to
produce a desired result, so the tests are largely about the machinery
that enforces that: targets carry their provenance, failures are reported
rather than suppressed, and the loss is comparable across moments measured
in different units.

Measurements are checked against constructed populations with known
properties rather than against previous output, so a broken measurement
fails rather than merely changing.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.economy.calibration import (
    MEASUREMENTS,
    URBAN_INDIA_MOMENTS,
    CalibrationReport,
    Confidence,
    Moment,
    MomentResult,
    evaluate,
    search,
)
from moirai.engine.economy.households import (
    EmploymentStatus,
    Population,
    PopulationParameters,
    generate_population,
)


def make_population(
    *,
    n: int = 1_000,
    income: float | np.ndarray = 100_000.0,
    wealth: float | np.ndarray = 50_000.0,
    debt: float | np.ndarray = 0.0,
    floating: bool = True,
    employment: int = EmploymentStatus.EMPLOYED,
) -> Population:
    """A population with controlled properties, for checking measurements."""

    def spread(value) -> np.ndarray:
        return np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value, dtype=float)

    return Population(
        age=np.full(n, 40, dtype=np.int16),
        income=spread(income),
        wealth=spread(wealth),
        debt=spread(debt),
        debt_is_floating=np.full(n, floating, dtype=bool),
        employment=np.full(n, employment, dtype=np.int8),
        expected_inflation=np.full(n, 0.05),
        risk_aversion=np.full(n, 2.0),
        parameters=PopulationParameters(n_households=max(n, 100)),
    )


@pytest.fixture(scope="module")
def generated() -> Population:
    return generate_population(PopulationParameters(n_households=20_000, seed=1))


# --- measurement registry --------------------------------------------------

def test_every_declared_moment_has_a_measurement():
    """A target naming a measurement that does not exist must fail loudly."""
    for moment in URBAN_INDIA_MOMENTS:
        assert moment.name in MEASUREMENTS


def test_measurements_return_finite_scalars(generated):
    for name, measure in MEASUREMENTS.items():
        value = measure(generated)
        assert np.isfinite(value), f"{name} returned {value}"


# --- individual measurements -----------------------------------------------

def test_share_indebted_counts_positive_debt():
    debt = np.zeros(1_000)
    debt[:250] = 100_000.0
    assert MEASUREMENTS["share_indebted"](make_population(debt=debt)) == 0.25


def test_nobody_indebted_measures_zero():
    assert MEASUREMENTS["share_indebted"](make_population(debt=0.0)) == 0.0


def test_hand_to_mouth_counts_households_without_a_buffer():
    """Below half a month of income in liquid wealth."""
    population = make_population(income=120_000.0, wealth=1_000.0)
    assert MEASUREMENTS["share_hand_to_mouth"](population) == 1.0


def test_wealthy_households_are_not_hand_to_mouth():
    population = make_population(income=120_000.0, wealth=500_000.0)
    assert MEASUREMENTS["share_hand_to_mouth"](population) == 0.0


def test_hand_to_mouth_is_relative_to_income():
    """The same wealth is a buffer for a poor household and not for a rich one."""
    poor = make_population(income=12_000.0, wealth=10_000.0)
    rich = make_population(income=12_000_000.0, wealth=10_000.0)
    assert MEASUREMENTS["share_hand_to_mouth"](poor) == 0.0
    assert MEASUREMENTS["share_hand_to_mouth"](rich) == 1.0


def test_negative_wealth_counts_as_hand_to_mouth():
    population = make_population(income=120_000.0, wealth=-50_000.0)
    assert MEASUREMENTS["share_hand_to_mouth"](population) == 1.0


def test_gini_of_an_equal_population_is_zero():
    assert MEASUREMENTS["income_gini"](make_population()) == pytest.approx(0.0, abs=1e-9)


def test_employment_rate_counts_only_the_employed():
    population = make_population(employment=EmploymentStatus.UNEMPLOYED)
    assert MEASUREMENTS["employment_rate"](population) == 0.0


def test_rate_exposure_requires_debt_and_floating():
    with_debt = make_population(debt=100_000.0, floating=True)
    fixed = make_population(debt=100_000.0, floating=False)
    assert MEASUREMENTS["share_rate_exposed"](with_debt) == 1.0
    assert MEASUREMENTS["share_rate_exposed"](fixed) == 0.0


def test_mean_debt_to_mean_income_is_an_aggregate_ratio():
    population = make_population(income=100_000.0, debt=50_000.0)
    assert MEASUREMENTS["mean_debt_to_mean_income"](population) == pytest.approx(0.5)


def test_median_debt_to_income_ignores_the_debt_free():
    debt = np.zeros(1_000)
    debt[:500] = 200_000.0
    population = make_population(income=100_000.0, debt=debt)
    assert MEASUREMENTS["median_debt_to_income_of_borrowers"](population) == pytest.approx(2.0)


def test_indebtedness_gradient_is_infinite_when_the_bottom_never_borrows():
    """Guards the division rather than raising: an infinite gradient is a
    meaningful answer, since it says the bottom quintile has no credit."""
    income = np.linspace(10_000, 1_000_000, 1_000)
    debt = np.where(income > 500_000, 100_000.0, 0.0)
    population = make_population(n=1_000, income=income, debt=debt)
    assert np.isinf(MEASUREMENTS["indebtedness_gradient"](population))


def test_indebtedness_gradient_is_one_when_borrowing_is_uniform():
    """The failure this moment exists to catch: uniform rate exposure."""
    income = np.linspace(10_000, 1_000_000, 1_000)
    population = make_population(n=1_000, income=income, debt=100_000.0)
    assert MEASUREMENTS["indebtedness_gradient"](population) == pytest.approx(1.0)


# --- Moment ----------------------------------------------------------------

def moment(target: float, tolerance: float = 0.05, name: str = "share_indebted") -> Moment:
    return Moment(
        name=name,
        target=target,
        tolerance=tolerance,
        source="test",
        confidence=Confidence.SOURCED,
    )


def test_gap_is_measured_minus_target():
    population = make_population(debt=np.where(np.arange(1_000) < 300, 1.0, 0.0))
    assert moment(0.25).gap(population) == pytest.approx(0.05)


def test_a_gap_within_tolerance_passes():
    population = make_population(debt=np.where(np.arange(1_000) < 300, 1.0, 0.0))
    assert moment(0.25, tolerance=0.1).passes(population) is True


def test_a_gap_beyond_tolerance_fails():
    population = make_population(debt=np.where(np.arange(1_000) < 300, 1.0, 0.0))
    assert moment(0.25, tolerance=0.01).passes(population) is False


def test_tolerance_is_symmetric():
    population = make_population(debt=np.where(np.arange(1_000) < 300, 1.0, 0.0))
    assert moment(0.35, tolerance=0.06).passes(population) is True


def test_moment_is_frozen():
    with pytest.raises(Exception):
        moment(0.25).target = 0.5  # type: ignore[misc]


def test_zero_tolerance_is_rejected():
    with pytest.raises(Exception):
        Moment(
            name="share_indebted",
            target=0.2,
            tolerance=0.0,
            source="test",
            confidence=Confidence.SOURCED,
        )


def test_moment_serialises_with_its_provenance():
    payload = moment(0.25).to_ledger_dict()
    assert payload["confidence"] == "sourced"
    assert payload["source"] == "test"


# --- declared targets ------------------------------------------------------

def test_every_declared_target_names_a_source():
    for m in URBAN_INDIA_MOMENTS:
        assert m.source.strip(), f"{m.name} has no source"


def test_unsourced_targets_say_so():
    """An unsourced placeholder must not read like a measured quantity."""
    for m in URBAN_INDIA_MOMENTS:
        if m.confidence is Confidence.UNSOURCED:
            assert "not verified" in m.source.lower() or "qualitative" in m.source.lower() or m.note


def test_the_indebtedness_target_is_sourced():
    """The one target taken directly from AIDIS."""
    indebted = next(m for m in URBAN_INDIA_MOMENTS if m.name == "share_indebted")
    assert indebted.confidence is Confidence.SOURCED
    assert "AIDIS" in indebted.source


def test_target_names_are_unique():
    names = [m.name for m in URBAN_INDIA_MOMENTS]
    assert len(set(names)) == len(names)


def test_most_targets_are_honestly_marked_unsourced():
    """A reminder in test form: this calibration is provisional."""
    unsourced = sum(1 for m in URBAN_INDIA_MOMENTS if m.confidence is Confidence.UNSOURCED)
    assert unsourced > 0, "if every target is sourced, update this test"


# --- evaluate --------------------------------------------------------------

def test_evaluate_returns_one_result_per_moment(generated):
    report = evaluate(generated)
    assert len(report.results) == len(URBAN_INDIA_MOMENTS)


def test_evaluate_records_the_parameters(generated):
    assert evaluate(generated).parameters == generated.parameters


def test_evaluate_rejects_an_empty_moment_set(generated):
    with pytest.raises(EngineError, match="at least one moment"):
        evaluate(generated, [])


def test_evaluate_rejects_a_population_without_parameters():
    """A report that cannot be attributed to a calibration is not a result."""
    population = make_population()
    population.parameters = None
    with pytest.raises(EngineError, match="no parameters recorded"):
        evaluate(population)


def test_evaluate_is_deterministic(generated):
    assert evaluate(generated).loss() == evaluate(generated).loss()


# --- CalibrationReport -----------------------------------------------------

def build_report(gaps: list[float], tolerance: float = 0.05) -> CalibrationReport:
    results = tuple(
        MomentResult(
            moment=moment(0.5, tolerance=tolerance),
            measured=0.5 + gap,
            gap=gap,
            passed=abs(gap) <= tolerance,
        )
        for gap in gaps
    )
    return CalibrationReport(results=results, parameters=PopulationParameters())


def test_loss_is_zero_when_every_moment_matches():
    assert build_report([0.0, 0.0]).loss() == 0.0


def test_loss_grows_with_the_gap():
    assert build_report([0.1]).loss() > build_report([0.05]).loss()


def test_loss_scales_gaps_by_tolerance():
    """A moment measured in different units must not dominate by scale alone."""
    tight = build_report([0.1], tolerance=0.01)
    loose = build_report([0.1], tolerance=1.0)
    assert tight.loss() > loose.loss()


def test_loss_is_symmetric_in_the_sign_of_the_gap():
    assert build_report([0.1]).loss() == pytest.approx(build_report([-0.1]).loss())


def test_passed_count_matches_the_results():
    report = build_report([0.0, 0.01, 0.5])
    assert report.n_passed == 2
    assert len(report.failures) == 1


def test_sourced_failures_are_separated():
    """A missed placeholder may mean the target is wrong, not the model."""
    unsourced = Moment(
        name="share_indebted",
        target=0.5,
        tolerance=0.01,
        source="not verified",
        confidence=Confidence.UNSOURCED,
    )
    results = (
        MomentResult(moment=moment(0.5, 0.01), measured=1.0, gap=0.5, passed=False),
        MomentResult(moment=unsourced, measured=1.0, gap=0.5, passed=False),
    )
    report = CalibrationReport(results=results, parameters=PopulationParameters())
    assert len(report.failures) == 2
    assert len(report.sourced_failures) == 1


def test_summary_reports_the_counts(generated):
    summary = evaluate(generated).summary()
    assert "moments matched" in summary
    assert "loss" in summary


def test_table_shows_confidence(generated):
    table = evaluate(generated).table()
    assert "confidence" in table
    assert "sourced" in table


def test_table_marks_failures(generated):
    """Failures must be visible, not filtered out of the report."""
    report = evaluate(generated)
    if report.failures:
        assert "MISS" in report.table()


def test_report_serialises_for_the_ledger(generated):
    payload = evaluate(generated).to_ledger_dict()
    assert payload["n_moments"] == len(URBAN_INDIA_MOMENTS)
    assert "parameters" in payload
    assert all("confidence" in r["moment"] for r in payload["results"])


def test_report_is_frozen(generated):
    report = evaluate(generated)
    with pytest.raises(Exception):
        report.results = ()  # type: ignore[misc]


# --- search ----------------------------------------------------------------

def test_search_evaluates_every_combination():
    _, _, trials = search(
        {"log_wealth_mean": [9.0, 10.0], "log_wealth_sd": [1.4, 1.6]},
        n_households=2_000,
    )
    assert len(trials) == 4


def test_search_returns_the_lowest_loss():
    _, report, trials = search(
        {"log_wealth_mean": [8.0, 10.0, 12.0]}, n_households=2_000
    )
    assert report.loss() == pytest.approx(min(loss for _, loss in trials))


def test_search_applies_the_chosen_parameters():
    best, _, _ = search({"log_wealth_mean": [8.0, 10.0]}, n_households=2_000)
    assert best.log_wealth_mean in (8.0, 10.0)


def test_search_preserves_unvaried_parameters():
    base = PopulationParameters(seed=99, log_income_sd=0.9)
    best, _, _ = search(
        {"log_wealth_mean": [9.0, 10.0]}, base=base, n_households=2_000
    )
    assert best.seed == 99
    assert best.log_income_sd == 0.9


def test_search_is_reproducible():
    first, _, _ = search({"log_wealth_mean": [9.0, 10.0, 11.0]}, n_households=2_000)
    second, _, _ = search({"log_wealth_mean": [9.0, 10.0, 11.0]}, n_households=2_000)
    assert first.log_wealth_mean == second.log_wealth_mean


def test_search_rejects_an_empty_grid():
    with pytest.raises(EngineError, match="at least one parameter"):
        search({}, n_households=2_000)


def test_search_rejects_an_unknown_parameter():
    with pytest.raises(EngineError, match="not a population parameter"):
        search({"not_a_parameter": [1.0]}, n_households=2_000)


def test_a_flat_loss_surface_would_be_visible():
    """If a parameter does not move the loss, these moments do not identify
    it, and its fitted value should not be quoted as though it did."""
    _, _, trials = search(
        {"initial_expected_inflation": [0.03, 0.05, 0.07]}, n_households=2_000
    )
    losses = [loss for _, loss in trials]
    assert max(losses) - min(losses) < 1e-6


def test_wealth_level_does_move_the_loss():
    """The contrast with the flat surface above: this parameter is identified.

    The grid must span the minimum rather than only its two sides. Two
    equally-wrong endpoints can produce near-identical losses, which would
    look like a flat surface when it is in fact a valley.
    """
    _, _, trials = search(
        {"log_wealth_mean": [8.0, 10.0, 12.0]}, n_households=5_000
    )
    losses = [loss for _, loss in trials]
    assert max(losses) - min(losses) > 1.0