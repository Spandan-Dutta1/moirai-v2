"""Tests for preference estimation.

The important test here is recovery: simulate a rate path from known
coefficients, then check the estimator gets them back. Everything else in
this file is secondary.

That matters because this estimator failed twice before it worked. At
monthly frequency both parameters pinned to their bounds. With a level
specification the lagged dependent variable explained ninety percent of a
persistent series and the policy coefficients collapsed to zero. A
recovery test catches either immediately: an estimator that cannot recover
parameters from data it generated itself has no business touching real
data, and both failures were diagnosed by eye instead.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.game import Confidence
from moirai.engine.financial.preferences import (
    compare_detrending,
    estimate_preferences,
    hp_gap,
    linear_gap,
    target_rate,
    to_quarterly,
)

TRUE_INFLATION_RESPONSE = 0.60
TRUE_OUTPUT_RESPONSE = 0.40
TRUE_SPEED = 0.30
NEUTRAL_REAL_RATE = 0.02
TARGET = 0.02


def simulate_policy(
    n: int = 200,
    *,
    inflation_response: float = TRUE_INFLATION_RESPONSE,
    output_response: float = TRUE_OUTPUT_RESPONSE,
    speed: float = TRUE_SPEED,
    noise: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a rate path from a known rule.

    Inflation and output follow persistent processes so the gaps have the
    variation an estimator needs. A path generated from a rule with no
    variation in its inputs is unidentifiable by construction, which was
    exactly the monthly failure.
    """
    rng = np.random.default_rng(seed)

    inflation = np.zeros(n)
    output = np.zeros(n)
    for t in range(1, n):
        inflation[t] = 0.02 + 0.85 * (inflation[t - 1] - 0.02) + rng.normal(0, 0.004)
        output[t] = 0.90 * output[t - 1] + rng.normal(0, 0.015)

    rates = np.zeros(n)
    rates[0] = 0.04
    for t in range(1, n):
        target = (
            NEUTRAL_REAL_RATE
            + inflation[t]
            + inflation_response * (inflation[t] - TARGET)
            + output_response * output[t]
        )
        rates[t] = rates[t - 1] + speed * (target - rates[t - 1])
        if noise:
            rates[t] += rng.normal(0, noise)

    # The estimator detrends `output`, so hand it a level rather than a gap.
    return rates, inflation, output + 4.5


def fit(rates, inflation, output, **options):
    return estimate_preferences(
        "Test Bank",
        rates,
        inflation,
        output,
        inflation_target=TARGET,
        neutral_real_rate=NEUTRAL_REAL_RATE,
        **options,
    )


# --- recovery --------------------------------------------------------------

def test_the_inflation_response_is_recovered():
    """The parameter that collapsed to zero in both failed specifications."""
    estimate = fit(*simulate_policy(400))
    assert estimate.inflation_weight == pytest.approx(TRUE_INFLATION_RESPONSE, abs=0.15)


def test_the_output_response_is_recovered():
    estimate = fit(*simulate_policy(400))
    assert estimate.output_weight == pytest.approx(TRUE_OUTPUT_RESPONSE, abs=0.15)


def test_the_adjustment_speed_is_recovered():
    estimate = fit(*simulate_policy(400))
    assert estimate.smoothing_weight == pytest.approx(TRUE_SPEED, abs=0.08)


def test_the_estimate_is_interior():
    """A parameter at its bound is not an estimate, it is the optimiser
    running out of room. Both earlier failures looked like this."""
    assert fit(*simulate_policy(400)).converged is True


def test_recovery_improves_with_sample_size():
    def error(n: int) -> float:
        estimate = fit(*simulate_policy(n))
        return abs(estimate.inflation_weight - TRUE_INFLATION_RESPONSE)

    assert error(600) <= error(80) + 0.05


def test_a_noiseless_path_is_fitted_almost_exactly():
    """High but not perfect. The estimator detrends linearly while the
    simulated output is an AR process, so the recovered gap differs
    slightly from the one that generated the rates. That mismatch caps the
    achievable fit even with no noise, which is worth knowing: a perfect
    R-squared here would mean the test was circular.
    """
    assert fit(*simulate_policy(300)).r_squared > 0.85


def test_noise_lowers_the_fit_without_breaking_recovery():
    noisy = fit(*simulate_policy(400, noise=0.002))
    assert noisy.r_squared < 0.99
    assert noisy.inflation_weight == pytest.approx(TRUE_INFLATION_RESPONSE, abs=0.3)


def test_a_more_hawkish_rule_is_recovered_as_more_hawkish():
    """Ordering, which is weaker than recovery but must never fail."""
    dovish = fit(*simulate_policy(400, inflation_response=0.2))
    hawkish = fit(*simulate_policy(400, inflation_response=1.5))
    assert hawkish.inflation_weight > dovish.inflation_weight


def test_a_slower_bank_is_recovered_as_slower():
    fast = fit(*simulate_policy(400, speed=0.6))
    slow = fit(*simulate_policy(400, speed=0.1))
    assert slow.smoothing_weight < fast.smoothing_weight


# --- the target rate -------------------------------------------------------

def test_the_target_equals_neutral_plus_inflation_at_target():
    rate = target_rate(
        np.array([0.02]),
        np.array([0.0]),
        inflation_target=0.02,
        inflation_response=0.5,
        output_response=0.5,
        neutral_real_rate=0.02,
    )
    assert rate[0] == pytest.approx(0.04)


def test_an_inflation_overshoot_raises_the_target():
    above = target_rate(
        np.array([0.05]), np.array([0.0]),
        inflation_target=0.02, inflation_response=0.5,
        output_response=0.5, neutral_real_rate=0.02,
    )
    at = target_rate(
        np.array([0.02]), np.array([0.0]),
        inflation_target=0.02, inflation_response=0.5,
        output_response=0.5, neutral_real_rate=0.02,
    )
    assert above[0] > at[0]


def test_the_nominal_rate_rises_more_than_one_for_one():
    """The Taylor principle. If it does not, the real rate falls when
    inflation rises, which amplifies shocks instead of damping them."""
    low = target_rate(
        np.array([0.02]), np.array([0.0]),
        inflation_target=0.02, inflation_response=0.5,
        output_response=0.0, neutral_real_rate=0.02,
    )[0]
    high = target_rate(
        np.array([0.03]), np.array([0.0]),
        inflation_target=0.02, inflation_response=0.5,
        output_response=0.0, neutral_real_rate=0.02,
    )[0]
    assert (high - low) > 0.01


def test_a_positive_output_gap_raises_the_target():
    rates = target_rate(
        np.array([0.02, 0.02]), np.array([0.0, 0.05]),
        inflation_target=0.02, inflation_response=0.5,
        output_response=0.5, neutral_real_rate=0.02,
    )
    assert rates[1] > rates[0]


# --- detrending ------------------------------------------------------------

def test_the_linear_gap_has_mean_zero():
    gap, _ = linear_gap(np.linspace(0.0, 10.0, 100) + np.sin(np.arange(100)))
    assert gap.mean() == pytest.approx(0.0, abs=1e-9)


def test_the_linear_gap_removes_a_pure_trend():
    gap, _ = linear_gap(np.linspace(0.0, 10.0, 100))
    assert np.abs(gap).max() < 1e-9


def test_the_hp_gap_removes_a_trend_less_completely():
    """The filter follows the trend rather than assuming it is linear, so
    a pure trend leaves a small residual at the sample edges."""
    gap, _ = hp_gap(np.linspace(0.0, 10.0, 100))
    assert np.abs(gap).max() < 0.5


def test_the_hp_gap_records_its_smoothing_parameter():
    _, method = hp_gap(np.random.default_rng(0).normal(0, 1, 60))
    assert method.parameter == 14_400.0


def test_the_hp_method_records_its_criticism():
    """Hamilton argued the filter should not be used at all, and the note
    exists so nobody treats it as a neutral choice."""
    _, method = hp_gap(np.random.default_rng(0).normal(0, 1, 60))
    assert "Hamilton" in method.note


def test_a_short_series_cannot_be_filtered():
    with pytest.raises(EngineError, match="at least five"):
        hp_gap(np.array([1.0, 2.0, 3.0]))


def test_the_two_methods_broadly_agree_on_a_clean_series():
    series = np.linspace(0.0, 5.0, 200) + np.sin(np.linspace(0, 20, 200))
    linear, _ = linear_gap(series)
    hp, _ = hp_gap(series)
    assert np.corrcoef(linear, hp)[0, 1] > 0.5


def test_both_detrendings_are_reported_together():
    """Reported rather than choosing the better fit: a large disagreement
    means the output gap construction is doing the work."""
    estimates = compare_detrending(
        "Test Bank", *simulate_policy(300),
        inflation_target=TARGET, neutral_real_rate=NEUTRAL_REAL_RATE,
    )
    assert set(estimates) == {"linear", "hp"}


# --- quarterly aggregation -------------------------------------------------

def test_three_months_become_one_quarter():
    assert len(to_quarterly(np.arange(12.0))) == 4


def test_the_quarter_is_the_mean_of_its_months():
    assert to_quarterly(np.array([1.0, 2.0, 3.0]))[0] == pytest.approx(2.0)


def test_a_trailing_partial_quarter_is_dropped():
    assert len(to_quarterly(np.arange(14.0))) == 4


def test_last_month_sampling_is_available():
    assert to_quarterly(np.array([1.0, 2.0, 3.0]), how="last")[0] == 3.0


def test_fewer_than_three_months_is_rejected():
    with pytest.raises(EngineError, match="fewer than one quarter"):
        to_quarterly(np.array([1.0, 2.0]))


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(EngineError, match="unknown aggregation"):
        to_quarterly(np.arange(6.0), how="median")


# --- validation ------------------------------------------------------------

def test_mismatched_lengths_are_rejected():
    rates, inflation, output = simulate_policy(100)
    with pytest.raises(EngineError, match="lengths differ"):
        fit(rates, inflation[:50], output)


def test_too_few_observations_are_rejected():
    with pytest.raises(EngineError, match="too few"):
        fit(*simulate_policy(15))


def test_non_finite_rates_are_rejected():
    rates, inflation, output = simulate_policy(100)
    rates[10] = np.nan
    with pytest.raises(EngineError, match="finite"):
        fit(rates, inflation, output)


# --- reporting -------------------------------------------------------------

def test_the_estimate_is_marked_derived():
    """Not sourced. It recovers revealed preferences from behaviour,
    including behaviour that was mistaken."""
    assert fit(*simulate_policy(300)).confidence is Confidence.DERIVED


def test_the_ratio_is_reported():
    estimate = fit(*simulate_policy(400))
    expected = TRUE_OUTPUT_RESPONSE / TRUE_INFLATION_RESPONSE
    assert estimate.output_to_inflation_ratio == pytest.approx(expected, abs=0.4)


def test_a_near_zero_inflation_response_gives_an_infinite_ratio():
    """Signals a failed estimate rather than an extreme preference."""
    estimate = fit(*simulate_policy(300))
    broken = estimate.model_copy(update={"inflation_weight": 0.0})
    assert broken.output_to_inflation_ratio == float("inf")


def test_the_taylor_principle_is_reported():
    assert fit(*simulate_policy(300)).satisfies_taylor_principle is True


def test_the_detrending_used_is_recorded():
    """The estimate depends on it, so it travels with the result."""
    assert fit(*simulate_policy(300), detrend="hp").detrend.name == "hp_filter"


def test_the_sample_is_recorded():
    estimate = fit(
        *simulate_policy(300), sample_start="1986-01-01", sample_end="2007-06-01"
    )
    assert estimate.sample_start == "1986-01-01"


def test_the_estimate_serialises_for_the_ledger():
    payload = fit(*simulate_policy(300)).to_ledger_dict()
    assert payload["confidence"] == "derived"
    assert "detrend" in payload
    assert "r_squared" in payload


def test_the_estimate_is_frozen():
    estimate = fit(*simulate_policy(300))
    with pytest.raises(Exception):
        estimate.output_weight = 9.9  # type: ignore[misc]


def test_estimation_does_not_mutate_the_input():
    rates, inflation, output = simulate_policy(200)
    before = rates.copy()
    fit(rates, inflation, output)
    assert np.array_equal(rates, before)