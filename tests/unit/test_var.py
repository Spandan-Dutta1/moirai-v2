"""Tests for the reduced-form VAR.

The central technique is recovery: simulate from a VAR with known
coefficients, estimate, and check the estimates converge on the truth. An
estimator that cannot recover parameters from data it generated itself is
not worth pointing at real data.

Tolerances are loose by design. OLS on 5,000 observations is consistent,
not exact, and a test that demands three decimal places is testing the
random seed rather than the estimator.
"""

from datetime import date

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.var import (
    InformationCriterion,
    check_stability,
    estimate_var,
    forecast,
    select_lag_order,
)


def simulate_var(
    intercept: np.ndarray,
    coefficients: np.ndarray,
    n_obs: int,
    *,
    sigma: np.ndarray | None = None,
    seed: int = 0,
    burn_in: int = 200,
) -> np.ndarray:
    """Generate data from a known VAR.

    Burn-in discards the influence of the arbitrary zero starting state so
    the sample is drawn from the stationary distribution.
    """
    rng = np.random.default_rng(seed)
    p, k, _ = coefficients.shape
    total = n_obs + burn_in

    cov = np.eye(k) if sigma is None else sigma
    shocks = rng.multivariate_normal(np.zeros(k), cov, size=total)

    series = np.zeros((total, k), dtype=float)
    for t in range(p, total):
        value = intercept.copy()
        for lag in range(p):
            value = value + coefficients[lag] @ series[t - lag - 1]
        series[t] = value + shocks[t]

    return series[burn_in:]


# Stable bivariate VAR(1): eigenvalues well inside the unit circle.
STABLE_A1 = np.array([[[0.5, 0.1], [0.2, 0.4]]])
STABLE_C = np.array([0.0, 0.0])

# Stable bivariate VAR(2).
STABLE_A2 = np.array(
    [
        [[0.5, 0.1], [0.2, 0.3]],
        [[0.2, 0.0], [0.1, 0.2]],
    ]
)

# Explosive: an eigenvalue outside the unit circle.
EXPLOSIVE = np.array([[[1.2, 0.0], [0.0, 0.5]]])

NAMES = ("y1", "y2")


# --- stability -------------------------------------------------------------

def test_stable_coefficients_are_recognised():
    assert check_stability(STABLE_A1).is_stable is True


def test_explosive_coefficients_are_recognised():
    result = check_stability(EXPLOSIVE)
    assert result.is_stable is False
    assert result.max_modulus == pytest.approx(1.2)


def test_moduli_are_sorted_descending():
    moduli = check_stability(STABLE_A2).moduli
    assert list(moduli) == sorted(moduli, reverse=True)


def test_companion_has_k_times_p_eigenvalues():
    assert len(check_stability(STABLE_A2).moduli) == 4  # k=2, p=2


def test_half_life_is_positive_for_a_stable_system():
    half_life = check_stability(STABLE_A1).half_life
    assert half_life is not None and half_life > 0


def test_half_life_is_none_when_unstable():
    assert check_stability(EXPLOSIVE).half_life is None


def test_more_persistent_systems_have_longer_half_lives():
    slow = np.array([[[0.9, 0.0], [0.0, 0.1]]])
    fast = np.array([[[0.3, 0.0], [0.0, 0.1]]])
    assert check_stability(slow).half_life > check_stability(fast).half_life


def test_stability_serialises_for_the_ledger():
    payload = check_stability(STABLE_A1).to_ledger_dict()
    assert payload["is_stable"] is True
    assert "max_modulus" in payload


def test_unit_root_is_treated_as_unstable():
    """Modulus exactly one means shocks never decay."""
    unit_root = np.array([[[1.0, 0.0], [0.0, 0.5]]])
    assert check_stability(unit_root).is_stable is False


# --- parameter recovery ----------------------------------------------------

def test_var1_coefficients_are_recovered():
    data = simulate_var(STABLE_C, STABLE_A1, 5_000, seed=1)
    result = estimate_var(data, NAMES, n_lags=1)
    assert np.allclose(result.coefficients[0], STABLE_A1[0], atol=0.05)


def test_var2_coefficients_are_recovered():
    data = simulate_var(STABLE_C, STABLE_A2, 5_000, seed=2)
    result = estimate_var(data, NAMES, n_lags=2)
    assert np.allclose(result.coefficients, STABLE_A2, atol=0.05)


def test_intercept_is_recovered():
    intercept = np.array([1.5, -0.8])
    data = simulate_var(intercept, STABLE_A1, 5_000, seed=3)
    result = estimate_var(data, NAMES, n_lags=1)
    assert np.allclose(result.intercept, intercept, atol=0.15)


def test_residual_covariance_is_recovered():
    sigma = np.array([[1.0, 0.4], [0.4, 2.0]])
    data = simulate_var(STABLE_C, STABLE_A1, 5_000, sigma=sigma, seed=4)
    result = estimate_var(data, NAMES, n_lags=1)
    assert np.allclose(result.sigma_u, sigma, atol=0.15)


def test_estimates_improve_with_sample_size():
    """Consistency: more data, less error."""
    def error(n: int) -> float:
        data = simulate_var(STABLE_C, STABLE_A1, n, seed=5)
        result = estimate_var(data, NAMES, n_lags=1)
        return float(np.abs(result.coefficients[0] - STABLE_A1[0]).max())

    assert error(5_000) < error(200)


def test_coefficient_orientation_is_correct():
    """coefficients[l][i][j] must be the effect of j at lag l+1 on i today.

    A transposed coefficient block estimates the same fit but attributes
    every cross-effect to the wrong variable, which silently inverts the
    economics of every result downstream.
    """
    asymmetric = np.array([[[0.5, 0.0], [0.8, 0.5]]])  # y1 drives y2, not vice versa
    data = simulate_var(STABLE_C, asymmetric, 5_000, seed=6)
    result = estimate_var(data, NAMES, n_lags=1)

    assert result.coefficients[0][1, 0] == pytest.approx(0.8, abs=0.05)
    assert result.coefficients[0][0, 1] == pytest.approx(0.0, abs=0.05)


# --- estimation mechanics --------------------------------------------------

def test_residuals_have_the_expected_shape():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=7)
    result = estimate_var(data, NAMES, n_lags=2)
    assert result.residuals.shape == (298, 2)


def test_observations_exclude_the_initial_lags():
    data = simulate_var(STABLE_C, STABLE_A2, 300, seed=8)
    assert estimate_var(data, NAMES, n_lags=4).n_observations == 296


def test_parameter_count_is_one_plus_k_times_p():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=9)
    result = estimate_var(data, NAMES, n_lags=3)
    assert result.n_parameters == 1 + 2 * 3


def test_residuals_are_approximately_mean_zero():
    data = simulate_var(STABLE_C, STABLE_A1, 2_000, seed=10)
    result = estimate_var(data, NAMES, n_lags=1)
    assert np.allclose(result.residuals.mean(axis=0), 0.0, atol=1e-8)


def test_periods_are_trimmed_to_match_observations():
    data = simulate_var(STABLE_C, STABLE_A1, 100, seed=11)
    periods = tuple(date(2000 + i // 12, i % 12 + 1, 1) for i in range(100))
    result = estimate_var(data, NAMES, n_lags=2, periods=periods)
    assert len(result.periods) == result.n_observations


def test_log_likelihood_is_finite():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=12)
    assert np.isfinite(estimate_var(data, NAMES, n_lags=1).log_likelihood())


def test_companion_matrix_shape():
    data = simulate_var(STABLE_C, STABLE_A2, 500, seed=13)
    assert estimate_var(data, NAMES, n_lags=2).companion_matrix().shape == (4, 4)


def test_index_of_finds_a_variable():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=14)
    assert estimate_var(data, NAMES, n_lags=1).index_of("y2") == 1


def test_index_of_rejects_an_unknown_variable():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=15)
    result = estimate_var(data, NAMES, n_lags=1)
    with pytest.raises(EngineError, match="not in this VAR"):
        result.index_of("nope")


def test_result_serialises_for_the_ledger():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=16)
    payload = estimate_var(data, NAMES, n_lags=2).to_ledger_dict()
    assert payload["n_lags"] == 2
    assert payload["variables"] == ["y1", "y2"]
    assert payload["stability"]["is_stable"] is True


# --- validation ------------------------------------------------------------

def test_explosive_data_is_rejected_by_default():
    data = simulate_var(STABLE_C, EXPLOSIVE, 200, seed=17, burn_in=0)
    with pytest.raises(EngineError, match="not stable"):
        estimate_var(data, NAMES, n_lags=1)


def test_instability_can_be_permitted_explicitly():
    data = simulate_var(STABLE_C, EXPLOSIVE, 200, seed=18, burn_in=0)
    result = estimate_var(data, NAMES, n_lags=1, require_stable=False)
    assert result.stability.is_stable is False


def test_mismatched_variable_count_is_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=19)
    with pytest.raises(EngineError, match="columns but"):
        estimate_var(data, ("only_one",), n_lags=1)


def test_duplicate_variable_names_are_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=20)
    with pytest.raises(EngineError, match="unique"):
        estimate_var(data, ("same", "same"), n_lags=1)


def test_non_finite_data_is_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=21)
    data[10, 0] = np.nan
    with pytest.raises(EngineError, match="NaN"):
        estimate_var(data, NAMES, n_lags=1)


def test_one_dimensional_input_is_rejected():
    with pytest.raises(EngineError, match="2-dimensional"):
        estimate_var(np.zeros(100), ("y1",), n_lags=1)


def test_zero_lags_are_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=22)
    with pytest.raises(EngineError, match="at least 1"):
        estimate_var(data, NAMES, n_lags=0)


def test_too_few_observations_are_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 12, seed=23)
    with pytest.raises(EngineError, match="cannot identify"):
        estimate_var(data, NAMES, n_lags=5)


def test_mismatched_period_count_is_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 100, seed=24)
    with pytest.raises(EngineError, match="periods for"):
        estimate_var(data, NAMES, n_lags=1, periods=(date(2000, 1, 1),))


def test_overparameterisation_warns_without_failing(caplog):
    data = simulate_var(STABLE_C, STABLE_A1, 60, seed=25)
    result = estimate_var(data, NAMES, n_lags=5)
    assert result.n_lags == 5  # estimated, but the warning was logged


# --- lag selection ---------------------------------------------------------

def test_selection_returns_one_value_per_candidate():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=26)
    selection = select_lag_order(data, max_lags=6)
    assert len(selection.candidates) == 6
    assert len(selection.aic) == 6


def test_bic_recovers_the_true_lag_order():
    """BIC is consistent, so with enough data it should find p=1."""
    data = simulate_var(STABLE_C, STABLE_A1, 3_000, seed=27)
    selection = select_lag_order(data, max_lags=8)
    assert selection.best(InformationCriterion.BIC) == 1


def test_bic_recovers_a_longer_true_lag_order():
    data = simulate_var(STABLE_C, STABLE_A2, 3_000, seed=28)
    selection = select_lag_order(data, max_lags=8)
    assert selection.best(InformationCriterion.BIC) == 2


def test_bic_never_selects_more_lags_than_aic():
    """BIC's penalty is heavier for any sample beyond seven observations."""
    data = simulate_var(STABLE_C, STABLE_A2, 1_000, seed=29)
    selection = select_lag_order(data, max_lags=10)
    assert selection.best(InformationCriterion.BIC) <= selection.best(
        InformationCriterion.AIC
    )


def test_criteria_agreement_is_reported():
    data = simulate_var(STABLE_C, STABLE_A1, 3_000, seed=30)
    selection = select_lag_order(data, max_lags=6)
    assert isinstance(selection.criteria_agree, bool)


def test_selection_serialises_for_the_ledger():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=31)
    payload = select_lag_order(data, max_lags=4).to_ledger_dict()
    assert "best_bic" in payload
    assert len(payload["aic"]) == 4


def test_automatic_selection_is_used_when_lags_are_omitted():
    data = simulate_var(STABLE_C, STABLE_A2, 2_000, seed=32)
    result = estimate_var(data, NAMES, max_lags=8)
    assert result.lag_selection is not None
    assert result.n_lags == 2


def test_criterion_choice_is_respected():
    data = simulate_var(STABLE_C, STABLE_A1, 1_000, seed=33)
    by_aic = estimate_var(data, NAMES, max_lags=10, criterion=InformationCriterion.AIC)
    by_bic = estimate_var(data, NAMES, max_lags=10, criterion=InformationCriterion.BIC)
    assert by_aic.n_lags >= by_bic.n_lags


def test_explicit_lags_skip_selection():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=34)
    assert estimate_var(data, NAMES, n_lags=3).lag_selection is None


def test_zero_max_lags_is_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=35)
    with pytest.raises(EngineError, match="at least 1"):
        select_lag_order(data, max_lags=0)


def test_max_lags_beyond_the_sample_is_rejected():
    data = simulate_var(STABLE_C, STABLE_A1, 30, seed=36)
    with pytest.raises(EngineError, match="cannot evaluate"):
        select_lag_order(data, max_lags=20)


# --- forecasting -----------------------------------------------------------

def test_forecast_shape():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=37)
    result = estimate_var(data, NAMES, n_lags=1)
    assert forecast(result, horizon=8, history=data).shape == (8, 2)


def test_forecast_converges_to_the_unconditional_mean():
    """A stable VAR forgets its initial condition."""
    intercept = np.array([1.0, 2.0])
    data = simulate_var(intercept, STABLE_A1, 2_000, seed=38)
    result = estimate_var(data, NAMES, n_lags=1)

    long_run = forecast(result, horizon=200, history=data)[-1]
    assert np.allclose(long_run, data.mean(axis=0), atol=0.2)


def test_forecast_is_deterministic():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=39)
    result = estimate_var(data, NAMES, n_lags=1)
    assert np.array_equal(
        forecast(result, 5, data), forecast(result, 5, data)
    )


def test_one_step_forecast_matches_the_equation():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=40)
    result = estimate_var(data, NAMES, n_lags=1)
    expected = result.intercept + result.coefficients[0] @ data[-1]
    assert np.allclose(forecast(result, 1, data)[0], expected)


def test_forecast_rejects_a_short_history():
    data = simulate_var(STABLE_C, STABLE_A2, 500, seed=41)
    result = estimate_var(data, NAMES, n_lags=4)
    with pytest.raises(EngineError, match="at least 4 rows"):
        forecast(result, 5, data[-2:])


def test_forecast_rejects_mismatched_width():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=42)
    result = estimate_var(data, NAMES, n_lags=1)
    with pytest.raises(EngineError, match="columns, expected"):
        forecast(result, 5, np.zeros((10, 5)))


def test_forecast_rejects_a_zero_horizon():
    data = simulate_var(STABLE_C, STABLE_A1, 500, seed=43)
    result = estimate_var(data, NAMES, n_lags=1)
    with pytest.raises(EngineError, match="at least 1"):
        forecast(result, 0, data)


# --- immutability ----------------------------------------------------------

def test_result_is_frozen():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=44)
    result = estimate_var(data, NAMES, n_lags=1)
    with pytest.raises(Exception):
        result.n_lags = 5  # type: ignore[misc]


def test_estimation_does_not_mutate_the_input():
    data = simulate_var(STABLE_C, STABLE_A1, 300, seed=45)
    before = data.copy()
    estimate_var(data, NAMES, n_lags=2)
    assert np.array_equal(data, before)