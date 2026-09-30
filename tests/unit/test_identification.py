"""Tests for structural identification.

The properties being checked are mathematical rather than economic. An
identification is correct when B reproduces Sigma, the shocks come out
orthogonal, and the assumed restrictions actually hold. Whether the
assumption is *defensible* is an economic question no test can settle,
which is why the assumption is carried in the result rather than hidden.
"""

from typing import NamedTuple

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.identification import (
    IdentificationScheme,
    Sign,
    SignRestriction,
    identify_cholesky,
    identify_external,
    identify_sign_restrictions,
    ordering_sensitivity,
)
from moirai.engine.causal.var import estimate_var

NAMES = ("output", "prices", "rate")

# Stable trivariate VAR(1) with correlated shocks, so identification bites.
COEFFICIENTS = np.array(
    [
        [
            [0.50, 0.10, -0.20],
            [0.15, 0.60, -0.10],
            [0.20, 0.25, 0.55],
        ]
    ]
)
SIGMA = np.array(
    [
        [1.00, 0.30, 0.20],
        [0.30, 1.00, 0.40],
        [0.20, 0.40, 1.00],
    ]
)


def simulate(n_obs: int = 2_000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burn_in = 200
    total = n_obs + burn_in
    shocks = rng.multivariate_normal(np.zeros(3), SIGMA, size=total)

    series = np.zeros((total, 3))
    for t in range(1, total):
        series[t] = COEFFICIENTS[0] @ series[t - 1] + shocks[t]
    return series[burn_in:]


@pytest.fixture(scope="module")
def var():
    return estimate_var(simulate(), NAMES, n_lags=1)


# --- Cholesky: correctness -------------------------------------------------

def test_impact_reproduces_the_covariance(var):
    """B B' must equal Sigma exactly, whatever the assumption."""
    assert identify_cholesky(var).reconstruction_error() < 1e-10


def test_impact_is_lower_triangular(var):
    """The recursive assumption is exactly this triangularity."""
    impact = identify_cholesky(var, ordering=NAMES).impact
    assert np.allclose(np.triu(impact, k=1), 0.0)


def test_shocks_are_orthogonal(var):
    assert identify_cholesky(var).max_off_diagonal_correlation() < 1e-8


def test_shocks_have_unit_variance(var):
    """eps = B^-1 u, so the structural shocks are standardised by construction."""
    shocks = identify_cholesky(var).shocks
    assert np.allclose(shocks.std(axis=0, ddof=1), 1.0, atol=0.05)


def test_shock_count_matches_variable_count(var):
    assert identify_cholesky(var).shocks.shape[1] == 3


def test_shock_rows_match_residual_rows(var):
    model = identify_cholesky(var)
    assert model.shocks.shape[0] == var.residuals.shape[0]


def test_first_variable_responds_only_to_its_own_shock(var):
    """The whole content of the ordering, in one row of B."""
    impact = identify_cholesky(var, ordering=NAMES).impact
    assert impact[0, 1] == 0.0
    assert impact[0, 2] == 0.0
    assert impact[0, 0] > 0.0


def test_last_variable_responds_to_every_shock(var):
    impact = identify_cholesky(var, ordering=NAMES).impact
    assert np.all(np.abs(impact[2]) > 0.0)


# --- Cholesky: ordering ----------------------------------------------------

def test_ordering_is_recorded(var):
    assert identify_cholesky(var, ordering=("rate", "prices", "output")).ordering == (
        "rate",
        "prices",
        "output",
    )


def test_ordering_changes_the_impact_matrix(var):
    """If it did not, the assumption would be doing no work."""
    forward = identify_cholesky(var, ordering=NAMES).impact
    reversed_ = identify_cholesky(var, ordering=tuple(reversed(NAMES))).impact
    assert not np.allclose(forward, reversed_)


def test_every_ordering_reproduces_the_covariance(var):
    """Different assumptions, same observed data."""
    for ordering in [NAMES, tuple(reversed(NAMES)), ("prices", "rate", "output")]:
        assert identify_cholesky(var, ordering).reconstruction_error() < 1e-10


def test_reversed_ordering_zeroes_the_opposite_corner(var):
    impact = identify_cholesky(var, ordering=tuple(reversed(NAMES))).impact
    # "rate" is now first, so it responds to nothing else on impact.
    assert impact[2, 0] == pytest.approx(0.0, abs=1e-12)


def test_default_ordering_is_the_var_ordering(var):
    assert identify_cholesky(var).ordering == NAMES


def test_assumption_text_names_the_ordering(var):
    assumptions = identify_cholesky(var, ordering=NAMES).assumptions
    assert "output" in assumptions and "rate" in assumptions
    assert "Untestable" in assumptions


def test_scheme_is_recorded(var):
    assert identify_cholesky(var).scheme is IdentificationScheme.CHOLESKY


# --- Cholesky: validation --------------------------------------------------

def test_wrong_length_ordering_is_rejected(var):
    with pytest.raises(EngineError, match="names for"):
        identify_cholesky(var, ordering=("output", "prices"))


def test_unknown_variable_in_ordering_is_rejected(var):
    with pytest.raises(EngineError, match="mismatch"):
        identify_cholesky(var, ordering=("output", "prices", "nope"))


def test_duplicate_in_ordering_is_rejected(var):
    with pytest.raises(EngineError, match="mismatch"):
        identify_cholesky(var, ordering=("output", "output", "prices"))


def test_wrong_number_of_shock_names_is_rejected(var):
    with pytest.raises(EngineError, match="shock names"):
        identify_cholesky(var, shock_names=("a", "b"))


def test_custom_shock_names_are_used(var):
    names = ("supply", "demand", "monetary")
    assert identify_cholesky(var, shock_names=names).shock_names == names


def test_singular_covariance_is_rejected(var):
    """A non-positive-definite Sigma has no Cholesky factor."""
    singular = np.ones((3, 3))
    broken = var.model_copy(update={"sigma_u": singular})
    with pytest.raises(EngineError, match="positive definite"):
        identify_cholesky(broken)


def test_model_serialises_for_the_ledger(var):
    payload = identify_cholesky(var, ordering=NAMES).to_ledger_dict()
    assert payload["scheme"] == "cholesky"
    assert payload["ordering"] == list(NAMES)
    assert "assumptions" in payload
    assert payload["reconstruction_error"] < 1e-10


# --- ordering sensitivity --------------------------------------------------

def test_sensitivity_covers_every_permutation(var):
    assert ordering_sensitivity(var, "output", "rate").n_orderings == 6


def test_sensitivity_minimum_is_zero_when_the_response_can_be_assumed_away(var):
    """Orderings placing `output` before `rate` force the response to zero.

    The lower bound is therefore the assumption itself, not an estimate.
    """
    result = ordering_sensitivity(var, "output", "rate")
    assert result.minimum == pytest.approx(0.0, abs=1e-12)


def test_own_response_is_never_zero(var):
    result = ordering_sensitivity(var, "rate", "rate")
    assert result.minimum > 0.0


def test_own_response_is_less_ordering_sensitive_than_cross_response(var):
    own = ordering_sensitivity(var, "rate", "rate")
    cross = ordering_sensitivity(var, "output", "rate")
    assert own.relative_spread < cross.relative_spread


def test_spread_is_maximum_minus_minimum(var):
    result = ordering_sensitivity(var, "output", "rate")
    assert result.spread == pytest.approx(result.maximum - result.minimum)


def test_median_lies_within_the_range(var):
    result = ordering_sensitivity(var, "prices", "rate")
    assert result.minimum <= result.median <= result.maximum


def test_sign_flips_are_reported(var):
    assert isinstance(ordering_sensitivity(var, "output", "rate").sign_flips, bool)


def test_relative_spread_is_infinite_when_the_median_is_zero(var):
    from moirai.engine.causal.identification import OrderingSensitivity

    result = OrderingSensitivity(
        response_of="a",
        response_to="b",
        n_orderings=2,
        minimum=-1.0,
        maximum=1.0,
        median=0.0,
        sign_flips=True,
        values_by_ordering=((("a", "b"), -1.0), (("b", "a"), 1.0)),
    )
    assert result.relative_spread == float("inf")


def test_sensitivity_rejects_an_unknown_variable(var):
    with pytest.raises(EngineError, match="not in this VAR"):
        ordering_sensitivity(var, "nope", "rate")


def test_sensitivity_serialises_for_the_ledger(var):
    payload = ordering_sensitivity(var, "output", "rate").to_ledger_dict()
    assert payload["n_orderings"] == 6
    assert "sign_flips" in payload


# --- sign restrictions -----------------------------------------------------

SHOCKS = ("supply", "demand", "monetary")

CONTRACTIONARY = (
    SignRestriction(variable="rate", shock="monetary", sign=Sign.POSITIVE),
    SignRestriction(variable="output", shock="monetary", sign=Sign.NEGATIVE),
)


def test_restriction_checks_positive():
    restriction = SignRestriction(variable="x", shock="s", sign=Sign.POSITIVE)
    assert restriction.is_satisfied(0.5) is True
    assert restriction.is_satisfied(-0.5) is False


def test_restriction_checks_negative():
    restriction = SignRestriction(variable="x", shock="s", sign=Sign.NEGATIVE)
    assert restriction.is_satisfied(-0.5) is True
    assert restriction.is_satisfied(0.5) is False


def test_unrestricted_accepts_anything():
    restriction = SignRestriction(variable="x", shock="s", sign=Sign.UNRESTRICTED)
    assert restriction.is_satisfied(0.5) and restriction.is_satisfied(-0.5)


def test_accepted_draws_reproduce_the_covariance(var):
    """Every rotation of a Cholesky factor is still a valid decomposition."""
    result = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500)
    for candidate in result.accepted[:20]:
        assert np.abs(candidate @ candidate.T - var.sigma_u).max() < 1e-10


def test_accepted_draws_satisfy_every_restriction(var):
    """The restrictions are imposed, so they must hold by construction."""
    result = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500)
    rate_row, output_row = var.index_of("rate"), var.index_of("output")
    monetary = SHOCKS.index("monetary")

    assert np.all(result.accepted[:, rate_row, monetary] > 0)
    assert np.all(result.accepted[:, output_row, monetary] < 0)


def test_identification_is_a_set_not_a_point(var):
    """Sign restrictions do not pin down a single answer."""
    result = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=1_000)
    assert result.n_accepted > 1
    low, _, high = result.impact_quantiles("output", "monetary")
    assert low < high


def test_acceptance_rate_is_between_zero_and_one(var):
    result = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500)
    assert 0.0 < result.acceptance_rate <= 1.0


def test_more_restrictions_lower_the_acceptance_rate(var):
    loose = identify_sign_restrictions(var, CONTRACTIONARY[:1], SHOCKS, n_draws=2_000)
    extra = SignRestriction(variable="prices", shock="monetary", sign=Sign.NEGATIVE)
    tight = identify_sign_restrictions(
        var,
        CONTRACTIONARY + (extra,),
        SHOCKS,
        n_draws=2_000,
    )
    assert tight.acceptance_rate < loose.acceptance_rate


def test_result_is_reproducible(var):
    """A fixed seed matters: an identified set that moves is not a result."""
    first = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500, seed=1)
    second = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500, seed=1)
    assert np.array_equal(first.accepted, second.accepted)


def test_different_seeds_explore_different_draws(var):
    first = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500, seed=1)
    second = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=500, seed=2)
    assert not np.array_equal(first.accepted, second.accepted)


def test_quantiles_are_ordered(var):
    result = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=1_000)
    low, mid, high = result.impact_quantiles("rate", "monetary")
    assert low <= mid <= high


def test_impossible_restrictions_raise(var):
    """A shock cannot raise and lower the same variable."""
    contradictory = (
        SignRestriction(variable="rate", shock="monetary", sign=Sign.POSITIVE),
        SignRestriction(variable="rate", shock="monetary", sign=Sign.NEGATIVE),
    )
    with pytest.raises(EngineError, match="no draw satisfied"):
        identify_sign_restrictions(var, contradictory, SHOCKS, n_draws=200)


def test_empty_restrictions_are_rejected(var):
    with pytest.raises(EngineError, match="at least one"):
        identify_sign_restrictions(var, (), SHOCKS)


def test_unknown_variable_in_restriction_is_rejected(var):
    bad = (SignRestriction(variable="nope", shock="monetary", sign=Sign.POSITIVE),)
    with pytest.raises(EngineError, match="not in this VAR"):
        identify_sign_restrictions(var, bad, SHOCKS)


def test_unknown_shock_in_restriction_is_rejected(var):
    bad = (SignRestriction(variable="rate", shock="fiscal", sign=Sign.POSITIVE),)
    with pytest.raises(EngineError, match="unknown shock"):
        identify_sign_restrictions(var, bad, SHOCKS)


def test_duplicate_shock_names_are_rejected(var):
    with pytest.raises(EngineError, match="unique"):
        identify_sign_restrictions(var, CONTRACTIONARY, ("a", "a", "b"))


def test_wrong_shock_name_count_is_rejected(var):
    with pytest.raises(EngineError, match="shock names for"):
        identify_sign_restrictions(var, CONTRACTIONARY, ("a", "b"))


# --- sign restrictions beyond impact ---------------------------------------

SLOW_OUTPUT = (
    SignRestriction(variable="rate", shock="monetary", sign=Sign.POSITIVE),
    SignRestriction(
        variable="output", shock="monetary", sign=Sign.NEGATIVE, horizons=(0, 1, 2)
    ),
)


def test_a_restriction_defaults_to_impact_only():
    restriction = SignRestriction(variable="x", shock="s", sign=Sign.POSITIVE)
    assert restriction.horizons == (0,)


@pytest.mark.parametrize("horizons", [(), (-1, 0), (0, 0), (2, 1)])
def test_invalid_horizons_are_rejected(horizons):
    with pytest.raises(ValueError):
        SignRestriction(variable="x", shock="s", sign=Sign.NEGATIVE, horizons=horizons)


def test_ma_coefficients_are_powers_of_a_var1(var):
    """For a VAR(1) the closed form is Psi_h = A^h."""
    ma = var.ma_coefficients(4)
    a = var.coefficients[0]
    for h in range(5):
        assert np.allclose(ma[h], np.linalg.matrix_power(a, h))


def test_response_paths_match_the_closed_form(var):
    """Response at horizon h to shock j is A^h B[:, j] for a VAR(1)."""
    result = identify_sign_restrictions(var, SLOW_OUTPUT, SHOCKS, n_draws=500)
    paths = result.response_paths("output", "monetary", horizon=6)
    a, row, column = var.coefficients[0], var.index_of("output"), SHOCKS.index("monetary")
    for n, candidate in enumerate(result.accepted[:20]):
        for h in range(7):
            expected = (np.linalg.matrix_power(a, h) @ candidate[:, column])[row]
            assert paths[n, h] == pytest.approx(expected, abs=1e-12)


def test_horizon_restrictions_hold_at_every_restricted_horizon(var):
    result = identify_sign_restrictions(var, SLOW_OUTPUT, SHOCKS, n_draws=2_000)
    output = result.response_paths("output", "monetary", horizon=2)
    rate = result.response_paths("rate", "monetary", horizon=0)
    assert (output < 0).all()
    assert (rate[:, 0] > 0).all()


def test_later_horizons_can_only_shrink_the_set(var):
    """With the same seed the draws are identical, so every draw accepted
    under the longer restriction is accepted under the impact one."""
    impact = identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=2_000, seed=3)
    longer = identify_sign_restrictions(var, SLOW_OUTPUT, SHOCKS, n_draws=2_000, seed=3)
    assert longer.n_accepted <= impact.n_accepted
    impact_draws = {candidate.tobytes() for candidate in impact.accepted}
    assert all(candidate.tobytes() in impact_draws for candidate in longer.accepted)


def test_horizons_are_recorded_in_the_ledger(var):
    payload = identify_sign_restrictions(var, SLOW_OUTPUT, SHOCKS, n_draws=500).to_ledger_dict()
    assert payload["restrictions"][1]["horizons"] == [0, 1, 2]


def test_sign_result_serialises_for_the_ledger(var):
    payload = identify_sign_restrictions(
        var, CONTRACTIONARY, SHOCKS, n_draws=500
    ).to_ledger_dict()
    assert payload["scheme"] == "sign_restrictions"
    assert payload["n_accepted"] > 0
    assert len(payload["restrictions"]) == 2


# --- external --------------------------------------------------------------

def test_external_impact_is_accepted(var):
    valid = np.linalg.cholesky(var.sigma_u)
    model = identify_external(var, valid, SHOCKS, "long-run neutrality")
    assert model.scheme is IdentificationScheme.EXTERNAL
    assert model.reconstruction_error() < 1e-10


def test_external_records_the_assumption(var):
    valid = np.linalg.cholesky(var.sigma_u)
    model = identify_external(var, valid, SHOCKS, "narrative identification, Romer dates")
    assert "Romer" in model.assumptions


def test_external_rejects_an_invalid_decomposition(var):
    """B that does not reproduce Sigma is not an identification of this VAR."""
    with pytest.raises(EngineError, match="does not reproduce"):
        identify_external(var, np.eye(3), SHOCKS, "wrong")


def test_external_rejects_a_wrong_shape(var):
    with pytest.raises(EngineError, match="expected"):
        identify_external(var, np.eye(2), SHOCKS, "wrong")


def test_external_rejects_wrong_shock_name_count(var):
    valid = np.linalg.cholesky(var.sigma_u)
    with pytest.raises(EngineError, match="shock names for"):
        identify_external(var, valid, ("a", "b"), "wrong")


# --- immutability ----------------------------------------------------------

def test_model_is_frozen(var):
    model = identify_cholesky(var)
    with pytest.raises(Exception):
        model.scheme = IdentificationScheme.EXTERNAL  # type: ignore[misc]


def test_identification_does_not_mutate_the_var(var):
    before = var.sigma_u.copy()
    identify_cholesky(var, ordering=tuple(reversed(NAMES)))
    identify_sign_restrictions(var, CONTRACTIONARY, SHOCKS, n_draws=200)
    assert np.array_equal(var.sigma_u, before)


# --- external instrument (proxy SVAR), F10 ------------------------------------


class _ProxyWorld(NamedTuple):
    var: object
    instrument: dict
    true_impact: np.ndarray
    data: np.ndarray
    periods: tuple


def _proxy_world(noise: float = 0.5, n: int = 600, seed: int = 7) -> _ProxyWorld:
    """A VAR(1) with a known impact matrix and an instrument for shock 3.

    The instrument is the true policy shock plus independent noise, so it
    is relevant and exogenous by construction and the true column is known.
    """
    from datetime import date as _date

    from moirai.engine.causal.var import estimate_var

    rng = np.random.default_rng(seed)
    true_impact = np.array([[1.0, 0.0, -0.3], [0.4, 0.8, 0.2], [0.3, 0.5, 0.9]])
    a1 = np.array([[0.5, 0.1, -0.1], [0.0, 0.4, 0.1], [0.1, 0.1, 0.6]])
    eps = rng.standard_normal((n, 3))
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = a1 @ y[t - 1] + true_impact @ eps[t]
    periods = tuple(_date(1950 + m // 12, m % 12 + 1, 1) for m in range(n))
    var = estimate_var(y, ("output", "prices", "rate"), n_lags=1, periods=periods)
    z = eps[:, 2] + noise * rng.standard_normal(n)
    instrument = {periods[t]: float(z[t]) for t in range(n)}
    return _ProxyWorld(var, instrument, true_impact, y, periods)


def test_proxy_recovers_the_true_policy_column():
    from moirai.engine.causal.identification import identify_proxy

    var, instrument, true_impact, _, _ = _proxy_world()
    model, first = identify_proxy(var, instrument, "rate")
    column = model.impact[:, var.index_of("rate")]
    assert np.allclose(column, true_impact[:, 2], atol=0.15)
    assert not first.is_weak


def test_proxy_reproduces_the_covariance():
    from moirai.engine.causal.identification import identify_proxy

    var, instrument, _, _, _ = _proxy_world()
    model, _ = identify_proxy(var, instrument, "rate")
    assert model.reconstruction_error() < 1e-10


def test_proxy_names_only_the_policy_shock():
    from moirai.engine.causal.identification import IdentificationScheme, identify_proxy

    var, instrument, _, _, _ = _proxy_world()
    model, _ = identify_proxy(var, instrument, "rate")
    assert model.scheme is IdentificationScheme.PROXY
    assert model.shock_names[var.index_of("rate")] == "rate_shock"
    assert all(name.startswith("unidentified_") for name in model.shock_names[:2])


def test_the_policy_variable_rises_on_impact():
    from moirai.engine.causal.identification import identify_proxy

    var, instrument, _, _, _ = _proxy_world()
    flipped = {d: -v for d, v in instrument.items()}
    model, _ = identify_proxy(var, flipped, "rate")
    assert model.impact[var.index_of("rate"), var.index_of("rate")] > 0


def test_proxy_does_not_depend_on_the_variable_order():
    """What Cholesky cannot offer: reordering the VAR's variables leaves the
    identified column unchanged, because no ordering is assumed."""
    from moirai.engine.causal.identification import identify_proxy
    from moirai.engine.causal.var import estimate_var

    world = _proxy_world()
    var, instrument, y, periods = world.var, world.instrument, world.data, world.periods
    model, _ = identify_proxy(var, instrument, "rate")
    reordered = estimate_var(
        y[:, [2, 0, 1]], ("rate", "output", "prices"), n_lags=1, periods=periods
    )
    other, _ = identify_proxy(reordered, instrument, "rate")
    original_column = model.impact[:, var.index_of("rate")]
    reordered_column = other.impact[:, reordered.index_of("rate")]
    assert np.allclose(reordered_column[[1, 2, 0]], original_column, atol=1e-10)


def test_a_pure_noise_instrument_is_flagged_weak():
    from moirai.engine.causal.identification import identify_proxy

    var, instrument, _, _, _ = _proxy_world()
    rng = np.random.default_rng(99)
    noise = {d: float(rng.standard_normal()) for d in instrument}
    _, first = identify_proxy(var, noise, "rate")
    assert first.is_weak


def test_proxy_refuses_what_it_cannot_do():
    from moirai.engine.causal.identification import identify_proxy

    var, instrument, _, _, _ = _proxy_world()
    few = dict(list(instrument.items())[:10])
    with pytest.raises(EngineError):
        identify_proxy(var, few, "rate")
    constant = {d: 1.0 for d in instrument}
    with pytest.raises(EngineError):
        identify_proxy(var, constant, "rate")
    with pytest.raises(EngineError):
        identify_proxy(var, instrument, "not_a_variable")
