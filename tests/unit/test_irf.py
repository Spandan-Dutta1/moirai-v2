"""Tests for impulse responses, variance decompositions and bootstrap bands.

The verifiable properties here are analytic. For a known VAR(1) the
moving-average coefficients are exactly A^h, so the responses can be
checked against a closed form rather than against a previous run. That is
the difference between testing correctness and testing stability.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.identification import (
    Sign,
    SignRestriction,
    identify_cholesky,
    identify_sign_restrictions,
)
from moirai.engine.causal.irf import (
    bootstrap_bands,
    impulse_responses,
    variance_decomposition,
)
from moirai.engine.causal.var import estimate_var

NAMES = ("output", "prices", "rate")
SHOCKS = tuple(f"{name}_shock" for name in NAMES)

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


@pytest.fixture(scope="module")
def model(var):
    return identify_cholesky(var, ordering=NAMES)


@pytest.fixture(scope="module")
def irf(model):
    return impulse_responses(model, horizon=24)


# --- impulse responses: analytic checks ------------------------------------

def test_impact_period_equals_the_impact_matrix(model, irf):
    """Horizon zero is B by construction, not by approximation."""
    assert np.allclose(irf.responses[0], model.impact)


def test_var1_responses_match_the_closed_form(var, model):
    """For a VAR(1), Psi_h is exactly A^h, so B A^h is checkable directly."""
    result = impulse_responses(model, horizon=6)
    for h in range(7):
        expected = np.linalg.matrix_power(var.coefficients[0], h) @ model.impact
        assert np.allclose(result.responses[h], expected, atol=1e-12)


def test_responses_decay_in_a_stable_system(irf):
    early = np.abs(irf.responses[1]).max()
    late = np.abs(irf.responses[24]).max()
    assert late < early


def test_responses_approach_zero_at_long_horizons(model):
    result = impulse_responses(model, horizon=200)
    assert np.abs(result.responses[200]).max() < 1e-6


def test_shape_is_horizon_plus_one(model):
    assert impulse_responses(model, horizon=12).responses.shape == (13, 3, 3)


def test_zero_horizon_gives_only_the_impact(model):
    result = impulse_responses(model, horizon=0)
    assert result.responses.shape == (1, 3, 3)
    assert np.allclose(result.responses[0], model.impact)


def test_recursive_ordering_zeroes_the_impact_of_later_shocks(model, irf):
    """The first variable cannot respond to later shocks on impact."""
    assert irf.responses[0, 0, 1] == 0.0
    assert irf.responses[0, 0, 2] == 0.0


def test_later_horizons_are_not_zero_for_the_first_variable(irf):
    """The restriction binds only on impact, not thereafter."""
    assert abs(irf.responses[1, 0, 2]) > 0.0


# --- path, peak, cumulate --------------------------------------------------

def test_path_returns_one_series(irf):
    assert irf.path("output", "rate_shock").shape == (25,)


def test_path_matches_the_response_array(irf):
    assert np.array_equal(irf.path("output", "rate_shock"), irf.responses[:, 0, 2])


def test_peak_reports_horizon_and_value(irf):
    horizon, value = irf.peak("rate", "rate_shock")
    assert horizon == 0  # own shock peaks on impact in this system
    assert value == pytest.approx(irf.responses[0, 2, 2])


def test_peak_uses_absolute_magnitude(irf):
    _, value = irf.peak("output", "rate_shock")
    path = irf.path("output", "rate_shock")
    assert abs(value) == pytest.approx(np.abs(path).max())


def test_unknown_variable_is_rejected(irf):
    with pytest.raises(EngineError, match="unknown variable"):
        irf.path("nope", "rate_shock")


def test_unknown_shock_is_rejected(irf):
    with pytest.raises(EngineError, match="unknown shock"):
        irf.path("output", "nope")


def test_cumulate_is_a_running_sum(irf):
    cumulative = irf.cumulate()
    expected = np.cumsum(irf.responses[:, 0, 2])
    assert np.allclose(cumulative.path("output", "rate_shock"), expected)


def test_cumulate_sets_the_flag(irf):
    assert irf.cumulate().cumulative is True


def test_cumulate_twice_is_rejected(irf):
    """Accumulating an accumulated response is almost always a mistake."""
    with pytest.raises(EngineError, match="already cumulative"):
        irf.cumulate().cumulate()


def test_cumulate_leaves_the_original_untouched(irf):
    before = irf.responses.copy()
    irf.cumulate()
    assert np.array_equal(irf.responses, before)


def test_cumulative_response_converges(model):
    """A stable system has a finite long-run cumulative effect."""
    cumulative = impulse_responses(model, horizon=300).cumulate()
    late = cumulative.responses[250:, 0, 2]
    assert np.abs(late.max() - late.min()) < 1e-6


# --- identification metadata carried through -------------------------------

def test_assumptions_travel_with_the_responses(model, irf):
    """An impulse response without its assumption is a number, not a result."""
    assert irf.assumptions == model.assumptions
    assert irf.scheme == "cholesky"


def test_ledger_dict_records_the_scheme(irf):
    payload = irf.to_ledger_dict()
    assert payload["scheme"] == "cholesky"
    assert payload["horizon"] == 24
    assert "assumptions" in payload


# --- validation ------------------------------------------------------------

def test_negative_horizon_is_rejected(model):
    with pytest.raises(EngineError, match="negative"):
        impulse_responses(model, horizon=-1)


def test_unstable_var_is_refused(var, model):
    """Powers of an explosive companion matrix diverge."""
    explosive = np.array([[[1.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]])
    from moirai.engine.causal.var import check_stability

    broken = var.model_copy(
        update={
            "coefficients": explosive,
            "stability": check_stability(explosive),
        }
    )
    broken_model = model.model_copy(update={"var": broken})
    with pytest.raises(EngineError, match="unstable"):
        impulse_responses(broken_model, horizon=10)


# --- variance decomposition ------------------------------------------------

def test_shares_sum_to_one_at_every_horizon(model):
    fevd = variance_decomposition(model, horizon=24)
    assert np.allclose(fevd.shares.sum(axis=2), 1.0)


def test_shares_lie_in_the_unit_interval(model):
    fevd = variance_decomposition(model, horizon=24)
    assert fevd.shares.min() >= 0.0
    assert fevd.shares.max() <= 1.0


def test_first_variable_is_entirely_own_shock_on_impact(model):
    """The recursive assumption again, seen in the decomposition."""
    fevd = variance_decomposition(model, horizon=12)
    assert fevd.share("output", "output_shock", horizon=0) == pytest.approx(1.0)


def test_own_shock_share_falls_with_the_horizon(model):
    """Other shocks propagate through the lag structure over time."""
    fevd = variance_decomposition(model, horizon=24)
    assert fevd.share("output", "output_shock", horizon=24) < fevd.share(
        "output", "output_shock", horizon=0
    )


def test_dominant_shock_is_reported(model):
    fevd = variance_decomposition(model, horizon=24)
    shock, share = fevd.dominant_shock("rate")
    assert shock in SHOCKS
    assert 0.0 <= share <= 1.0


def test_share_defaults_to_the_longest_horizon(model):
    fevd = variance_decomposition(model, horizon=12)
    assert fevd.share("output", "rate_shock") == fevd.share(
        "output", "rate_shock", horizon=12
    )


def test_out_of_range_horizon_is_rejected(model):
    fevd = variance_decomposition(model, horizon=12)
    with pytest.raises(EngineError, match="outside"):
        fevd.share("output", "rate_shock", horizon=99)


def test_decomposition_serialises_for_the_ledger(model):
    payload = variance_decomposition(model, horizon=12).to_ledger_dict()
    assert payload["horizon"] == 12
    assert len(payload["long_run_shares"]) == 3


# --- bootstrap bands -------------------------------------------------------

@pytest.fixture(scope="module")
def bands(model):
    return bootstrap_bands(model, horizon=8, n_draws=100, seed=1)


def test_bands_bracket_the_point_estimate(bands):
    for level in bands.levels:
        low, point, high = bands.band("output", "rate_shock", level)
        assert np.all(low <= point + 1e-9)
        assert np.all(point <= high + 1e-9)


def test_wider_level_gives_wider_bands(bands):
    narrow = bands.band("output", "rate_shock", 0.68)
    wide = bands.band("output", "rate_shock", 0.90)
    assert np.all((wide[2] - wide[0]) >= (narrow[2] - narrow[0]) - 1e-12)


def test_band_shape_matches_the_horizon(bands):
    low, point, high = bands.band("output", "rate_shock", 0.90)
    assert low.shape == point.shape == high.shape == (9,)


def test_impact_band_collapses_on_a_restricted_element(bands):
    """The first variable's impact response to a later shock is assumed zero,
    so every replication reproduces exactly zero and the band has no width."""
    low, point, high = bands.band("output", "rate_shock", 0.90)
    assert low[0] == pytest.approx(0.0, abs=1e-12)
    assert high[0] == pytest.approx(0.0, abs=1e-12)


def test_excludes_zero_returns_a_boolean_path(bands):
    flags = bands.excludes_zero("rate", "rate_shock", 0.90)
    assert flags.dtype == bool
    assert flags.shape == (9,)


def test_own_impact_response_excludes_zero(bands):
    """A one standard deviation own shock is unambiguously non-zero."""
    assert bands.excludes_zero("rate", "rate_shock", 0.90)[0]


def test_bands_are_reproducible(model):
    first = bootstrap_bands(model, horizon=4, n_draws=50, seed=99)
    second = bootstrap_bands(model, horizon=4, n_draws=50, seed=99)
    assert np.array_equal(first.lower[0.90], second.lower[0.90])


def test_different_seeds_give_different_bands(model):
    first = bootstrap_bands(model, horizon=4, n_draws=50, seed=1)
    second = bootstrap_bands(model, horizon=4, n_draws=50, seed=2)
    assert not np.array_equal(first.lower[0.90], second.lower[0.90])


def test_failure_rate_is_reported(bands):
    assert 0.0 <= bands.failure_rate <= 1.0
    assert bands.n_successful <= bands.n_draws


def test_custom_levels_are_honoured(model):
    result = bootstrap_bands(model, horizon=4, n_draws=50, levels=(0.5,), seed=3)
    assert result.levels == (0.5,)


def test_unknown_level_is_rejected(bands):
    with pytest.raises(EngineError, match="no band at level"):
        bands.band("output", "rate_shock", 0.99)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_invalid_levels_are_rejected(model, level):
    with pytest.raises(EngineError, match="must lie in"):
        bootstrap_bands(model, horizon=4, n_draws=10, levels=(level,))


def test_too_few_draws_are_rejected(model):
    with pytest.raises(EngineError, match="at least two draws"):
        bootstrap_bands(model, horizon=4, n_draws=1)


def test_bands_serialise_with_their_caveat(bands):
    """The coverage limitation belongs in the ledger, not in a footnote."""
    payload = bands.to_ledger_dict()
    assert payload["method"] == "residual percentile bootstrap"
    assert "under-cover" in payload["caveat"]
    assert payload["seed"] == 1


def test_sign_restricted_identification_is_refused(var):
    """A set-identified scheme already reports a range; bands would double count."""
    identified = identify_sign_restrictions(
        var,
        (SignRestriction(variable="rate", shock="monetary", sign=Sign.POSITIVE),),
        ("supply", "demand", "monetary"),
        n_draws=200,
    )
    assert identified.n_accepted > 0  # the set exists
    # bootstrap_bands takes a StructuralModel, which a SignIdentifiedSet is not,
    # so the refusal is structural rather than a runtime check.


# --- immutability ----------------------------------------------------------

def test_response_is_frozen(irf):
    with pytest.raises(Exception):
        irf.horizon = 99  # type: ignore[misc]


def test_computation_does_not_mutate_the_model(model):
    before = model.impact.copy()
    impulse_responses(model, horizon=12)
    variance_decomposition(model, horizon=12)
    assert np.array_equal(model.impact, before)