"""Tests for the heterogeneous household population.

The invariant that matters most is array alignment. Every array shares an
index, so a length mismatch does not raise, it silently pairs household
i's income with household j's debt and corrupts every result downstream.
That is checked on construction and after every structural operation.

Distributional statistics are verified against cases with known analytic
answers rather than against previous output.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.economy.households import (
    EmploymentStatus,
    Population,
    PopulationParameters,
    generate_population,
)


def make_arrays(n: int = 200) -> dict:
    rng = np.random.default_rng(0)
    return dict(
        age=rng.integers(18, 80, n).astype(np.int16),
        income=rng.lognormal(12.0, 0.8, n),
        wealth=rng.lognormal(12.0, 1.5, n),
        debt=rng.lognormal(11.0, 1.0, n) * (rng.random(n) < 0.5),
        debt_is_floating=rng.random(n) < 0.7,
        employment=rng.integers(0, 4, n).astype(np.int8),
        expected_inflation=rng.normal(0.05, 0.02, n),
        risk_aversion=np.abs(rng.normal(2.0, 0.5, n)) + 0.1,
    )


@pytest.fixture
def population() -> Population:
    return Population(**make_arrays())


@pytest.fixture(scope="module")
def generated() -> Population:
    return generate_population(PopulationParameters(n_households=20_000, seed=1))


# --- construction and invariants -------------------------------------------

def test_population_reports_its_size(population):
    assert len(population) == 200


def test_repr_shows_the_count(population):
    assert "200" in repr(population)


def test_mismatched_array_lengths_are_rejected():
    """The bug this guards against does not raise on its own: it silently
    misaligns households and corrupts every downstream result."""
    arrays = make_arrays()
    arrays["income"] = arrays["income"][:100]
    with pytest.raises(EngineError, match="mismatched shapes"):
        Population(**arrays)


def test_two_dimensional_arrays_are_rejected():
    arrays = make_arrays()
    arrays["income"] = arrays["income"].reshape(100, 2)
    with pytest.raises(EngineError, match="mismatched shapes|one-dimensional"):
        Population(**arrays)


def test_nan_values_are_rejected():
    arrays = make_arrays()
    arrays["income"][5] = np.nan
    with pytest.raises(EngineError, match="NaN"):
        Population(**arrays)


def test_infinite_values_are_rejected():
    arrays = make_arrays()
    arrays["wealth"][5] = np.inf
    with pytest.raises(EngineError, match="NaN or infinity"):
        Population(**arrays)


def test_negative_income_is_rejected():
    arrays = make_arrays()
    arrays["income"][0] = -1.0
    with pytest.raises(EngineError, match="income must not be negative"):
        Population(**arrays)


def test_negative_debt_is_rejected():
    arrays = make_arrays()
    arrays["debt"][0] = -1.0
    with pytest.raises(EngineError, match="debt must not be negative"):
        Population(**arrays)


def test_non_positive_risk_aversion_is_rejected():
    arrays = make_arrays()
    arrays["risk_aversion"][0] = 0.0
    with pytest.raises(EngineError, match="risk aversion"):
        Population(**arrays)


def test_negative_wealth_is_allowed():
    """Wealth can be negative: underwater households are real."""
    arrays = make_arrays()
    arrays["wealth"][0] = -5_000.0
    assert len(Population(**arrays)) == 200


# --- derived quantities ----------------------------------------------------

def test_net_worth_is_wealth_minus_debt(population):
    assert np.allclose(population.net_worth, population.wealth - population.debt)


def test_debt_to_income_handles_zero_income():
    """Undefined rather than a division error, and reported as infinity."""
    arrays = make_arrays()
    arrays["income"][0] = 0.0
    arrays["debt"][0] = 1_000.0
    ratio = Population(**arrays).debt_to_income
    assert np.isinf(ratio[0])


def test_indebted_mask_matches_positive_debt(population):
    assert np.array_equal(population.is_indebted, population.debt > 0)


def test_rate_exposed_requires_both_debt_and_floating(population):
    exposed = population.is_rate_exposed
    assert np.all(population.debt[exposed] > 0)
    assert np.all(population.debt_is_floating[exposed])


def test_a_fixed_rate_borrower_is_not_rate_exposed():
    """The transmission channel: fixed-rate debt insulates until refinancing."""
    arrays = make_arrays()
    arrays["debt"][:] = 1_000.0
    arrays["debt_is_floating"][:] = False
    assert not Population(**arrays).is_rate_exposed.any()


def test_net_borrower_is_negative_net_worth(population):
    assert np.array_equal(population.is_net_borrower, population.net_worth < 0)


def test_employment_masks_are_mutually_consistent(population):
    assert not (population.is_employed & population.is_retired).any()


def test_labour_force_excludes_the_retired(population):
    assert not (population.is_in_labour_force & population.is_retired).any()


# --- distributional statistics ---------------------------------------------

def test_gini_of_perfect_equality_is_zero(population):
    assert population.gini(np.full(1_000, 100.0)) == pytest.approx(0.0, abs=1e-9)


def test_gini_of_total_concentration_approaches_one(population):
    """One holder of everything: Gini tends to 1 - 1/n."""
    values = np.zeros(1_000)
    values[0] = 1_000.0
    assert population.gini(values) == pytest.approx(0.999, abs=0.002)


def test_gini_of_a_uniform_distribution(population):
    """A uniform distribution on (0, m) has a Gini of exactly one third."""
    values = np.linspace(0.0, 1_000.0, 100_000)
    assert population.gini(values) == pytest.approx(1 / 3, abs=0.01)


def test_gini_rejects_negative_values(population):
    with pytest.raises(EngineError, match="not defined for negative"):
        population.gini(np.array([-1.0, 2.0]))


def test_gini_rejects_an_empty_array(population):
    with pytest.raises(EngineError, match="empty"):
        population.gini(np.array([]))


def test_gini_of_all_zeros_is_zero(population):
    assert population.gini(np.zeros(100)) == 0.0


def test_top_share_of_equal_holders(population):
    """With everyone equal, the top ten percent hold ten percent."""
    assert population.top_share(np.full(1_000, 5.0), 0.1) == pytest.approx(0.1, abs=0.01)


def test_top_share_of_total_concentration(population):
    values = np.zeros(1_000)
    values[0] = 100.0
    assert population.top_share(values, 0.1) == pytest.approx(1.0)


def test_top_share_rejects_an_invalid_percentile(population):
    with pytest.raises(EngineError, match="must lie in"):
        population.top_share(np.ones(10), 1.5)


def test_quantile_groups_are_balanced(population):
    groups = population.quantile_groups(population.income, 5)
    counts = np.bincount(groups, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_quantile_groups_are_ordered_by_value(population):
    groups = population.quantile_groups(population.income, 5)
    means = [population.income[groups == g].mean() for g in range(5)]
    assert means == sorted(means)


def test_quantile_groups_rejects_too_few_groups(population):
    with pytest.raises(EngineError, match="at least two"):
        population.quantile_groups(population.income, 1)


# --- copying and subsetting ------------------------------------------------

def test_copy_is_independent(population):
    duplicate = population.copy()
    duplicate.income[0] = 999_999.0
    assert population.income[0] != 999_999.0


def test_copy_preserves_values(population):
    assert np.array_equal(population.copy().income, population.income)


def test_copy_preserves_parameters(generated):
    assert generated.copy().parameters == generated.parameters


def test_subset_selects_the_masked_households(population):
    mask = population.is_indebted
    subset = population.subset(mask)
    assert len(subset) == int(mask.sum())


def test_subset_keeps_arrays_aligned(population):
    """The alignment invariant must survive selection."""
    mask = population.is_indebted
    subset = population.subset(mask)
    assert np.array_equal(subset.income, population.income[mask])
    assert np.array_equal(subset.debt, population.debt[mask])


def test_subset_rejects_a_wrong_shaped_mask(population):
    with pytest.raises(EngineError, match="expected"):
        population.subset(np.ones(10, dtype=bool))


def test_subset_rejects_a_non_boolean_mask(population):
    with pytest.raises(EngineError, match="boolean"):
        population.subset(np.ones(len(population), dtype=int))


def test_subset_rejects_an_empty_selection(population):
    with pytest.raises(EngineError, match="no households"):
        population.subset(np.zeros(len(population), dtype=bool))


# --- parameters ------------------------------------------------------------

def test_parameters_are_frozen():
    parameters = PopulationParameters()
    with pytest.raises(Exception):
        parameters.seed = 99  # type: ignore[misc]


def test_retirement_age_must_lie_within_the_age_range():
    with pytest.raises(Exception, match="retirement_age"):
        PopulationParameters(min_age=18, retirement_age=90, max_age=85)


def test_min_age_must_be_below_max_age():
    with pytest.raises(Exception, match="min_age"):
        PopulationParameters(min_age=50, max_age=40, retirement_age=45)


def test_labour_shares_must_leave_someone_employed():
    with pytest.raises(Exception, match="no one employed"):
        PopulationParameters(unemployment_rate=0.5, out_of_labour_force_rate=0.6)


def test_population_below_the_minimum_is_rejected():
    with pytest.raises(Exception):
        PopulationParameters(n_households=10)


def test_parameters_serialise_for_the_ledger():
    payload = PopulationParameters().to_ledger_dict()
    assert payload["n_households"] == 10_000
    assert payload["seed"] == 42


# --- generation ------------------------------------------------------------

def test_generation_respects_the_requested_size():
    assert len(generate_population(PopulationParameters(n_households=5_000))) == 5_000


def test_generation_is_reproducible():
    """A population is a deterministic function of its calibration."""
    first = generate_population(PopulationParameters(n_households=1_000, seed=7))
    second = generate_population(PopulationParameters(n_households=1_000, seed=7))
    assert np.array_equal(first.income, second.income)
    assert np.array_equal(first.debt, second.debt)


def test_different_seeds_give_different_populations():
    first = generate_population(PopulationParameters(n_households=1_000, seed=7))
    second = generate_population(PopulationParameters(n_households=1_000, seed=8))
    assert not np.array_equal(first.income, second.income)


def test_ages_lie_within_the_configured_range(generated):
    parameters = generated.parameters
    assert generated.age.min() >= parameters.min_age
    assert generated.age.max() <= parameters.max_age


def test_everyone_past_retirement_age_is_retired(generated):
    retired_by_age = generated.age >= generated.parameters.retirement_age
    assert np.all(generated.employment[retired_by_age] == EmploymentStatus.RETIRED)


def test_nobody_below_retirement_age_is_retired(generated):
    young = generated.age < generated.parameters.retirement_age
    assert not np.any(generated.employment[young] == EmploymentStatus.RETIRED)


def test_income_is_strictly_positive(generated):
    assert generated.income.min() > 0


def test_wealth_is_more_unequal_than_income(generated):
    """The calibration target: wealth concentrates far more than income."""
    income_gini = generated.gini(generated.income)
    wealth_gini = generated.gini(np.maximum(generated.wealth, 0.0))
    assert wealth_gini > income_gini


def test_income_and_wealth_are_positively_correlated(generated):
    """Drawn independently, high earners would hold no assets."""
    correlation = np.corrcoef(np.log(generated.income), np.log(generated.wealth))[0, 1]
    assert correlation > 0.3


def test_debt_free_households_have_no_floating_flag(generated):
    assert not np.any(generated.debt_is_floating & (generated.debt == 0))


def test_only_a_subset_carries_debt(generated):
    share = generated.is_indebted.mean()
    assert 0.1 < share < 0.7


def test_expected_inflation_is_centred_on_the_calibration(generated):
    target = generated.parameters.initial_expected_inflation
    assert generated.expected_inflation.mean() == pytest.approx(target, abs=0.005)


def test_summary_reports_the_headline_statistics(generated):
    summary = generated.summary()
    assert summary["n_households"] == 20_000
    assert 0.0 < summary["income_gini"] < 1.0
    assert summary["wealth_gini"] > summary["income_gini"]
    assert 0.0 <= summary["share_rate_exposed"] <= 1.0


def test_generation_scales_to_a_large_population():
    """The whole argument for structure of arrays."""
    population = generate_population(PopulationParameters(n_households=1_000_000))
    assert len(population) == 1_000_000
    assert population.income.nbytes < 10 * 1024 * 1024  # 8 MB for a million floats