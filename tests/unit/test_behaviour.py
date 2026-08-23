"""Tests for household behaviour rules.

These test *mechanisms*, not magnitudes. Whether a rate rise cuts
consumption by 0.3 percent or 3 percent depends on calibration that has
not yet been validated against survey data. Whether a floating-rate
borrower's debt service rises when rates rise does not: that is the model
being internally coherent, and it must hold at any calibration.

Testing magnitudes here would lock in provisional parameters as though
they were findings. Moment matching against real targets belongs in a
calibration module, not in unit tests.
"""

import numpy as np
import pytest

from moirai.engine.economy.behaviour import (
    BehaviourParameters,
    bank_rates,
    counterfactual,
    debt_service,
    interest_income,
    job_loss_probability,
    marginal_propensity_to_consume,
    simulate,
    step,
    update_expectations,
)
from moirai.engine.economy.households import (
    EmploymentStatus,
    Population,
    PopulationParameters,
    generate_population,
)
from moirai.engine.economy.shock_path import MacroVariable, ShockPath
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM

BASELINE_RATE = 0.065
BASELINE_INFLATION = 0.05
BASELINE_GROWTH = 0.06

#: Household-facing lending rate at the baseline policy rate, using the
#: flat fallback spread. Tests of the debt channel work in lending rates
#: now, because that is what a household actually faces.
BASELINE_LENDING = BASELINE_RATE + 0.03


def make_path(
    horizon: int = 12,
    *,
    rate_deviation: float = 0.0,
    inflation_deviation: float = 0.0,
    growth_deviation: float = 0.0,
) -> ShockPath:
    """A flat shock path, so a test isolates one channel at a time."""
    n = horizon + 1
    return ShockPath(
        shock_name="test_shock",
        horizon=horizon,
        paths={
            MacroVariable.POLICY_RATE: np.full(n, BASELINE_RATE + rate_deviation),
            MacroVariable.INFLATION: np.full(n, BASELINE_INFLATION + inflation_deviation),
            MacroVariable.INCOME_GROWTH: np.full(n, BASELINE_GROWTH + growth_deviation),
        },
        baselines={
            MacroVariable.POLICY_RATE: BASELINE_RATE,
            MacroVariable.INFLATION: BASELINE_INFLATION,
            MacroVariable.INCOME_GROWTH: BASELINE_GROWTH,
        },
    )


@pytest.fixture(scope="module")
def population() -> Population:
    return generate_population(PopulationParameters(n_households=5_000, seed=1))


@pytest.fixture
def parameters() -> BehaviourParameters:
    return BehaviourParameters()


def two_households(
    *,
    income=(100_000.0, 100_000.0),
    wealth=(0.0, 0.0),
    debt=(0.0, 0.0),
    floating=(False, False),
    employment=(EmploymentStatus.EMPLOYED, EmploymentStatus.EMPLOYED),
) -> Population:
    """A minimal population where every difference is deliberate."""
    return Population(
        age=np.array([40, 40], dtype=np.int16),
        income=np.array(income, dtype=float),
        wealth=np.array(wealth, dtype=float),
        debt=np.array(debt, dtype=float),
        debt_is_floating=np.array(floating, dtype=bool),
        employment=np.array(employment, dtype=np.int8),
        expected_inflation=np.array([0.05, 0.05]),
        risk_aversion=np.array([2.0, 2.0]),
    )


# --- marginal propensity to consume ----------------------------------------

def test_mpc_falls_with_liquid_wealth(parameters):
    """The empirical regularity the whole distributional result rests on."""
    people = two_households(wealth=(0.0, 10_000_000.0))
    mpc = marginal_propensity_to_consume(people, parameters)
    assert mpc[0] > mpc[1]


def test_mpc_is_bounded_by_the_calibration(population, parameters):
    mpc = marginal_propensity_to_consume(population, parameters)
    assert mpc.min() >= parameters.mpc_high_wealth - 1e-9
    assert mpc.max() <= parameters.mpc_low_wealth + 1e-9


def test_zero_wealth_gives_the_hand_to_mouth_mpc(parameters):
    people = two_households(wealth=(0.0, 0.0))
    mpc = marginal_propensity_to_consume(people, parameters)
    assert mpc[0] == pytest.approx(parameters.mpc_low_wealth, abs=1e-9)


def test_mpc_is_scale_free_in_income(parameters):
    """Wealth is measured in months of income, so doubling both is neutral."""
    poor = two_households(income=(50_000.0, 50_000.0), wealth=(25_000.0, 25_000.0))
    rich = two_households(income=(500_000.0, 500_000.0), wealth=(250_000.0, 250_000.0))
    assert marginal_propensity_to_consume(poor, BehaviourParameters())[0] == pytest.approx(
        marginal_propensity_to_consume(rich, BehaviourParameters())[0]
    )


def test_negative_wealth_does_not_break_the_mpc(parameters):
    people = two_households(wealth=(-50_000.0, 0.0))
    mpc = marginal_propensity_to_consume(people, parameters)
    assert np.all(np.isfinite(mpc))


# --- debt service ----------------------------------------------------------

def test_floating_borrower_pays_more_when_rates_rise(parameters):
    people = two_households(debt=(500_000.0, 500_000.0), floating=(True, True))
    low = debt_service(
        people, BASELINE_LENDING, parameters, baseline_lending_rate=BASELINE_LENDING
    )
    high = debt_service(
        people,
        BASELINE_LENDING + 0.02,
        parameters,
        baseline_lending_rate=BASELINE_LENDING,
    )
    assert high[0] > low[0]


def test_fixed_borrower_is_insulated(parameters):
    """The transmission channel: fixed-rate debt does not reprice."""
    people = two_households(debt=(500_000.0, 500_000.0), floating=(False, False))
    low = debt_service(
        people, BASELINE_LENDING, parameters, baseline_lending_rate=BASELINE_LENDING
    )
    high = debt_service(
        people,
        BASELINE_LENDING + 0.02,
        parameters,
        baseline_lending_rate=BASELINE_LENDING,
    )
    assert np.allclose(low, high)


def test_only_the_floating_borrower_is_affected(parameters):
    people = two_households(debt=(500_000.0, 500_000.0), floating=(True, False))
    low = debt_service(
        people, BASELINE_LENDING, parameters, baseline_lending_rate=BASELINE_LENDING
    )
    high = debt_service(
        people,
        BASELINE_LENDING + 0.02,
        parameters,
        baseline_lending_rate=BASELINE_LENDING,
    )
    assert high[0] > low[0]
    assert high[1] == pytest.approx(low[1])


def test_debt_free_households_pay_nothing(parameters):
    people = two_households(debt=(0.0, 0.0))
    payments = debt_service(
        people, 0.10, parameters, baseline_lending_rate=BASELINE_LENDING
    )
    assert np.allclose(payments, 0.0)


def test_debt_service_scales_with_the_balance(parameters):
    people = two_households(debt=(100_000.0, 200_000.0), floating=(True, True))
    payments = debt_service(
        people, BASELINE_LENDING, parameters, baseline_lending_rate=BASELINE_LENDING
    )
    assert payments[1] == pytest.approx(2 * payments[0])


def test_zero_rate_does_not_divide_by_zero(parameters):
    """The annuity formula is singular at zero and must be handled."""
    people = two_households(debt=(500_000.0, 500_000.0), floating=(True, True))
    payments = debt_service(people, 0.0, parameters, baseline_lending_rate=0.0)
    assert np.all(np.isfinite(payments))
    assert np.all(payments > 0)


# --- the banking layer sets the rates --------------------------------------

def test_without_banks_a_flat_spread_is_used():
    """The pre-Layer-2 behaviour, kept so the household layer can run
    standalone. Visibly cruder, which is the point of having Layer 2."""
    lending, baseline_lending, deposit = bank_rates(0.06, 0.05, None)
    assert lending == pytest.approx(0.09)
    assert baseline_lending == pytest.approx(0.08)
    assert deposit == pytest.approx(0.045)


def test_banks_pass_through_less_than_one_for_one():
    """Lenders absorb part of a policy move in their margins. This used to
    be a household parameter; it now emerges from bank characteristics."""
    before, _, _ = bank_rates(0.05, 0.05, INDIAN_BANKING_SYSTEM)
    after, _, _ = bank_rates(0.06, 0.05, INDIAN_BANKING_SYSTEM)
    passed = after - before
    assert 0.0 < passed < 0.01


def test_banks_produce_a_wedge_between_lending_and_deposits():
    """The transfer: borrowers absorb more of a rise than savers receive,
    and the difference accrues to the banking system as margin."""
    lending_before, _, deposit_before = bank_rates(0.05, 0.05, INDIAN_BANKING_SYSTEM)
    lending_after, _, deposit_after = bank_rates(0.06, 0.05, INDIAN_BANKING_SYSTEM)
    assert (lending_after - lending_before) > (deposit_after - deposit_before)


def test_the_baseline_lending_rate_is_unchanged_by_a_move():
    """Fixed-rate borrowers keep paying the old rate, so the baseline must
    not move with the policy rate."""
    _, baseline_at_five, _ = bank_rates(0.05, 0.05, INDIAN_BANKING_SYSTEM)
    _, baseline_at_six, _ = bank_rates(0.06, 0.05, INDIAN_BANKING_SYSTEM)
    assert baseline_at_five == pytest.approx(baseline_at_six)


# --- interest income -------------------------------------------------------

def test_savers_earn_more_when_rates_rise():
    people = two_households(wealth=(1_000_000.0, 1_000_000.0))
    assert interest_income(people, 0.05)[0] > interest_income(people, 0.03)[0]


def test_no_wealth_means_no_interest_income():
    people = two_households(wealth=(0.0, 0.0))
    assert np.allclose(interest_income(people, 0.10), 0.0)


def test_negative_wealth_earns_nothing_rather_than_negative():
    people = two_households(wealth=(-100_000.0, 0.0))
    assert np.all(interest_income(people, 0.08) >= 0.0)


# --- job loss --------------------------------------------------------------

def test_no_job_loss_when_growth_is_at_baseline(population, parameters):
    hazard = job_loss_probability(
        population, BASELINE_GROWTH, parameters, baseline_growth=BASELINE_GROWTH
    )
    assert np.allclose(hazard, 0.0)


def test_no_job_loss_when_growth_exceeds_baseline(population, parameters):
    hazard = job_loss_probability(
        population, BASELINE_GROWTH + 0.02, parameters, baseline_growth=BASELINE_GROWTH
    )
    assert np.allclose(hazard, 0.0)


def test_weaker_growth_raises_the_hazard(population, parameters):
    mild = job_loss_probability(
        population, BASELINE_GROWTH - 0.01, parameters, baseline_growth=BASELINE_GROWTH
    )
    severe = job_loss_probability(
        population, BASELINE_GROWTH - 0.05, parameters, baseline_growth=BASELINE_GROWTH
    )
    assert severe.mean() > mild.mean()


def test_job_loss_risk_falls_with_income(population, parameters):
    """Recessions concentrate in cyclical, lower paid, less secure work.

    Compared within the labour force. Including the retired and inactive
    would compare against people whose hazard is zero by construction, and
    those are concentrated at the bottom of the income distribution because
    retirement reduces income.
    """
    hazard = job_loss_probability(
        population, BASELINE_GROWTH - 0.04, parameters, baseline_growth=BASELINE_GROWTH
    )
    active = population.is_in_labour_force
    quintile = population.quantile_groups(population.income, 5)

    low = hazard[active & (quintile == 0)]
    high = hazard[active & (quintile == 4)]
    assert low.mean() > high.mean()


def test_a_flat_gradient_removes_the_income_difference(population):
    parameters = BehaviourParameters(job_loss_income_gradient=1.0)
    hazard = job_loss_probability(
        population, BASELINE_GROWTH - 0.04, parameters, baseline_growth=BASELINE_GROWTH
    )
    active = population.is_in_labour_force
    quintile = population.quantile_groups(population.income, 5)

    low = hazard[active & (quintile == 0)]
    high = hazard[active & (quintile == 4)]
    assert low.mean() == pytest.approx(high.mean(), rel=0.05)


def test_those_outside_the_labour_force_face_no_hazard(population, parameters):
    hazard = job_loss_probability(
        population, BASELINE_GROWTH - 0.04, parameters, baseline_growth=BASELINE_GROWTH
    )
    assert np.allclose(hazard[~population.is_in_labour_force], 0.0)


def test_hazard_is_a_valid_probability(population, parameters):
    hazard = job_loss_probability(
        population, BASELINE_GROWTH - 0.20, parameters, baseline_growth=BASELINE_GROWTH
    )
    assert hazard.min() >= 0.0
    assert hazard.max() <= 1.0


# --- expectations ----------------------------------------------------------

def test_expectations_move_toward_realised_inflation(parameters):
    people = two_households()
    updated = update_expectations(people, 0.10, parameters)
    assert np.all(updated > people.expected_inflation)


def test_expectations_do_not_overshoot(parameters):
    """Partial adjustment: households move part of the way, not all of it."""
    people = two_households()
    updated = update_expectations(people, 0.10, parameters)
    assert np.all(updated < 0.10)


def test_correct_expectations_are_not_revised(parameters):
    people = two_households()
    updated = update_expectations(people, 0.05, parameters)
    assert np.allclose(updated, people.expected_inflation)


def test_a_zero_learning_rate_freezes_beliefs():
    people = two_households()
    parameters = BehaviourParameters(expectation_learning_rate=0.0)
    assert np.allclose(update_expectations(people, 0.20, parameters), 0.05)


def test_a_unit_learning_rate_adopts_the_realisation():
    people = two_households()
    parameters = BehaviourParameters(expectation_learning_rate=1.0)
    assert np.allclose(update_expectations(people, 0.20, parameters), 0.20)


def test_repeated_learning_converges(parameters):
    people = two_households()
    for _ in range(200):
        people.expected_inflation = update_expectations(people, 0.09, parameters)
    assert people.expected_inflation[0] == pytest.approx(0.09, abs=1e-4)


# --- the period step -------------------------------------------------------

def test_step_does_not_mutate_the_input(population, parameters):
    before = population.income.copy()
    rng = np.random.default_rng(0)
    step(population, 0, make_path(), parameters, rng)
    assert np.array_equal(population.income, before)


def test_step_preserves_the_population_size(population, parameters):
    rng = np.random.default_rng(0)
    updated, _ = step(population, 0, make_path(), parameters, rng)
    assert len(updated) == len(population)


def test_step_keeps_arrays_aligned(population, parameters):
    """The alignment invariant must survive a simulation step."""
    rng = np.random.default_rng(0)
    updated, _ = step(population, 0, make_path(), parameters, rng)
    updated._validate()  # raises if any array drifted


def test_consumption_is_non_negative(population, parameters):
    rng = np.random.default_rng(0)
    _, outcome = step(population, 0, make_path(), parameters, rng)
    assert outcome.consumption.min() >= 0.0


def test_consumption_does_not_exceed_disposable_income(population, parameters):
    """MPC is at most one, so nobody spends more than they have."""
    rng = np.random.default_rng(0)
    _, outcome = step(population, 0, make_path(), parameters, rng)
    assert np.all(outcome.consumption <= outcome.disposable_income + 1e-6)


def test_outcome_aggregates(population, parameters):
    rng = np.random.default_rng(0)
    _, outcome = step(population, 0, make_path(), parameters, rng)
    aggregate = outcome.aggregate()
    assert aggregate["total_consumption"] > 0
    assert aggregate["job_losses"] >= 0


def test_the_outcome_records_the_rates_faced(population, parameters):
    """A household's experience is the bank rate, not the policy rate, so
    the outcome records what it actually faced."""
    rng = np.random.default_rng(0)
    _, outcome = step(population, 0, make_path(), parameters, rng)
    assert outcome.lending_rate > outcome.deposit_rate


def test_a_period_outside_the_path_is_rejected(population, parameters):
    from moirai.core.exceptions import EngineError

    rng = np.random.default_rng(0)
    with pytest.raises(EngineError, match="outside"):
        step(population, 999, make_path(horizon=12), parameters, rng)


def test_banks_change_what_households_face(population, parameters):
    """The point of Layer 2: with a banking system the household sees a
    different rate from the flat fallback spread."""
    rng = np.random.default_rng(0)
    without = step(population, 0, make_path(rate_deviation=0.02), parameters, rng)[1]
    rng = np.random.default_rng(0)
    with_banks = step(
        population,
        0,
        make_path(rate_deviation=0.02),
        parameters,
        rng,
        INDIAN_BANKING_SYSTEM,
    )[1]
    assert without.lending_rate != with_banks.lending_rate


# --- simulation ------------------------------------------------------------

def test_simulation_runs_every_period(population, parameters):
    path = make_path(horizon=11)
    _, outcomes = simulate(population, path, parameters)
    assert len(outcomes) == 12


def test_simulation_is_reproducible(population, parameters):
    path = make_path()
    _, first = simulate(population, path, parameters)
    _, second = simulate(population, path, parameters)
    assert np.array_equal(first[0].consumption, second[0].consumption)


def test_simulation_with_banks_is_reproducible(population, parameters):
    path = make_path(rate_deviation=0.01)
    _, first = simulate(population, path, parameters, INDIAN_BANKING_SYSTEM)
    _, second = simulate(population, path, parameters, INDIAN_BANKING_SYSTEM)
    assert np.array_equal(first[0].consumption, second[0].consumption)


def test_different_seeds_give_different_paths(population):
    """A path with no shortfall produces no job losses at all, so the
    comparison needs a period where the random draw actually matters."""
    path = make_path(growth_deviation=-0.04)
    _, first = simulate(population, path, BehaviourParameters(seed=1))
    _, second = simulate(population, path, BehaviourParameters(seed=2))
    assert not np.array_equal(first[5].became_unemployed, second[5].became_unemployed)


def test_simulation_does_not_mutate_the_input(population, parameters):
    before = population.wealth.copy()
    simulate(population, make_path(), parameters)
    assert np.array_equal(population.wealth, before)


# --- directional effects, not magnitudes -----------------------------------

def test_a_rate_rise_raises_debt_service_in_aggregate(population, parameters):
    """Direction, not size: the magnitude depends on calibration."""
    _, base = simulate(population, make_path(), parameters)
    _, shocked = simulate(population, make_path(rate_deviation=0.02), parameters)

    assert sum(o.debt_service.sum() for o in shocked) > sum(
        o.debt_service.sum() for o in base
    )


def test_a_rate_rise_raises_interest_income_in_aggregate(population, parameters):
    _, base = simulate(population, make_path(), parameters)
    _, shocked = simulate(population, make_path(rate_deviation=0.02), parameters)

    assert sum(o.interest_income.sum() for o in shocked) > sum(
        o.interest_income.sum() for o in base
    )


def test_the_same_holds_through_the_banking_layer(population, parameters):
    """Banks dampen the pass-through but must not reverse the direction."""
    _, base = simulate(population, make_path(), parameters, INDIAN_BANKING_SYSTEM)
    _, shocked = simulate(
        population, make_path(rate_deviation=0.02), parameters, INDIAN_BANKING_SYSTEM
    )

    assert sum(o.debt_service.sum() for o in shocked) > sum(
        o.debt_service.sum() for o in base
    )


def test_weaker_growth_causes_more_job_losses(population, parameters):
    _, base = simulate(population, make_path(), parameters)
    _, shocked = simulate(population, make_path(growth_deviation=-0.04), parameters)

    assert sum(int(o.became_unemployed.sum()) for o in shocked) > sum(
        int(o.became_unemployed.sum()) for o in base
    )


def test_floating_borrowers_fare_worse_than_savers_under_tightening(
    population, parameters
):
    """The transfer that a representative agent averages away."""
    base, shocked = counterfactual(population, make_path(rate_deviation=0.02), parameters)

    base_total = np.sum([o.consumption for o in base], axis=0)
    shock_total = np.sum([o.consumption for o in shocked], axis=0)
    change = (shock_total - base_total) / np.maximum(base_total, 1.0)

    borrowers = population.is_rate_exposed
    savers = ~population.is_indebted
    assert change[borrowers].mean() < change[savers].mean()


# --- counterfactual --------------------------------------------------------

def test_counterfactual_returns_two_runs(population, parameters):
    base, shocked = counterfactual(population, make_path(rate_deviation=0.02), parameters)
    assert len(base) == len(shocked)


def test_a_zero_shock_makes_the_two_runs_identical(population, parameters):
    """With no deviation from baseline, the counterfactual is the shock."""
    base, shocked = counterfactual(population, make_path(), parameters)
    assert np.allclose(base[0].consumption, shocked[0].consumption)


def test_the_counterfactual_accepts_a_banking_system(population, parameters):
    base, shocked = counterfactual(
        population, make_path(rate_deviation=0.02), parameters, INDIAN_BANKING_SYSTEM
    )
    assert len(base) == len(shocked)


def test_the_baseline_holds_every_variable_at_its_baseline(population, parameters):
    base, _ = counterfactual(population, make_path(rate_deviation=0.03), parameters)
    first = base[0].debt_service.sum()
    last = base[-1].debt_service.sum()
    assert last == pytest.approx(first, rel=0.5)  # only inflation erosion differs


# --- parameters ------------------------------------------------------------

def test_parameters_are_frozen():
    parameters = BehaviourParameters()
    with pytest.raises(Exception):
        parameters.seed = 99  # type: ignore[misc]


def test_parameters_serialise_for_the_ledger():
    payload = BehaviourParameters().to_ledger_dict()
    assert payload["mpc_low_wealth"] == 0.70
    assert payload["seed"] == 42


def test_pass_through_is_no_longer_a_household_parameter():
    """It moved to the banking layer, where it is derived from bank
    characteristics rather than assumed."""
    assert not hasattr(BehaviourParameters(), "floating_pass_through")
    assert not hasattr(BehaviourParameters(), "deposit_pass_through")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mpc_low_wealth", 0.0),
        ("mpc_low_wealth", 1.5),
        ("expectation_learning_rate", 1.5),
        ("debt_maturity_years", 0.0),
        ("job_finding_rate", 0.0),
    ],
)
def test_invalid_parameters_are_rejected(field, value):
    with pytest.raises(Exception):
        BehaviourParameters(**{field: value})