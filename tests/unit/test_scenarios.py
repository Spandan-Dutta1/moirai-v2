"""Tests for named scenarios.

A scenario is a declared set of conditions run through the whole chain, so
these tests are mostly about the wiring holding: conditions reach the
banks, the game's answer reaches the shock, the shock reaches households,
and the provenance survives the trip.

The macro estimation is stubbed with a synthetic impulse response rather
than fetched. Testing against live FRED data would test the network, and
the point here is whether a scenario is carried through faithfully.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.diagnostics import (
    CheckOutcome,
    DiagnosticReport,
    Severity,
)
from moirai.engine.causal.irf import ImpulseResponse
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import (
    DEFAULT_SCENARIOS,
    Condition,
    Scenario,
    SolutionMode,
    compare,
    run_scenario,
)

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
VARIABLES = ("indpro", "cpiaucsl", "fedfunds")


def make_irf(horizon: int = 24) -> ImpulseResponse:
    """A synthetic impulse response with the shape a monetary VAR produces.

    The policy rate responds on impact and decays; output falls with a lag
    and recovers; prices drift down. Enough structure for the seam to have
    something to scale, without depending on a live fetch.
    """
    n = horizon + 1
    responses = np.zeros((n, 3, 3))
    decay = np.exp(-np.arange(n) / 8.0)

    responses[:, 2, 2] = 0.20 * decay          # rate responds to its own shock
    responses[:, 0, 2] = -0.05 * decay         # output falls
    responses[:, 1, 2] = -0.02 * decay         # prices fall
    responses[0, 0, 2] = 0.0                   # recursive ordering: no impact
    responses[0, 1, 2] = 0.0

    return ImpulseResponse(
        variables=VARIABLES,
        shock_names=tuple(f"{v}_shock" for v in VARIABLES),
        horizon=horizon,
        responses=responses,
        scheme="cholesky",
        assumptions="synthetic, for testing",
    )


def make_report(usable: bool = True) -> DiagnosticReport:
    outcome = CheckOutcome(
        name="portmanteau",
        statistic=10.0,
        pvalue=0.5 if usable else 0.001,
        null_hypothesis="residuals are serially uncorrelated",
        severity=Severity.CRITICAL,
    )
    return DiagnosticReport(
        variables=VARIABLES, n_lags=12, n_observations=250, outcomes=(outcome,)
    )


@pytest.fixture(scope="module")
def population():
    return generate_population(PopulationParameters(n_households=3_000, seed=1))


@pytest.fixture(scope="module")
def spillovers() -> SpilloverMatrix:
    return SpilloverMatrix.from_tiers(DEFAULT_TIERS)


def execute(scenario: Scenario, population, spillovers, *, usable: bool = True):
    return run_scenario(
        scenario,
        BANKS,
        spillovers,
        make_irf(),
        shock_name="fedfunds_shock",
        rate_variable="fedfunds",
        price_variable="cpiaucsl",
        output_variable="indpro",
        population=population,
        banking_system=INDIAN_BANKING_SYSTEM,
        diagnostics=make_report(usable),
        calibration_loss=3.0,
        allow_extreme=True,
    )


SIMPLE = Scenario(
    name="test_shock",
    description="a US inflation shock",
    conditions=(Condition(bank=FED.name, inflation=0.045),),
    shock_origin=FED.name,
)


# --- conditions ------------------------------------------------------------

def test_a_condition_overrides_inflation():
    adjusted = Condition(bank=FED.name, inflation=0.06).apply(FED)
    assert adjusted.current_inflation == 0.06


def test_a_condition_leaves_other_fields_alone():
    adjusted = Condition(bank=FED.name, inflation=0.06).apply(FED)
    assert adjusted.inflation_target == FED.inflation_target
    assert adjusted.current_rate == FED.current_rate


def test_an_empty_condition_changes_nothing():
    assert Condition(bank=FED.name).apply(FED) == FED


def test_conditions_reach_only_their_own_bank():
    conditioned = SIMPLE.apply_to(BANKS)
    by_name = {b.name: b for b in conditioned}
    assert by_name[FED.name].current_inflation == 0.045
    assert by_name[RBI.name].current_inflation == RBI.current_inflation


def test_the_bank_order_is_preserved():
    """Downstream code indexes by position in places, so reordering here
    would misalign the network."""
    conditioned = SIMPLE.apply_to(BANKS)
    assert [b.name for b in conditioned] == [b.name for b in BANKS]


def test_a_condition_on_an_unknown_bank_is_rejected():
    scenario = Scenario(
        name="bad",
        description="x",
        conditions=(Condition(bank="Bank of Nowhere", inflation=0.05),),
        shock_origin=FED.name,
    )
    with pytest.raises(EngineError, match="no bank named"):
        scenario.apply_to(BANKS)


# --- scenario validation ---------------------------------------------------

def test_led_mode_requires_a_leader():
    with pytest.raises(Exception, match="needs a leader"):
        Scenario(
            name="x",
            description="x",
            mode=SolutionMode.LED,
            shock_origin=FED.name,
        )


def test_a_leader_is_meaningless_when_simultaneous():
    with pytest.raises(Exception, match="meaningless"):
        Scenario(
            name="x",
            description="x",
            mode=SolutionMode.SIMULTANEOUS,
            leader=FED.name,
            shock_origin=FED.name,
        )


def test_a_scenario_is_frozen():
    with pytest.raises(Exception):
        SIMPLE.name = "renamed"  # type: ignore[misc]


def test_a_scenario_serialises_with_its_conditions():
    payload = SIMPLE.to_ledger_dict()
    assert payload["name"] == "test_shock"
    assert payload["conditions"][0]["inflation"] == 0.045


# --- running the chain -----------------------------------------------------

def test_a_scenario_runs_end_to_end(population, spillovers):
    result = execute(SIMPLE, population, spillovers)
    assert result.scenario is SIMPLE
    assert len(result.equilibrium.rates) == len(BANKS)


def test_the_conditioned_bank_moves_most(population, spillovers):
    """The Fed faces the inflation shock, so it should respond hardest."""
    result = execute(SIMPLE, population, spillovers)
    moves = {
        b.name: abs(result.equilibrium.rates[b.name] - b.current_rate)
        for b in SIMPLE.apply_to(BANKS)
    }
    assert max(moves, key=moves.get) == FED.name


def test_the_shock_scale_comes_from_the_game(population, spillovers):
    """Not from a number the analyst chose. The scale is the equilibrium
    move divided by the impulse response's cumulative peak."""
    result = execute(SIMPLE, population, spillovers)
    assert result.shock.bank == FED.name
    assert result.shock.scale != 0.0


def test_a_larger_shock_produces_a_larger_response(population, spillovers):
    """A bigger inflation gap should mean a bigger policy move and a
    bigger shock. Both figures stay inside the plausibility guard: an
    inflation gap of five percentage points scales the synthetic response
    hard enough to imply a thirty percent deflation, which the guard
    correctly refuses.
    """
    mild = Scenario(
        name="mild",
        description="x",
        conditions=(Condition(bank=FED.name, inflation=0.025),),
        shock_origin=FED.name,
    )
    severe = Scenario(
        name="severe",
        description="x",
        conditions=(Condition(bank=FED.name, inflation=0.045),),
        shock_origin=FED.name,
    )
    assert abs(execute(severe, population, spillovers).shock.scale) > abs(
        execute(mild, population, spillovers).shock.scale
    )


def test_an_unknown_shock_origin_is_rejected(population, spillovers):
    scenario = Scenario(
        name="x",
        description="x",
        shock_origin="Bank of Nowhere",
    )
    with pytest.raises(EngineError, match="not a bank in this network"):
        execute(scenario, population, spillovers)


def test_the_specification_gate_blocks_every_scenario(population, spillovers):
    """A misspecified VAR is not something a scenario can opt out of."""
    with pytest.raises(EngineError, match="specification checks"):
        execute(SIMPLE, population, spillovers, usable=False)


# --- the distributional outcome --------------------------------------------

def test_borrowers_fare_worse_than_savers(population, spillovers):
    """The transfer, surviving four layers of transmission."""
    result = execute(SIMPLE, population, spillovers)
    assert result.borrower_change < result.saver_change


def test_the_spread_is_the_difference(population, spillovers):
    result = execute(SIMPLE, population, spillovers)
    assert result.spread == pytest.approx(
        result.borrower_change - result.saver_change
    )


def test_fixed_borrowers_sit_between_the_two(population, spillovers):
    """Insulated from the rate move but not from the income channel."""
    result = execute(SIMPLE, population, spillovers)
    assert result.borrower_change < result.fixed_borrower_change


def test_the_banking_wedge_is_positive(population, spillovers):
    """Borrowers absorb more of the move than savers receive."""
    result = execute(SIMPLE, population, spillovers)
    assert result.bank_wedge_bp > 0
    assert result.lending_rate_change_bp > result.deposit_rate_change_bp


def test_the_change_array_covers_every_household(population, spillovers):
    result = execute(SIMPLE, population, spillovers)
    assert result.change_by_household.shape == (len(population),)


def test_job_losses_are_counted(population, spillovers):
    result = execute(SIMPLE, population, spillovers)
    assert result.extra_job_losses >= 0


# --- leadership ------------------------------------------------------------

def test_leadership_changes_the_equilibrium(population, spillovers):
    """Whether commitment matters is the question the mode exists to ask."""
    simultaneous = execute(SIMPLE, population, spillovers)
    led = execute(
        Scenario(
            name="led",
            description="x",
            conditions=(Condition(bank=FED.name, inflation=0.045),),
            mode=SolutionMode.LED,
            leader=FED.name,
            shock_origin=FED.name,
        ),
        population,
        spillovers,
    )
    assert led.equilibrium.rates != simultaneous.equilibrium.rates


def test_the_leader_is_recorded(population, spillovers):
    led = execute(
        Scenario(
            name="led",
            description="x",
            mode=SolutionMode.LED,
            leader=FED.name,
            shock_origin=FED.name,
        ),
        population,
        spillovers,
    )
    assert led.equilibrium.leader == FED.name


# --- reproducibility -------------------------------------------------------

def test_a_scenario_is_reproducible(population, spillovers):
    first = execute(SIMPLE, population, spillovers)
    second = execute(SIMPLE, population, spillovers)
    assert first.aggregate_consumption_change == second.aggregate_consumption_change
    assert np.array_equal(first.change_by_household, second.change_by_household)


def test_running_a_scenario_does_not_mutate_the_banks(population, spillovers):
    before = FED.current_inflation
    execute(SIMPLE, population, spillovers)
    assert FED.current_inflation == before


def test_running_a_scenario_does_not_mutate_the_population(population, spillovers):
    before = population.wealth.copy()
    execute(SIMPLE, population, spillovers)
    assert np.array_equal(population.wealth, before)


# --- provenance ------------------------------------------------------------

def test_the_result_carries_its_diagnostics(population, spillovers):
    result = execute(SIMPLE, population, spillovers)
    assert result.diagnostics_usable is True


def test_the_result_carries_the_calibration_loss(population, spillovers):
    """Four of seven household targets are unsourced, so a result should
    say how well the population matched what it was fitted to."""
    result = execute(SIMPLE, population, spillovers)
    assert result.calibration_loss == 3.0


def test_the_result_serialises_completely(population, spillovers):
    payload = execute(SIMPLE, population, spillovers).to_ledger_dict()
    assert "scenario" in payload
    assert "equilibrium" in payload
    assert "shock" in payload
    assert payload["provenance"]["diagnostics_usable"] is True


def test_the_serialised_result_records_the_conditions(population, spillovers):
    payload = execute(SIMPLE, population, spillovers).to_ledger_dict()
    assert payload["scenario"]["conditions"][0]["bank"] == FED.name


# --- comparison ------------------------------------------------------------

def test_comparison_covers_every_scenario(population, spillovers):
    results = tuple(
        execute(s, population, spillovers)
        for s in (SIMPLE, DEFAULT_SCENARIOS[2])
    )
    summary = compare(results)
    assert summary["n_scenarios"] == 2
    assert len(summary["by_scenario"]) == 2


def test_comparison_names_the_widest_spread(population, spillovers):
    results = tuple(
        execute(s, population, spillovers)
        for s in (SIMPLE, DEFAULT_SCENARIOS[2])
    )
    summary = compare(results)
    assert summary["widest_spread"] in summary["by_scenario"]


def test_comparing_nothing_is_rejected():
    with pytest.raises(EngineError, match="nothing to compare"):
        compare(())


# --- the declared scenarios ------------------------------------------------
def test_every_declared_scenario_has_a_rationale():
    """A scenario without a stated reason is a number nobody asked for."""
    for scenario in DEFAULT_SCENARIOS:
        assert scenario.rationale.strip(), f"{scenario.name} has no rationale"


def test_declared_scenario_names_are_unique():
    names = [s.name for s in DEFAULT_SCENARIOS]
    assert len(set(names)) == len(names)


def test_every_declared_scenario_runs(population, spillovers):
    for scenario in DEFAULT_SCENARIOS:
        result = execute(scenario, population, spillovers)
        assert result.scenario.name == scenario.name


def test_the_twin_scenario_originates_from_the_rbi():
    """Its point is a domestically driven Indian tightening, which
    transmits differently from an imported one."""
    twin = next(s for s in DEFAULT_SCENARIOS if s.name == "twin_tightening")
    assert twin.shock_origin == RBI.name

def test_an_implausible_shock_is_refused(population, spillovers):
    """The guard catches a linear extrapolation past anything sensible.
    A seven percent inflation reading drives an equilibrium move large
    enough to imply a thirty one percent deflation path, which is a model
    running out of range rather than an economy.
    """
    extreme = Scenario(
        name="extreme",
        description="x",
        conditions=(Condition(bank=FED.name, inflation=0.070),),
        shock_origin=FED.name,
    )
    with pytest.raises(EngineError, match="plausible range"):
        execute(extreme, population, spillovers)