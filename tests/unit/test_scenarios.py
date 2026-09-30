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
from moirai.engine.economy.shock_path import MacroVariable
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
    CentralBank,
)
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix, network_nash
from moirai.engine.scenarios import (
    DEFAULT_SCENARIOS,
    HEADLINE_SCENARIO,
    HISTORICAL_2013,
    HISTORICAL_2022,
    HISTORICAL_SCENARIOS,
    HORIZON_CAPPED_FOR_INDIA,
    IMPORTED_TIGHTENING,
    INFLATION_HELD_FOR_INDIA,
    OBSERVED_2013_MOVES_BP,
    OBSERVED_2022_MOVES_BP,
    TWIN_TIGHTENING,
    Condition,
    HeldChannel,
    HorizonCap,
    Scenario,
    SolutionMode,
    compare,
    historical_banks,
    run_scenario,
)

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
DECLARED = (*DEFAULT_SCENARIOS, *(scenario for scenario, _ in HISTORICAL_SCENARIOS))
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

def test_leadership_is_worth_nothing_to_the_anchor(population, spillovers):
    """Moving first pays only if the followers' replies feed back into the
    leader's own economy. The anchor's inward sensitivity is the smallest
    in the hierarchy, so for the Fed they barely do, and leading and
    simultaneous play agree on every rate to within a basis point. See
    ADR 011.

    The Fed carries no external weight, so the simultaneous solver's
    linearised external term cannot contaminate this comparison. That is
    not true of the other banks.
    """
    anchor_row = np.asarray(spillovers.demand)[spillovers.names.index(FED.name)]
    assert anchor_row.max() < np.asarray(spillovers.demand).max()

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
    for bank in BANKS:
        assert led.equilibrium.rates[bank.name] == pytest.approx(
            simultaneous.equilibrium.rates[bank.name], abs=0.0001
        ), bank.name


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
        for s in (SIMPLE, TWIN_TIGHTENING)
    )
    summary = compare(results)
    assert summary["n_scenarios"] == 2
    assert len(summary["by_scenario"]) == 2


def test_comparison_names_the_widest_spread(population, spillovers):
    results = tuple(
        execute(s, population, spillovers)
        for s in (SIMPLE, TWIN_TIGHTENING)
    )
    summary = compare(results)
    assert summary["widest_spread"] in summary["by_scenario"]


def test_comparing_nothing_is_rejected():
    with pytest.raises(EngineError, match="nothing to compare"):
        compare(())


# --- the declared scenarios ------------------------------------------------
def test_every_declared_scenario_has_a_rationale():
    """A scenario without a stated reason is a number nobody asked for."""
    for scenario in DECLARED:
        assert scenario.rationale.strip(), f"{scenario.name} has no rationale"


def test_declared_scenario_names_are_unique():
    names = [s.name for s in DECLARED]
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

def test_the_imported_scenario_conditions_only_the_fed():
    """Its point is a shock India did not cause. A condition on the RBI
    would make part of the move domestic."""
    assert IMPORTED_TIGHTENING.shock_origin == RBI.name
    assert {c.bank for c in IMPORTED_TIGHTENING.conditions} == {FED.name}


def test_the_imported_shock_is_the_rbis_response_to_the_fed(population, spillovers):
    """The RBI's move is measured against the game without the Fed's
    condition, so it is exactly what the Fed's inflation causes at the RBI,
    and it is a tightening."""
    result = execute(IMPORTED_TIGHTENING, population, spillovers)
    unconditioned = network_nash(BANKS, spillovers).rates[RBI.name]
    assert result.shock.bank == RBI.name
    assert result.shock.reference_rate == pytest.approx(unconditioned)
    assert result.shock.deviation > 0


def test_the_imported_shock_is_smaller_than_its_source(population, spillovers):
    """Spillovers attenuate. An imported move at least as large as the
    Fed's own would mean the network amplifies, which no tier allows."""
    imported = execute(IMPORTED_TIGHTENING, population, spillovers)
    source = execute(SIMPLE, population, spillovers)
    assert 0 < imported.shock.deviation < source.shock.deviation


def test_every_rbi_origin_scenario_declares_the_proxy_limits():
    """An RBI-origin path is carried by the US impulse response, whose
    inflation sign and persistence the Indian evidence contradicts. A
    declared scenario that used it without saying so would be silent
    about both. See ADR 010."""
    for scenario in DECLARED:
        if scenario.shock_origin != RBI.name:
            continue
        assert INFLATION_HELD_FOR_INDIA in scenario.held_channels, scenario.name
        assert scenario.horizon_cap == HORIZON_CAPPED_FOR_INDIA, scenario.name


# --- the shock is a counterfactual difference ------------------------------

def test_the_shock_is_measured_against_the_unconditioned_game(population, spillovers):
    """Not against the current rate, which would count the move the bank
    makes with no conditions applied at all."""
    result = execute(SIMPLE, population, spillovers)
    unconditioned = network_nash(BANKS, spillovers).rates[FED.name]
    assert result.shock.reference_rate == pytest.approx(unconditioned)
    assert result.shock.deviation == pytest.approx(
        result.equilibrium.rates[FED.name] - unconditioned
    )


def test_the_unconditioned_game_already_moves_the_bank(spillovers):
    """The premise of the counterfactual. If the no-condition equilibrium
    sat at the current rate, measuring from either would be the same."""
    unconditioned = network_nash(BANKS, spillovers).rates[FED.name]
    assert unconditioned != pytest.approx(FED.current_rate)


def test_a_scenario_without_conditions_has_no_shock(population, spillovers):
    quiet = Scenario(name="quiet", description="x", shock_origin=FED.name)
    result = execute(quiet, population, spillovers)
    assert result.shock.deviation == pytest.approx(0.0)
    assert result.aggregate_consumption_change == pytest.approx(0.0)


def test_the_reference_game_uses_the_scenario_mode(population, spillovers):
    """A led scenario compared against a simultaneous baseline would
    measure leadership as well as the conditions."""
    led = Scenario(
        name="led",
        description="x",
        conditions=(Condition(bank=FED.name, inflation=0.045),),
        mode=SolutionMode.LED,
        leader=FED.name,
        shock_origin=FED.name,
    )
    result = execute(led, population, spillovers)
    assert result.reference_equilibrium.leader == FED.name


def test_the_reference_equilibrium_is_recorded(population, spillovers):
    payload = execute(SIMPLE, population, spillovers).to_ledger_dict()
    assert "reference_equilibrium" in payload
    assert "reference_rate" in payload["shock"]


# --- baselines come from the origin bank -----------------------------------

def test_income_growth_baseline_comes_from_the_origin_bank(population, spillovers):
    twin = next(s for s in DEFAULT_SCENARIOS if s.name == "twin_tightening")
    result = execute(twin, population, spillovers)
    assert result.path.baselines[MacroVariable.INCOME_GROWTH] == RBI.current_income_growth
    assert result.path.baselines[MacroVariable.POLICY_RATE] == RBI.current_rate


def test_a_bank_without_income_growth_cannot_originate(population, spillovers):
    """Refusing is better than a default that belongs to another economy."""
    assert ECB.current_income_growth is None
    scenario = Scenario(
        name="ecb",
        description="x",
        conditions=(Condition(bank=ECB.name, inflation=0.04),),
        shock_origin=ECB.name,
    )
    with pytest.raises(EngineError, match="current_income_growth"):
        execute(scenario, population, spillovers)


# --- declared departures from the estimated transmission -------------------

def test_a_held_channel_sits_at_baseline(population, spillovers):
    held = SIMPLE.model_copy(
        update={"held_channels": (HeldChannel(variable=MacroVariable.INFLATION, reason="x"),)}
    )
    path = execute(held, population, spillovers).path
    assert np.all(path.deviation(MacroVariable.INFLATION) == 0.0)
    assert np.any(path.deviation(MacroVariable.POLICY_RATE) != 0.0)


def test_holding_inflation_changes_the_outcome(population, spillovers):
    """The channel is live, so holding it is a real assumption."""
    held = SIMPLE.model_copy(
        update={"held_channels": (HeldChannel(variable=MacroVariable.INFLATION, reason="x"),)}
    )
    assert (
        execute(held, population, spillovers).aggregate_consumption_change
        != execute(SIMPLE, population, spillovers).aggregate_consumption_change
    )


def test_the_horizon_cap_shortens_the_simulation(population, spillovers):
    capped = SIMPLE.model_copy(update={"horizon_cap": HorizonCap(periods=12, reason="x")})
    result = execute(capped, population, spillovers)
    assert result.path.horizon == 12
    assert len(result.path) == 13


def test_a_cap_beyond_the_response_horizon_changes_nothing(population, spillovers):
    capped = SIMPLE.model_copy(update={"horizon_cap": HorizonCap(periods=100, reason="x")})
    assert execute(capped, population, spillovers).path.horizon == make_irf().horizon


def test_holding_the_policy_rate_is_rejected():
    with pytest.raises(Exception, match="removes the shock"):
        Scenario(
            name="x",
            description="x",
            shock_origin=FED.name,
            held_channels=(HeldChannel(variable=MacroVariable.POLICY_RATE, reason="x"),),
        )


def test_a_channel_cannot_be_held_twice():
    channel = HeldChannel(variable=MacroVariable.INFLATION, reason="x")
    with pytest.raises(Exception, match="held twice"):
        Scenario(
            name="x", description="x", shock_origin=FED.name, held_channels=(channel, channel)
        )


def test_a_departure_needs_a_reason():
    with pytest.raises(Exception):
        HeldChannel(variable=MacroVariable.INFLATION, reason="")
    with pytest.raises(Exception):
        HorizonCap(periods=30, reason="")


def test_the_departures_are_recorded_in_the_ledger():
    twin = next(s for s in DEFAULT_SCENARIOS if s.name == "twin_tightening")
    payload = twin.to_ledger_dict()
    assert payload["held_channels"][0]["variable"] == "inflation"
    assert "wrong sign" in payload["held_channels"][0]["reason"]
    assert payload["horizon_cap"]["periods"] == 30


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

# --- historical scenarios: 2022 and 2013 (ADRs 012, 013) -----------------

HISTORICAL_IDS = [scenario.name for scenario, _ in HISTORICAL_SCENARIOS]
OBSERVED = {
    HISTORICAL_2022.name: OBSERVED_2022_MOVES_BP,
    HISTORICAL_2013.name: OBSERVED_2013_MOVES_BP,
}


@pytest.fixture(scope="module")
def sourced_spillovers() -> SpilloverMatrix:
    return SpilloverMatrix.from_literature(DEFAULT_TIERS)


def solve_historical(
    scenario: Scenario,
    banks: tuple[CentralBank, ...],
    conditions: tuple[Condition, ...],
    spillovers: SpilloverMatrix,
):
    only = scenario.model_copy(update={"conditions": conditions})
    return network_nash(only.apply_to(banks), spillovers)


def test_every_historical_scenario_has_observed_moves():
    assert set(OBSERVED) == set(HISTORICAL_IDS)


@pytest.mark.parametrize(("scenario", "banks"), HISTORICAL_SCENARIOS, ids=HISTORICAL_IDS)
def test_a_historical_scenario_conditions_banks_it_carries(scenario, banks):
    names = {b.name for b in banks}
    assert {c.bank for c in scenario.conditions} <= names
    assert scenario.shock_origin in names
    assert set(OBSERVED[scenario.name]) == names


@pytest.mark.parametrize(("scenario", "banks"), HISTORICAL_SCENARIOS, ids=HISTORICAL_IDS)
def test_a_historical_reference_game_has_no_inflation_problem(scenario, banks):
    """Caused is measured against every bank at its target. The default
    banks' states describe a later period and would be incoherent next to
    historical rates."""
    for bank in banks:
        assert bank.current_inflation == bank.inflation_target, bank.name
        assert bank.current_output_gap == 0.0, bank.name


def test_historical_banks_need_a_rate_for_every_bank():
    with pytest.raises(EngineError, match="every major bank"):
        historical_banks({FED.name: 0.01})


@pytest.mark.parametrize(("scenario", "banks"), HISTORICAL_SCENARIOS, ids=HISTORICAL_IDS)
def test_a_historical_caused_move_splits_exactly_by_condition(
    scenario, banks, sourced_spillovers
):
    """The reaction system is linear in inflation, so each bank's
    condition contributes separately and the parts sum to the whole. The
    attributions in ADRs 012 and 013 rest on this."""
    reference = solve_historical(scenario, banks, (), sourced_spillovers)
    together = solve_historical(scenario, banks, scenario.conditions, sourced_spillovers)
    parts = [
        solve_historical(scenario, banks, (c,), sourced_spillovers)
        for c in scenario.conditions
    ]
    for bank in banks:
        caused = together.rates[bank.name] - reference.rates[bank.name]
        summed = sum(p.rates[bank.name] - reference.rates[bank.name] for p in parts)
        assert summed == pytest.approx(caused, abs=1e-12), bank.name


def conditioned_state(scenario, banks, spillovers):
    solved = solve_historical(scenario, banks, scenario.conditions, spillovers)
    names = spillovers.names
    conditioned = scenario.apply_to(banks)
    ordered = tuple(next(b for b in conditioned if b.name == n) for n in names)
    rates = np.array([solved.rates[n] for n in names])
    return ordered, rates


@pytest.mark.parametrize(("scenario", "banks"), HISTORICAL_SCENARIOS, ids=HISTORICAL_IDS)
def test_a_historical_scenario_leaves_the_rbi_outside_its_band(
    scenario, banks, sourced_spillovers
):
    """Both historical years do, which is why the README no longer claims
    no declared scenario does. Detected by the band penalty being active in
    the RBI's realised loss, not by restating the transmission."""
    from moirai.engine.financial.network import _losses_at

    ordered, rates = conditioned_state(scenario, banks, sourced_spillovers)
    unbanded = tuple(
        b.model_copy(update={"tolerance_lower": None, "tolerance_upper": None})
        if b.name == RBI.name
        else b
        for b in ordered
    )
    with_band = _losses_at(ordered, rates, sourced_spillovers)[RBI.name]
    without = _losses_at(unbanded, rates, sourced_spillovers)[RBI.name]
    assert with_band > without


@pytest.mark.parametrize(("scenario", "banks"), HISTORICAL_SCENARIOS, ids=HISTORICAL_IDS)
def test_the_simultaneous_solver_is_approximate_when_the_band_binds(
    scenario, banks, sourced_spillovers
):
    """ADR 011 predicted this: the reaction system omits the band penalty,
    so outside the band the solved rate is not the RBI's best reply under
    its true loss. The gap is measured, not assumed. ADRs 012 and 013
    record it."""
    from scipy.optimize import minimize_scalar

    from moirai.engine.financial.network import _losses_at

    ordered, rates = conditioned_state(scenario, banks, sourced_spillovers)
    i = sourced_spillovers.names.index(RBI.name)

    def rbi_loss(rate: float) -> float:
        trial = rates.copy()
        trial[i] = rate
        return _losses_at(ordered, trial, sourced_spillovers)[RBI.name]

    reply = minimize_scalar(
        rbi_loss,
        bounds=(rates[i] - 0.05, rates[i] + 0.05),
        method="bounded",
        options={"xatol": 1e-11},
    ).x
    assert reply - rates[i] > 0.0001


# --- the headline is the chain (ADR 016) ------------------------------------
#
# The pipeline used to report the Fed's own rate path, at an arbitrary two
# standard deviations, delivered straight to Indian households. These tests
# pin what the headline must be instead: every link solved, the RBI in
# between, and the move measured as what the Fed caused.


def test_the_headline_is_a_declared_scenario():
    assert HEADLINE_SCENARIO in DEFAULT_SCENARIOS


def test_the_headline_shock_originates_with_the_rbi():
    """Indian households face the RBI's rate, not the Fed's."""
    assert HEADLINE_SCENARIO.shock_origin == RBI.name


def test_the_headline_conditions_only_the_fed():
    """So the RBI's move is wholly imported rather than domestic."""
    assert {c.bank for c in HEADLINE_SCENARIO.conditions} == {FED.name}


def test_the_headline_declares_its_departures_from_the_us_transmission():
    """The shape is the US impulse response (ADR 010); where that is not
    credible for India, the headline says so rather than passing it on."""
    held = {c.variable for c in HEADLINE_SCENARIO.held_channels}
    assert MacroVariable.INFLATION in held
    assert HEADLINE_SCENARIO.horizon_cap is not None


def test_the_headline_shock_is_the_rbis_caused_move(population, spillovers):
    """Measured against the same game without the US condition, so the
    RBI's unprompted move is excluded from the shock."""
    result = execute(HEADLINE_SCENARIO, population, spillovers)
    caused = (
        result.equilibrium.rates[RBI.name]
        - result.reference_equilibrium.rates[RBI.name]
    )
    assert result.shock.bank == RBI.name
    assert result.shock.deviation == pytest.approx(caused, abs=1e-12)


def test_the_headline_needs_no_extreme_extrapolation(population, spillovers):
    """The pipeline runs the headline without allow_extreme, so the solved
    shock must sit inside the range the VAR sample supports."""
    result = execute(HEADLINE_SCENARIO, population, spillovers)
    assert not result.shock.is_extreme

