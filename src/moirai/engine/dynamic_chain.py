"""The chain on the dynamic game: households face the RBI's own rate path.

`scenarios.run_scenario` takes the origin bank's static caused move, one
number, converts it into a multiple of the estimated US policy shock, and
gives households the US impulse response rescaled to that size (ADR 010).
The path's shape is therefore America's: how long the rate stays up, how
income growth responds and when, all come from a VAR estimated on US data.

The dynamic game (ADR 023) produces the origin bank's rate path directly,
quarter by quarter, together with its economy's inflation and output
gaps. This module hands households that path instead. No VAR is used and
no shock is rescaled: the size and the shape of what Indian households
face both come from the game.

The difference is counterfactual, as in the static chain: the dynamic game
is solved with and without the scenario's conditions, and households face
the difference, added to the origin bank's current rate, inflation and
income growth. The scenario's declared departures, channels held at
baseline and a horizon cap, are applied as in the static chain.

Conversion to the household layer's monthly clock:

  - rates and inflation are set each quarter and held for its three
    months, since policy is decided at meetings rather than drifting;
  - income growth is the annualised change in the output gap over the
    quarter, spread evenly over its months, since the gap is a level and
    households read a growth rate.

This module adds a route through the chain. It changes nothing
`run_scenario` computes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.economy.behaviour import (
    BehaviourParameters,
    counterfactual,
    group_change,
    never_consuming,
)
from moirai.engine.economy.households import Population
from moirai.engine.economy.shock_path import MacroVariable, ShockPath
from moirai.engine.financial.central_banks import CentralBank
from moirai.engine.financial.commercial_banks import BankingSystem
from moirai.engine.financial.dynamic_game import (
    CALIBRATED_INERTIA_WEIGHT,
    CALIBRATED_RBI_EXTERNAL_WEIGHT,
    DynamicEquilibrium,
    DynamicParameters,
    dynamic_nash,
)
from moirai.engine.financial.network import INDIA_DOLLAR_INVOICING_SHARE, SpilloverMatrix
from moirai.engine.scenarios import Scenario

log = get_logger(__name__)

MONTHS_PER_QUARTER = 3


class DynamicScenarioResult(BaseModel):
    """What households bear when the origin bank's dynamic path reaches them."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    scenario: Scenario
    equilibrium: DynamicEquilibrium
    reference_equilibrium: DynamicEquilibrium
    caused_moves_bp: Any = Field(description="(quarters,) origin bank's caused move.")
    path: ShockPath
    aggregate_consumption_change: float
    extra_job_losses: int
    change_by_household: Any = Field(description="(n,) proportional change.")
    borrower_change: float
    fixed_borrower_change: float
    households_never_consuming: int = 0
    saver_change: float
    lending_rate_change_bp: float
    deposit_rate_change_bp: float

    @property
    def spread(self) -> float:
        """Floating-rate borrowers against net savers, as in ScenarioResult."""
        return self.borrower_change - self.saver_change

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "route": "dynamic",
            "caused_moves_bp": [round(float(v), 4) for v in self.caused_moves_bp],
            "aggregate_consumption_change": self.aggregate_consumption_change,
            "spread": self.spread,
            "extra_job_losses": self.extra_job_losses,
            "path": self.path.to_ledger_dict(),
        }


def calibrated_inputs(
    banks: tuple[CentralBank, ...],
    spillovers: SpilloverMatrix,
    *,
    rbi: str = "Reserve Bank of India",
    fed: str = "Federal Reserve",
) -> tuple[tuple[CentralBank, ...], SpilloverMatrix, DynamicParameters]:
    """The dynamic game as calibrated in ADRs 026 and 027.

    Inertia at the weight that gives the Fed its estimated smoothing; the
    RBI's external objective weighted by invoicing currency, the Fed's rate
    carrying India's dollar invoicing share and the rest split equally; and
    the RBI's external weight at which it follows the Fed by the measured
    floor. Every other bank and every validated default is left as it is.
    """
    others = [b.name for b in banks if b.name not in (rbi, fed)]
    rest = (1.0 - INDIA_DOLLAR_INVOICING_SHARE) / len(others)
    weights = {fed: INDIA_DOLLAR_INVOICING_SHARE, **{name: rest for name in others}}
    calibrated_banks = tuple(
        b.model_copy(update={"external_weight": CALIBRATED_RBI_EXTERNAL_WEIGHT})
        if b.name == rbi
        else b
        for b in banks
    )
    return (
        calibrated_banks,
        spillovers.with_reference(rbi, weights),
        DynamicParameters(inertia_weight=CALIBRATED_INERTIA_WEIGHT),
    )


def household_path(
    origin: CentralBank,
    shocked: DynamicEquilibrium,
    reference: DynamicEquilibrium,
    *,
    months: int,
    shock_name: str = "dynamic_game",
) -> tuple[ShockPath, np.ndarray]:
    """The monthly macro path households face, from two dynamic solutions.

    Returns the path and the origin bank's caused move by quarter, in basis
    points. The path has months + 1 entries, period 0 being the first
    month of the first quarter.
    """
    if origin.current_income_growth is None:
        raise EngineError(
            f"{origin.name} declares no current_income_growth; a household path "
            f"needs the origin economy's own income growth"
        )
    i = shocked.index_of(origin.name)
    quarters_needed = months // MONTHS_PER_QUARTER + 2
    available = np.asarray(shocked.moves).shape[0]
    if available < quarters_needed:
        raise EngineError(
            f"the dynamic solution covers {available} quarters; {quarters_needed} are "
            f"needed for {months} months"
        )

    moves = (np.asarray(shocked.moves) - np.asarray(reference.moves))[:, i]
    inflation_gap = (np.asarray(shocked.inflation_gaps) - np.asarray(reference.inflation_gaps))[
        :, i
    ]
    output_gap = (np.asarray(shocked.output_gaps) - np.asarray(reference.output_gaps))[:, i]
    # Annualised growth of the output gap over each quarter: a level gap
    # falling by 0.1 of potential in a quarter is growth 0.4 a year lower.
    growth = np.diff(output_gap) * 4.0

    period = np.arange(months + 1)
    quarter = period // MONTHS_PER_QUARTER

    baselines = {
        MacroVariable.POLICY_RATE: origin.current_rate,
        MacroVariable.INFLATION: origin.current_inflation,
        MacroVariable.INCOME_GROWTH: float(origin.current_income_growth),
    }
    paths = {
        MacroVariable.POLICY_RATE: origin.current_rate + moves[quarter],
        MacroVariable.INFLATION: origin.current_inflation + inflation_gap[quarter],
        MacroVariable.INCOME_GROWTH: float(origin.current_income_growth) + growth[quarter],
    }
    path = ShockPath(
        shock_name=shock_name,
        horizon=months,
        paths=paths,
        baselines=baselines,
        scale_factor=1.0,
        scheme="dynamic_game",
        assumptions=(
            f"{origin.name}'s caused rate path from the dynamic game (ADR 023), held "
            f"for each quarter's three months; inflation from the game's own "
            f"transmission; income growth the annualised change in the output gap."
        ),
        diagnostics_usable=None,
    )
    return path, moves * 10_000


def run_dynamic_scenario(
    scenario: Scenario,
    banks: tuple[CentralBank, ...],
    spillovers: SpilloverMatrix,
    *,
    population: Population,
    banking_system: BankingSystem,
    behaviour: BehaviourParameters | None = None,
    parameters: DynamicParameters | None = None,
    months: int = 36,
) -> DynamicScenarioResult:
    """Run one scenario through the chain with the dynamic game.

    The counterpart of `scenarios.run_scenario`. The same scenario, banks,
    spillovers, population and banking system give a result directly
    comparable to the static chain's, differing only in where the path
    households face comes from.
    """
    behaviour = behaviour or BehaviourParameters()
    conditioned = scenario.apply_to(banks)
    by_name = {b.name: b for b in conditioned}
    if scenario.shock_origin not in by_name:
        raise EngineError(
            f"{scenario.name}: shock origin {scenario.shock_origin!r} is not a bank "
            f"in this network"
        )
    origin = by_name[scenario.shock_origin]

    quarters = months // MONTHS_PER_QUARTER + 2
    shocked = dynamic_nash(conditioned, spillovers, parameters=parameters, horizon=quarters)
    reference = dynamic_nash(banks, spillovers, parameters=parameters, horizon=quarters)

    path, caused_bp = household_path(origin, shocked, reference, months=months)
    for channel in scenario.held_channels:
        path = path.hold(channel.variable)
    if scenario.horizon_cap is not None:
        path = path.truncate(scenario.horizon_cap.periods)

    baseline, shocked_households = counterfactual(population, path, behaviour, banking_system)
    base_total = np.sum([o.consumption for o in baseline], axis=0)
    shock_total = np.sum([o.consumption for o in shocked_households], axis=0)
    change = (shock_total - base_total) / np.maximum(base_total, 1.0)

    baseline_policy = path.baselines[MacroVariable.POLICY_RATE]
    peak_period, _ = path.peak(MacroVariable.POLICY_RATE)
    peak_policy = path.get(MacroVariable.POLICY_RATE)[peak_period]
    before = banking_system.effective_rates(baseline_policy, baseline_policy)
    after = banking_system.effective_rates(peak_policy, baseline_policy)

    result = DynamicScenarioResult(
        scenario=scenario,
        equilibrium=shocked,
        reference_equilibrium=reference,
        caused_moves_bp=caused_bp,
        path=path,
        aggregate_consumption_change=float(shock_total.sum() / base_total.sum() - 1.0),
        extra_job_losses=(
            sum(int(o.became_unemployed.sum()) for o in shocked_households)
            - sum(int(o.became_unemployed.sum()) for o in baseline)
        ),
        change_by_household=change,
        borrower_change=group_change(base_total, shock_total, population.is_rate_exposed),
        fixed_borrower_change=group_change(
            base_total, shock_total, population.is_indebted & ~population.debt_is_floating
        ),
        saver_change=group_change(base_total, shock_total, ~population.is_indebted),
        households_never_consuming=int(never_consuming(baseline).sum()),
        lending_rate_change_bp=(after["lending_rate"] - before["lending_rate"]) * 10_000,
        deposit_rate_change_bp=(after["deposit_rate"] - before["deposit_rate"]) * 10_000,
    )
    log.info(
        "dynamic_scenario_completed",
        scenario=scenario.name,
        first_move_bp=round(float(caused_bp[0]), 3),
        aggregate=round(result.aggregate_consumption_change * 100, 4),
        spread=round(result.spread * 100, 4),
    )
    return result
