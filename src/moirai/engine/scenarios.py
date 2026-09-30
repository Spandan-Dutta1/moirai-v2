"""
Named scenarios run end to end.

Every script so far hardcodes one set of conditions and prints one
answer. A scenario makes the conditions a declared object instead, so
several can be run and compared in the same shape, and so the conditions
that produced a result travel with it.

The chain a scenario runs:

    conditions -> policy game -> equilibrium rates
               -> estimated transmission
               -> commercial banks
               -> heterogeneous households
               -> distributional outcome

Two things are deliberately not automated. The macro estimation happens
once and is passed in, because refitting a VAR per scenario would be slow
and would obscure that every scenario shares the same estimated
transmission. And no scenario overrides the specification gate: a
misspecified VAR blocks every scenario built on it, rather than each one
deciding for itself.

A scenario is a claim about conditions, not about outcomes. Its
plausibility is the analyst's responsibility; the pipeline only carries
it through faithfully and records what it assumed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.diagnostics import DiagnosticReport
from moirai.engine.causal.irf import ImpulseResponse
from moirai.engine.economy.behaviour import BehaviourParameters, counterfactual
from moirai.engine.economy.households import Population
from moirai.engine.economy.policy_shock import PolicyShock, mappings_for_bank, path_from_game
from moirai.engine.economy.shock_path import MacroVariable, ShockPath
from moirai.engine.financial.central_banks import MAJOR_CENTRAL_BANKS, CentralBank
from moirai.engine.financial.commercial_banks import BankingSystem
from moirai.engine.financial.network import (
    NetworkEquilibrium,
    SpilloverMatrix,
    network_nash,
    network_stackelberg,
)

log = get_logger(__name__)


class SolutionMode(StrEnum):
    """How the banks are assumed to interact in this scenario."""

    SIMULTANEOUS = "simultaneous"
    LED = "led"


class Condition(BaseModel):
    """One departure from the baseline state of a central bank.

    Conditions describe the world the banks find themselves in, not the
    policy they choose. Inflation running above target is a condition; the
    rate response to it is an outcome of the game.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: str = Field(min_length=1)
    inflation: float | None = Field(default=None, description="Override, as a decimal.")
    output_gap: float | None = None
    note: str = ""

    def apply(self, bank: CentralBank) -> CentralBank:
        updates: dict[str, Any] = {}
        if self.inflation is not None:
            updates["current_inflation"] = self.inflation
        if self.output_gap is not None:
            updates["current_output_gap"] = self.output_gap
        return bank.model_copy(update=updates) if updates else bank


class HeldChannel(BaseModel):
    """A macro variable pinned at its baseline, and why.

    The estimated transmission is shared by every scenario, but it need
    not be credible for every economy a scenario delivers it to. Holding a
    channel is the declared alternative to passing through a response the
    evidence for that economy contradicts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variable: MacroVariable
    reason: str = Field(min_length=1)


class HorizonCap(BaseModel):
    """A limit on how many periods households are simulated, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    periods: int = Field(ge=1)
    reason: str = Field(min_length=1)


class Scenario(BaseModel):
    """A named set of conditions and how to solve under them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    conditions: tuple[Condition, ...] = ()
    mode: SolutionMode = SolutionMode.SIMULTANEOUS
    leader: str | None = Field(
        default=None, description="Required when mode is LED."
    )
    shock_origin: str = Field(
        description="Whose policy move drives the household-facing path."
    )
    rationale: str = Field(
        default="",
        description="Why these conditions are worth asking about.",
    )
    held_channels: tuple[HeldChannel, ...] = Field(
        default=(),
        description="Macro variables pinned at baseline on the household path.",
    )
    horizon_cap: HorizonCap | None = Field(
        default=None,
        description="Shorter simulation horizon than the impulse response offers.",
    )

    @model_validator(mode="after")
    def _leader_matches_mode(self) -> Scenario:
        if self.mode is SolutionMode.LED and not self.leader:
            raise ValueError(f"{self.name}: LED mode needs a leader")
        if self.mode is SolutionMode.SIMULTANEOUS and self.leader:
            raise ValueError(f"{self.name}: a leader is meaningless when simultaneous")
        return self

    @model_validator(mode="after")
    def _held_channels_leave_a_shock(self) -> Scenario:
        held = [c.variable for c in self.held_channels]
        if len(set(held)) != len(held):
            raise ValueError(f"{self.name}: a channel is held twice")
        if MacroVariable.POLICY_RATE in held:
            raise ValueError(
                f"{self.name}: holding the policy rate removes the shock itself"
            )
        return self

    def apply_to(self, banks: tuple[CentralBank, ...]) -> tuple[CentralBank, ...]:
        """Return the banks as this scenario finds them."""
        by_name = {b.name: b for b in banks}
        for condition in self.conditions:
            if condition.bank not in by_name:
                raise EngineError(
                    f"{self.name}: no bank named {condition.bank!r}; "
                    f"have {sorted(by_name)}"
                )
            by_name[condition.bank] = condition.apply(by_name[condition.bank])
        return tuple(by_name[b.name] for b in banks)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode.value,
            "leader": self.leader,
            "shock_origin": self.shock_origin,
            "rationale": self.rationale,
            "held_channels": [
                {"variable": c.variable.value, "reason": c.reason}
                for c in self.held_channels
            ],
            "horizon_cap": (
                None
                if self.horizon_cap is None
                else {"periods": self.horizon_cap.periods, "reason": self.horizon_cap.reason}
            ),
            "conditions": [
                {
                    "bank": c.bank,
                    "inflation": c.inflation,
                    "output_gap": c.output_gap,
                    "note": c.note,
                }
                for c in self.conditions
            ],
        }


class ScenarioResult(BaseModel):
    """What a scenario produced, with everything needed to reproduce it."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    scenario: Scenario
    equilibrium: NetworkEquilibrium
    reference_equilibrium: NetworkEquilibrium = Field(
        description="The same game without the scenario's conditions."
    )
    shock: PolicyShock
    path: ShockPath

    aggregate_consumption_change: float
    extra_job_losses: int
    change_by_household: Any = Field(description="(n,) proportional change.")

    borrower_change: float
    fixed_borrower_change: float
    saver_change: float

    lending_rate_change_bp: float
    deposit_rate_change_bp: float

    diagnostics_usable: bool
    calibration_loss: float

    @property
    def spread(self) -> float:
        """The transfer a representative household averages away."""
        return self.borrower_change - self.saver_change

    @property
    def bank_wedge_bp(self) -> float:
        """How much of the policy move stopped at the banking system."""
        return self.lending_rate_change_bp - self.deposit_rate_change_bp

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_ledger_dict(),
            "equilibrium": self.equilibrium.to_ledger_dict(),
            "reference_equilibrium": self.reference_equilibrium.to_ledger_dict(),
            "shock": self.shock.to_ledger_dict(),
            "outcome": {
                "aggregate_consumption_change": round(
                    self.aggregate_consumption_change, 6
                ),
                "extra_job_losses": self.extra_job_losses,
                "borrower_change": round(self.borrower_change, 6),
                "fixed_borrower_change": round(self.fixed_borrower_change, 6),
                "saver_change": round(self.saver_change, 6),
                "spread": round(self.spread, 6),
                "lending_rate_change_bp": round(self.lending_rate_change_bp, 1),
                "deposit_rate_change_bp": round(self.deposit_rate_change_bp, 1),
                "bank_wedge_bp": round(self.bank_wedge_bp, 1),
            },
            "provenance": {
                "diagnostics_usable": self.diagnostics_usable,
                "calibration_loss": round(self.calibration_loss, 4),
            },
        }


def run_scenario(
    scenario: Scenario,
    banks: tuple[CentralBank, ...],
    spillovers: SpilloverMatrix,
    responses: ImpulseResponse,
    *,
    shock_name: str,
    rate_variable: str,
    price_variable: str,
    output_variable: str,
    population: Population,
    banking_system: BankingSystem,
    diagnostics: DiagnosticReport,
    behaviour: BehaviourParameters | None = None,
    calibration_loss: float = 0.0,
    allow_extreme: bool = False,
) -> ScenarioResult:
    """Run one scenario through the full chain.

    The macro estimation is passed in rather than performed here. Every
    scenario shares the same estimated transmission, and refitting per
    scenario would both be slow and obscure that they do.

    The shock is a counterfactual difference. The game is solved twice,
    with and without the scenario's conditions, in the same mode, and the
    origin bank's move is the gap between the two. Measuring from the
    bank's current rate instead would count the move it makes with no
    conditions at all: in the default network the RBI goes from 5.25 to
    5.31 percent unprompted, which is a third of its response to a US
    inflation shock and none of it imported.
    """
    behaviour = behaviour or BehaviourParameters()

    # ---- the banks find themselves in these conditions ----
    conditioned = scenario.apply_to(banks)
    by_name = {b.name: b for b in conditioned}
    if scenario.shock_origin not in by_name:
        raise EngineError(
            f"{scenario.name}: shock origin {scenario.shock_origin!r} is not a "
            f"bank in this network"
        )

    # ---- they choose rates, with and without the conditions ----
    equilibrium = _solve(scenario, conditioned, spillovers)
    reference = _solve(scenario, banks, spillovers)

    # ---- the difference the conditions make becomes a shock ----
    origin = by_name[scenario.shock_origin]
    solution = _as_analytic(equilibrium)

    path, shock = path_from_game(
        solution,
        origin,
        responses,
        shock_name,
        rate_variable,
        mappings_for_bank(origin, rate_variable, price_variable, output_variable),
        diagnostics=diagnostics,
        require_usable=True,
        allow_extreme=allow_extreme,
        reference_rate=reference.rates[scenario.shock_origin],
    )

    # ---- declared departures from the estimated transmission ----
    for channel in scenario.held_channels:
        path = path.hold(channel.variable)
    if scenario.horizon_cap is not None:
        path = path.truncate(scenario.horizon_cap.periods)

    # ---- households bear it ----
    baseline, shocked = counterfactual(population, path, behaviour, banking_system)

    base_total = np.sum([o.consumption for o in baseline], axis=0)
    shock_total = np.sum([o.consumption for o in shocked], axis=0)
    change = (shock_total - base_total) / np.maximum(base_total, 1.0)

    baseline_policy = path.baselines[MacroVariable.POLICY_RATE]
    peak_period, _ = path.peak(MacroVariable.POLICY_RATE)
    peak_policy = path.get(MacroVariable.POLICY_RATE)[peak_period]
    before = banking_system.effective_rates(baseline_policy, baseline_policy)
    after = banking_system.effective_rates(peak_policy, baseline_policy)

    result = ScenarioResult(
        scenario=scenario,
        equilibrium=equilibrium,
        reference_equilibrium=reference,
        shock=shock,
        path=path,
        aggregate_consumption_change=float(shock_total.sum() / base_total.sum() - 1.0),
        extra_job_losses=(
            sum(int(o.became_unemployed.sum()) for o in shocked)
            - sum(int(o.became_unemployed.sum()) for o in baseline)
        ),
        change_by_household=change,
        borrower_change=float(change[population.is_rate_exposed].mean()),
        fixed_borrower_change=float(
            change[population.is_indebted & ~population.debt_is_floating].mean()
        ),
        saver_change=float(change[~population.is_indebted].mean()),
        lending_rate_change_bp=(after["lending_rate"] - before["lending_rate"]) * 10_000,
        deposit_rate_change_bp=(after["deposit_rate"] - before["deposit_rate"]) * 10_000,
        diagnostics_usable=diagnostics.is_usable,
        calibration_loss=calibration_loss,
    )

    log.info(
        "scenario_completed",
        scenario=scenario.name,
        mode=scenario.mode.value,
        shock_scale=round(shock.scale, 3),
        aggregate=round(result.aggregate_consumption_change * 100, 4),
        spread=round(result.spread * 100, 4),
    )
    return result


def _solve(
    scenario: Scenario, banks: tuple[CentralBank, ...], spillovers: SpilloverMatrix
) -> NetworkEquilibrium:
    """Solve the network in the scenario's mode."""
    if scenario.mode is SolutionMode.SIMULTANEOUS:
        return network_nash(banks, spillovers)
    assert scenario.leader is not None
    return network_stackelberg(banks, spillovers, scenario.leader)


def _as_analytic(equilibrium: NetworkEquilibrium):
    """Adapt a network equilibrium to the two-bank solution interface.

    `path_from_game` was written against the pairwise solver and needs
    only the rates and a concept label. Rather than duplicate it, the
    network result is wrapped. If the interfaces diverge further this
    should become a shared protocol instead.
    """
    from moirai.engine.financial.central_banks import AnalyticSolution

    return AnalyticSolution(
        rates=equilibrium.rates,
        losses=equilibrium.losses,
        concept=equilibrium.concept,
        condition_number=equilibrium.condition_number,
        is_well_conditioned=equilibrium.is_well_conditioned,
        note=equilibrium.note,
    )


def compare(results: tuple[ScenarioResult, ...]) -> dict[str, Any]:
    """Put several scenarios side by side.

    The comparison is the point of naming scenarios. A single number is
    hard to judge; the same number under three sets of conditions shows
    which conditions the answer is sensitive to.
    """
    if not results:
        raise EngineError("nothing to compare")

    return {
        "n_scenarios": len(results),
        "by_scenario": {
            r.scenario.name: {
                "aggregate": round(r.aggregate_consumption_change * 100, 4),
                "spread": round(r.spread * 100, 4),
                "borrowers": round(r.borrower_change * 100, 4),
                "savers": round(r.saver_change * 100, 4),
                "job_losses": r.extra_job_losses,
                "shock_scale": round(r.shock.scale, 3),
                "bank_wedge_bp": round(r.bank_wedge_bp, 1),
            }
            for r in results
        },
        "widest_spread": max(results, key=lambda r: abs(r.spread)).scenario.name,
        "largest_aggregate": max(
            results, key=lambda r: abs(r.aggregate_consumption_change)
        ).scenario.name,
    }


# ---- declared scenarios ---------------------------------------------------

FED = "Federal Reserve"
RBI = "Reserve Bank of India"
ECB = "European Central Bank"
BOE = "Bank of England"
BOJ = "Bank of Japan"

# ---- the US transmission delivered to Indian households (ADR 010) --------
#
# The Indian VAR is not credibly identified (ADR 007), so an RBI-origin
# path is the RBI's move carried by the US impulse response. Output timing
# roughly matches published Indian estimates. Inflation and persistence do
# not, and these two declarations say so wherever the proxy is used.

INFLATION_HELD_FOR_INDIA = HeldChannel(
    variable=MacroVariable.INFLATION,
    reason=(
        "The US impulse response has the wrong sign for India. After a "
        "tightening, US inflation rises for four months and the price level "
        "is still 0.14 percent higher at 36 months. Mohanty (2012), "
        "Khundrakpam and Jain (2012) and Kapur and Behera (2012) find Indian "
        "inflation falling after three to five quarters. Holding inflation at "
        "baseline asserts no price response rather than the contradicted one. "
        "See ADR 010."
    ),
)

HORIZON_CAPPED_FOR_INDIA = HorizonCap(
    periods=30,
    reason=(
        "US persistence exceeds Indian estimates. The US cumulative rate "
        "response has not halved from its month 10 peak by month 60, while "
        "published Indian estimates put the effects at eight to ten quarters. "
        "Beyond 30 months the path is US persistence without Indian support. "
        "See ADR 010."
    ),
)


US_INFLATION_SHOCK = Scenario(
    name="us_inflation_shock",
    description="US inflation reaches 4.5 percent; other economies unchanged",
    conditions=(
        Condition(
            bank=FED,
            inflation=0.045,
            note="roughly the 2023 peak, well above the 2 percent target",
        ),
    ),
    shock_origin=FED,
    rationale=(
        "The base case for the asymmetry the network models. The Fed "
        "tightens for domestic reasons and everyone else responds to a "
        "shock they did not cause."
    ),
)

IMPORTED_TIGHTENING = Scenario(
    name="imported_tightening",
    description=(
        "US inflation reaches 4.5 percent; Indian households bear the RBI's "
        "response to it"
    ),
    conditions=(
        Condition(
            bank=FED,
            inflation=0.045,
            note="the same US shock as us_inflation_shock; India is left unchanged",
        ),
    ),
    shock_origin=RBI,
    rationale=(
        "The chain the project exists to trace: the Fed tightens, the "
        "spillover reaches India, the RBI responds, Indian banks reprice and "
        "Indian households bear it. Only the Fed is conditioned, and the "
        "shock is measured against the game without that condition, so the "
        "RBI's move is wholly imported. Its size is the RBI's; its shape is "
        "the US transmission with inflation held and the horizon capped, "
        "because the Indian VAR is not credibly identified (ADRs 007, 010)."
    ),
    held_channels=(INFLATION_HELD_FOR_INDIA,),
    horizon_cap=HORIZON_CAPPED_FOR_INDIA,
)

COMMITMENT_VALUE = Scenario(
    name="commitment_value",
    description="The US shock with the Fed moving first: leadership is worth about nothing",
    conditions=(Condition(bank=FED, inflation=0.045),),
    mode=SolutionMode.LED,
    leader=FED,
    shock_origin=FED,
    rationale=(
        "A null result that follows from the network's structure. Moving "
        "first pays only if the followers' responses feed back into the "
        "leader's own inflation and output. The Fed sits at the top of the "
        "spillover hierarchy, so almost nothing does, and committing gains "
        "it nothing it would not get by moving simultaneously. Compare "
        "with us_inflation_shock: the rates agree to within a basis point. "
        "See ADR 011."
    ),
)

TWIN_TIGHTENING = Scenario(
    name="twin_tightening",
    description="Inflation above target in both the US and India",
    conditions=(
        Condition(bank=FED, inflation=0.045),
        Condition(
            bank=RBI,
            inflation=0.068,
            note="above the 6 percent ceiling, which obliges a written explanation",
        ),
    ),
    shock_origin=RBI,
    rationale=(
        "The RBI facing a domestic inflation problem and an external one "
        "at the same time. Its band breach adds an accountability cost the "
        "other banks do not carry."
    ),
    held_channels=(INFLATION_HELD_FOR_INDIA,),
    horizon_cap=HORIZON_CAPPED_FOR_INDIA,
)

GLOBAL_TIGHTENING = Scenario(
    name="global_tightening",
    description="Inflation above target across the major economies",
    conditions=(
        Condition(bank=FED, inflation=0.045),
        Condition(bank=ECB, inflation=0.040),
        Condition(bank=RBI, inflation=0.060),
    ),
    shock_origin=FED,
    rationale=(
        "The synchronised case. When everyone tightens together the "
        "spillovers compound rather than offset, which is the 2022 "
        "experience."
    ),
)

DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    US_INFLATION_SHOCK,
    IMPORTED_TIGHTENING,
    COMMITMENT_VALUE,
    TWIN_TIGHTENING,
    GLOBAL_TIGHTENING,
)

#: The scenario the pipeline reports as its headline (ADR 016).
#:
#: It is the only declared scenario that runs every link of the chain the
#: project exists to trace: US inflation moves the Fed, the central bank
#: network responds, and Indian households bear the part of the RBI's move
#: the Fed caused. Scenarios with the Fed as shock origin deliver the Fed's
#: own path to Indian households, which skips the RBI; they answer a
#: different question and are reported as comparisons, not as the finding.
HEADLINE_SCENARIO: Scenario = IMPORTED_TIGHTENING

# ---- a historical scenario: calendar 2022 (ADR 012) -----------------------
#
# The network's second out-of-sample test, after the banking layer's
# (ADR 009). Nothing was tuned to 2022. The banks start from their January
# 2022 rates, each faces its 2022 average inflation, and the solved moves
# are compared with what the banks did over the year.
#
# The default banks describe a later period, so this scenario is declared
# with its own starting state and must be run against it. The reference
# game puts every bank at its inflation target, so "caused" is what the
# 2022 inflation produced and "unprompted" is what the starting rates
# alone produce.

_JANUARY_2022_RATES: dict[str, float] = {
    FED: 0.0025,  # DFEDTARU, target range upper bound
    ECB: -0.0050,  # ECBDFR, deposit facility rate
    BOE: 0.0025,  # Bank Rate as published; not on FRED (SONIA was 0.191%)
    BOJ: -0.0002,  # IRSTCI01JPM156N, overnight call rate
    RBI: 0.0400,  # policy repo rate, RBI
}



def historical_banks(rates: dict[str, float]) -> tuple[CentralBank, ...]:
    """The major banks at historical starting rates, with no inflation problem.

    The reference state of every historical validation: each bank at its
    target with a zero output gap, so the scenario's conditions are the
    whole of what is caused. Mandates and weights are today's, whatever
    the year; a historical scenario says where that is anachronistic.
    """
    if set(rates) != {bank.name for bank in MAJOR_CENTRAL_BANKS}:
        raise EngineError(f"starting rates must cover every major bank, got {sorted(rates)}")
    return tuple(
        bank.model_copy(
            update={
                "current_rate": rates[bank.name],
                "current_inflation": bank.inflation_target,
                "current_output_gap": 0.0,
            }
        )
        for bank in MAJOR_CENTRAL_BANKS
    )


JANUARY_2022_BANKS: tuple[CentralBank, ...] = historical_banks(_JANUARY_2022_RATES)

#: What each bank did over calendar 2022, in basis points, measured on the
#: same series as the starting rates. The Bank of Japan's figure is the
#: call rate drifting; its policy balance rate stayed at -0.10 percent.
OBSERVED_2022_MOVES_BP: dict[str, float] = {
    FED: 425.0,
    ECB: 250.0,
    BOE: 325.0,
    BOJ: -5.0,
    RBI: 225.0,
}

HISTORICAL_2022 = Scenario(
    name="historical_2022",
    description=(
        "Calendar 2022: every bank from its January rate at its 2022 average "
        "inflation. Run against JANUARY_2022_BANKS"
    ),
    conditions=(
        Condition(
            bank=FED,
            inflation=0.06545,
            note="PCE price index (PCEPI), mean of 2022 monthly year-on-year rates",
        ),
        Condition(
            bank=ECB,
            inflation=0.08365,
            note="HICP (CP0000EZ19M086NEST), mean of 2022 monthly year-on-year rates",
        ),
        Condition(
            bank=BOE,
            inflation=0.07901,
            note="CPI (GBRCPIALLMINMEI), mean of 2022 monthly year-on-year rates",
        ),
        Condition(
            bank=BOJ,
            inflation=0.025,
            note=(
                "CPI all items, 2022 annual average, Statistics Bureau of Japan; "
                "FRED's OECD series ends in 2021"
            ),
        ),
        Condition(
            bank=RBI,
            inflation=0.06692,
            note=(
                "CPI Combined as published by the RBI, mean of 2022 year-on-year "
                "rates (warehouse in_cpi_inflation); peak 7.79 percent in April"
            ),
        ),
    ),
    shock_origin=RBI,
    rationale=(
        "An out-of-sample test of the network, with nothing tuned to 2022. "
        "Like for like, the model's RBI-to-Fed ratio of total moves is 58 "
        "percent against 53 observed, but both banks under-move by about a "
        "third, the ECB overshoots by 60 percent and the Bank of Japan "
        "tightens when it did not. The RBI ends outside its band, the first "
        "declared scenario to do so, which makes the simultaneous solver "
        "approximate here. See ADR 012 and scripts/validate_2022.py."
    ),
    held_channels=(INFLATION_HELD_FOR_INDIA,),
    horizon_cap=HORIZON_CAPPED_FOR_INDIA,
)


# ---- a historical scenario: calendar 2013, the taper tantrum (ADR 013) -----
#
# The same fixed design as 2022, applied to the year of the taper tantrum.
# The design conditions each bank on its inflation, and the tantrum was a
# shock to expected Fed policy with no move in the Fed's rate, so this
# tests whether the network explains 2013's rate decisions, not whether it
# reproduces the tantrum. Three anachronisms are carried, not corrected:
# the RBI had no inflation target until 2015-16, the ECB's target was
# "below but close to" two percent, and the model has no lower bound
# while four of the five banks were near zero.

_JANUARY_2013_RATES: dict[str, float] = {
    FED: 0.0025,  # DFEDTARU, target range upper bound
    ECB: 0.0000,  # ECBDFR, deposit facility rate; the MRO was 0.75%
    BOE: 0.0050,  # BOERUKM, Bank Rate
    BOJ: 0.00083,  # IRSTCI01JPM156N, overnight call rate, January average
    RBI: 0.0800,  # policy repo rate, RBI, in force on 1 January (cut on 29 January)
}

JANUARY_2013_BANKS: tuple[CentralBank, ...] = historical_banks(_JANUARY_2013_RATES)

#: What each bank did over calendar 2013, in basis points, on the same
#: series as the starting rates. The ECB cut its main refinancing rate by
#: 50bp while the deposit rate stayed at zero. The RBI's -25bp nets three
#: cuts before the tantrum against two hikes after it, and leaves out the
#: 200bp rise in its marginal standing facility rate from July to October.
OBSERVED_2013_MOVES_BP: dict[str, float] = {
    FED: 0.0,
    ECB: 0.0,
    BOE: 0.0,
    BOJ: -1.3,
    RBI: -25.0,
}

HISTORICAL_2013 = Scenario(
    name="historical_2013",
    description=(
        "Calendar 2013, the taper tantrum year: every bank from its 1 January "
        "rate at its 2013 average inflation. Run against JANUARY_2013_BANKS"
    ),
    conditions=(
        Condition(
            bank=FED,
            inflation=0.01319,
            note="PCE price index (PCEPI), mean of 2013 monthly year-on-year rates",
        ),
        Condition(
            bank=ECB,
            inflation=0.01354,
            note=(
                "HICP of the then 17-member euro area (CP0000EZ17M086NEST), mean "
                "of 2013 monthly year-on-year rates"
            ),
        ),
        Condition(
            bank=BOE,
            inflation=0.02293,
            note="CPI (GBRCPIALLMINMEI), mean of 2013 monthly year-on-year rates",
        ),
        Condition(
            bank=BOJ,
            inflation=0.00338,
            note="CPI all items (JPNCPIALLMINMEI), mean of 2013 monthly year-on-year rates",
        ),
        Condition(
            bank=RBI,
            inflation=0.10072,
            note=(
                "CPI Combined as published by the RBI, mean of 2013 year-on-year "
                "rates (warehouse in_cpi_inflation). The RBI had no CPI target in "
                "2013 and gave weight to WPI"
            ),
        ),
    ),
    shock_origin=RBI,
    rationale=(
        "The second application of the fixed design of ADR 012, to the taper "
        "tantrum year. The design can only express inflation conditions, and "
        "the tantrum was a shock to expected Fed policy with no move in the "
        "Fed's rate, so the scenario tests whether the network explains "
        "2013's rate decisions, not whether it reproduces the tantrum. It "
        "does not: the RBI hikes 153bp in the model against a 25bp cut, "
        "driven by 10 percent CPI against a target the RBI did not have. "
        "Three banks solve below zero, and the RBI ends outside its band. "
        "See ADR 013 and scripts/validate_2013.py."
    ),
    held_channels=(INFLATION_HELD_FOR_INDIA,),
    horizon_cap=HORIZON_CAPPED_FOR_INDIA,
)

#: Historical scenarios, each paired with the starting state it describes.
HISTORICAL_SCENARIOS: tuple[tuple[Scenario, tuple[CentralBank, ...]], ...] = (
    (HISTORICAL_2022, JANUARY_2022_BANKS),
    (HISTORICAL_2013, JANUARY_2013_BANKS),
)

