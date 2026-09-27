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
from moirai.engine.financial.central_banks import CentralBank
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

FED_LEADS = Scenario(
    name="fed_leads",
    description="The same shock, with the Fed moving first",
    conditions=(Condition(bank=FED, inflation=0.045),),
    mode=SolutionMode.LED,
    leader=FED,
    shock_origin=FED,
    rationale=(
        "Whether commitment changes the answer. Other central banks take "
        "Federal Reserve policy as given, so leadership is the more "
        "realistic structure, and the difference from simultaneous play "
        "measures how much that matters."
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
    FED_LEADS,
    TWIN_TIGHTENING,
    GLOBAL_TIGHTENING,
)