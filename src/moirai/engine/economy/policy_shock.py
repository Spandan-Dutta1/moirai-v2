"""
The seam between the policy game and the transmission layer.

Until now the shock fed to households was scaled arbitrarily: two standard
deviations, chosen because it produced a visible response. That is a
hypothetical, not a scenario. This module replaces it with a shock whose
size comes from the game, so the chain runs

    conditions -> equilibrium rate -> deviation -> IRF scaling -> households

and every step is either estimated or derived from a stated mandate.

The translation is where a unit error would hide. The game speaks in rate
levels: the Fed's optimal rate is 5.72 percent against a current 4.25, so
a deviation of 147 basis points. The impulse response speaks in standard
deviations: a one standard deviation shock moves the funds rate by 21
basis points on impact. The scale factor is therefore the ratio, and a
147 basis point move is about seven standard deviations.

That arithmetic is reported rather than performed silently, because a
factor-of-twelve error at exactly this seam once produced a consumption
response of two and a half million percent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.diagnostics import DiagnosticReport
from moirai.engine.causal.irf import ImpulseResponse
from moirai.engine.economy.shock_path import ShockPath, VariableMapping, build_shock_path
from moirai.engine.financial.central_banks import AnalyticSolution, CentralBank

log = get_logger(__name__)

#: Above this many standard deviations, the shock is outside anything the
#: VAR sample contains and the linear extrapolation is doing all the work.
EXTREME_SCALE = 10.0


class PolicyShock(BaseModel):
    """A shock whose size was chosen by a policy maker, not by the analyst.

    Carries the full derivation: what the bank's equilibrium rate was, what
    it moved from, how large that is in standard deviations of the
    estimated shock, and whether the resulting scale is inside the range
    the VAR sample supports.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: str
    equilibrium_rate: float
    current_rate: float = Field(description="Where the bank's rate sits; the path starts here.")
    reference_rate: float = Field(
        description=(
            "What the move is measured against. In a scenario this is the "
            "bank's rate in the same game solved without the scenario's "
            "conditions, so the shock excludes the move the bank would make "
            "anyway."
        )
    )
    impact_response: float = Field(
        description="One standard deviation move in the policy rate, on impact."
    )
    scale: float = Field(description="Deviation divided by the impact response.")
    concept: str = Field(description="Which solution concept produced the rate.")
    is_extreme: bool
    note: str = ""

    @property
    def deviation(self) -> float:
        return self.equilibrium_rate - self.reference_rate

    @property
    def deviation_bp(self) -> float:
        return self.deviation * 10_000

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "bank": self.bank,
            "concept": self.concept,
            "current_rate": round(self.current_rate, 6),
            "reference_rate": round(self.reference_rate, 6),
            "equilibrium_rate": round(self.equilibrium_rate, 6),
            "deviation_bp": round(self.deviation_bp, 1),
            "impact_response": round(self.impact_response, 6),
            "scale": round(self.scale, 4),
            "is_extreme": self.is_extreme,
            "note": self.note,
        }


def shock_from_equilibrium(
    solution: AnalyticSolution,
    bank: CentralBank,
    responses: ImpulseResponse,
    shock_name: str,
    rate_variable: str,
    *,
    reference_rate: float | None = None,
) -> PolicyShock:
    """Convert a game equilibrium into an impulse response scale factor.

    Parameters
    ----------
    solution
        The game outcome, holding each bank's equilibrium rate.
    bank
        Whose move drives the shock.
    responses
        The estimated impulse responses, in the VAR's own units.
    rate_variable
        Which VAR variable is the policy rate.
    reference_rate
        What the move is measured against. Omitted, it is the bank's
        `current_rate`, which counts every step toward the game's
        equilibrium as shock, including the step the bank would take with
        no conditions applied at all. A scenario passes the rate from the
        game solved without its conditions, so the shock is only the part
        the conditions cause.

    Raises
    ------
    EngineError
        If the impulse response has no impact effect on the policy rate,
        which would make the scale undefined.
    """
    if bank.name not in solution.rates:
        raise EngineError(
            f"{bank.name!r} is not in this solution; have {sorted(solution.rates)}"
        )

    equilibrium = solution.rates[bank.name]
    reference = bank.current_rate if reference_rate is None else reference_rate
    deviation = equilibrium - reference

    # The impact response is in the VAR's units. A differenced percent
    # series gives percentage points, so it converts to a decimal here.
    impact = float(responses.path(rate_variable, shock_name)[0]) / 100.0

    # The rate response accumulates: the VAR was estimated on a differenced
    # rate, so the cumulative path peaks well above its impact value. The
    # game's equilibrium is a level the bank wants to reach, so the peak is
    # what should match it. Scaling on impact instead overshoots by the
    # ratio of peak to impact, which for a persistent series is large: a
    # 147 basis point target became a 570 basis point path.
    raw = responses.path(rate_variable, shock_name)
    cumulative = np.cumsum(raw) / 100.0
    peak_index = int(np.argmax(np.abs(cumulative)))
    impact = float(cumulative[peak_index])

    if abs(impact) < 1e-9:
        raise EngineError(
            f"the cumulative impulse response for {rate_variable!r} is flat, "
            f"so a policy move cannot be expressed as a multiple of it. Check "
            f"that the named variable is the policy rate."
        )

    scale = deviation / impact
    is_extreme = abs(scale) > EXTREME_SCALE

    measured_from = (
        f"from {reference:.2%}"
        if reference_rate is None
        else f"against {reference:.2%} without the conditions"
    )
    note = (
        f"{bank.name} moves {deviation * 10_000:+.0f}bp {measured_from}, "
        f"to {equilibrium:.2%}; a one standard deviation shock moves the rate "
        f"{impact * 10_000:.0f}bp at its cumulative peak (period {peak_index}), "
        f"so this is {scale:.1f} standard deviations"
    )
    if is_extreme:
        note += (
            ". That is outside anything the estimation sample contains, so the "
            "linear extrapolation is doing the work rather than the data"
        )

    shock = PolicyShock(
        bank=bank.name,
        equilibrium_rate=equilibrium,
        current_rate=bank.current_rate,
        reference_rate=reference,
        impact_response=impact,
        scale=scale,
        concept=solution.concept,
        is_extreme=is_extreme,
        note=note,
    )

    log.info(
        "policy_shock_derived",
        bank=bank.name,
        concept=solution.concept,
        deviation_bp=round(shock.deviation_bp, 1),
        scale=round(scale, 3),
        extreme=is_extreme,
    )
    if is_extreme:
        log.warning(
            "extreme_shock_scale",
            scale=round(scale, 2),
            note="beyond the range the VAR sample supports",
        )
    return shock



def path_from_game(
    solution: AnalyticSolution,
    bank: CentralBank,
    responses: ImpulseResponse,
    shock_name: str,
    rate_variable: str,
    mappings: tuple[VariableMapping, ...],
    *,
    diagnostics: DiagnosticReport | None = None,
    require_usable: bool = True,
    allow_extreme: bool = False,
    reference_rate: float | None = None,
) -> tuple[ShockPath, PolicyShock]:
    """Build the household-facing macro path from a game equilibrium.

    Returns both the path and the derivation, so a result can be traced
    back to the conditions and mandate that produced the policy move
    rather than to a scale somebody chose.

    `allow_extreme` must be set explicitly for a shock beyond ten standard
    deviations. Refusing by default is the same principle as the
    specification gate: at that size the answer comes from extrapolating a
    linear model far outside its sample, and that should be a decision
    rather than an accident.

    `reference_rate` is passed through to `shock_from_equilibrium`.
    """
    shock = shock_from_equilibrium(
        solution, bank, responses, shock_name, rate_variable,
        reference_rate=reference_rate,
    )

    if shock.is_extreme and not allow_extreme:
        raise EngineError(
            f"the implied shock is {shock.scale:.1f} standard deviations, beyond "
            f"the range the estimation sample supports. Pass allow_extreme=True "
            f"to proceed, which will be recorded."
        )

    path = build_shock_path(
        responses,
        shock_name,
        mappings,
        scale=shock.scale,
        diagnostics=diagnostics,
        require_usable=require_usable,
    )
    return path, shock

def mappings_for_bank(
    bank: CentralBank,
    rate_variable: str,
    price_variable: str,
    output_variable: str,
    *,
    periods_per_year: float = 12.0,
) -> tuple[VariableMapping, ...]:
    """Build mappings anchored to the bank's own current state.

    Accepting a bank and a separately-constructed set of mappings lets the
    two disagree about where the economy starts. That happened: the game
    solved from a 4.25 percent policy rate while the mappings defaulted to
    6.5, so the resulting path was 225 basis points adrift from the rate
    the game had chosen. Deriving the baselines here makes the
    disagreement unrepresentable.

    Income growth comes from the bank for the same reason. It used to be a
    separate argument defaulting to two percent, so switching the origin
    to the RBI gave Indian households an Indian rate and inflation but
    American income growth.

    Raises
    ------
    EngineError
        If the bank declares no `current_income_growth`.
    """
    from moirai.engine.causal.preparation import Transformation
    from moirai.engine.economy.shock_path import MacroVariable

    if bank.current_income_growth is None:
        raise EngineError(
            f"{bank.name} declares no current_income_growth, so a household "
            f"path originating there has no income baseline. Set it on the "
            f"bank rather than passing a default."
        )

    return (
        VariableMapping(
            var_variable=rate_variable,
            macro_variable=MacroVariable.POLICY_RATE,
            transformation=Transformation.DIFFERENCE,
            scale=0.01,
            cumulate=True,
            baseline=bank.current_rate,
        ),
        VariableMapping(
            var_variable=price_variable,
            macro_variable=MacroVariable.INFLATION,
            transformation=Transformation.LOG_DIFFERENCE,
            scale=periods_per_year,
            cumulate=False,
            baseline=bank.current_inflation,
        ),
        VariableMapping(
            var_variable=output_variable,
            macro_variable=MacroVariable.INCOME_GROWTH,
            transformation=Transformation.LOG_DIFFERENCE,
            scale=periods_per_year,
            cumulate=False,
            baseline=bank.current_income_growth,
        ),
    )