"""
Translate an estimated impulse response into the macro path households face.

This is the seam between Layer 1 and Layer 3, and it is where units get
lost if it is done casually. The IRF speaks in the VAR's transformed
space: log differences, first differences, and one standard deviation
shocks. A household needs an interest rate it pays, an inflation rate that
erodes its wage, and an income growth rate. The translation between the
two is arithmetic, but it is arithmetic that fails silently, so it happens
here once rather than at every call site.

Two things this makes possible that most agent-based models do not have:

  * The aggregate shock is estimated rather than assumed. "Suppose rates
    rise one hundred basis points and stay there" is a guess. A path
    derived from an identified VAR is a claim about how this economy has
    actually behaved, with the identifying assumption stated.

  * Uncertainty propagates. Bootstrap bands can be carried through the
    simulation, so the output is a distribution of distributional
    outcomes rather than a single scenario presented as fact.

What it deliberately does not do is close the loop. Household responses
here do not feed back into the aggregate path. Doing that properly is the
heterogeneous agent general equilibrium problem, which is a research
programme rather than a function.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.diagnostics import DiagnosticReport
from moirai.engine.causal.irf import ConfidenceBands, ImpulseResponse
from moirai.engine.causal.preparation import Transformation

log = get_logger(__name__)


class MacroVariable(StrEnum):
    """The macro quantities a household actually responds to.

    Deliberately small. Every entry here must have a behavioural channel
    in the household rules; adding one without a channel would produce a
    variable that is computed, reported, and ignored.
    """


    POLICY_RATE = "policy_rate"
    INFLATION = "inflation"
    INCOME_GROWTH = "income_growth"
    UNEMPLOYMENT = "unemployment"
#: Ranges outside which a macro path is almost certainly a unit error
#: rather than an economy. Deliberately wide: hyperinflations and
#: emergency rate settings are real, and this guard exists to catch a
#: factor of twelve or a hundred, not to police economic plausibility.
PLAUSIBLE_RANGES: dict[MacroVariable, tuple[float, float]] = {
    MacroVariable.POLICY_RATE: (-0.05, 1.00),
    MacroVariable.INFLATION: (-0.30, 2.00),
    MacroVariable.INCOME_GROWTH: (-0.50, 1.00),
    MacroVariable.UNEMPLOYMENT: (0.0, 0.80),
}

class VariableMapping(BaseModel):
    """How one VAR variable becomes one macro variable a household sees.

    `transformation` records what was done to the series before estimation,
    because undoing it correctly is the whole point. A log-differenced
    series is a growth rate per period; a differenced rate series is a
    change in percentage points; a level series needs neither.

    `scale` converts units. FRED reports the federal funds rate in percent,
    so a response of 0.21 means twenty one basis points and must be divided
    by one hundred to become a decimal rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    var_variable: str = Field(min_length=1)
    macro_variable: MacroVariable
    transformation: Transformation
    scale: float = Field(default=1.0, description="Multiply the response by this.")
    cumulate: bool = Field(
        default=False,
        description="Accumulate the response. Required for differenced series.",
    )
    baseline: float = Field(
        default=0.0,
        description="Level the economy sits at before the shock.",
    )

    @model_validator(mode="after")
    def _cumulation_matches_transformation(self) -> Self:
        differenced = {
            Transformation.DIFFERENCE,
            Transformation.LOG_DIFFERENCE,
            Transformation.PERCENT_CHANGE,
        }
        if self.transformation in differenced and not self.cumulate and self.macro_variable in (
            MacroVariable.POLICY_RATE,
            MacroVariable.UNEMPLOYMENT,
        ):
            raise ValueError(
                f"{self.macro_variable.value} is a level, but "
                f"{self.var_variable} was {self.transformation.value}. "
                f"Set cumulate=True or the path will describe changes, "
                f"not levels."
            )
        return self


class ShockPath(BaseModel):
    """The macro path an economy follows after an identified shock.

    Each array has length horizon + 1 and is indexed by period since the
    shock, so index zero is the impact period. Values are levels, not
    deviations: `policy_rate[3]` is the rate itself in period three, with
    the baseline already added.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    shock_name: str
    horizon: int = Field(ge=0)
    paths: dict[MacroVariable, Any]
    baselines: dict[MacroVariable, float]
    scale_factor: float = Field(default=1.0, description="Multiple of a one-sd shock.")
    scheme: str = ""
    assumptions: str = ""
    diagnostics_usable: bool | None = None

    @model_validator(mode="after")
    def _paths_share_a_length(self) -> Self:
        expected = self.horizon + 1
        for variable, path in self.paths.items():
            if path.shape != (expected,):
                raise ValueError(
                    f"path for {variable.value} has shape {path.shape}, "
                    f"expected ({expected},)"
                )
        return self

    def __len__(self) -> int:
        return self.horizon + 1

    def get(self, variable: MacroVariable) -> np.ndarray:
        if variable not in self.paths:
            raise EngineError(
                f"{variable.value} is not in this shock path; "
                f"have {[v.value for v in self.paths]}"
            )
        return self.paths[variable]

    def at(self, period: int) -> dict[MacroVariable, float]:
        """Every macro variable at one period, which is what a household reads."""
        if not 0 <= period <= self.horizon:
            raise EngineError(f"period {period} is outside 0..{self.horizon}")
        return {variable: float(path[period]) for variable, path in self.paths.items()}

    def deviation(self, variable: MacroVariable) -> np.ndarray:
        """The path expressed as a deviation from its baseline."""
        return self.get(variable) - self.baselines[variable]

    def peak(self, variable: MacroVariable) -> tuple[int, float]:
        """Period and value of the largest deviation from baseline."""
        deviations = self.deviation(variable)
        index = int(np.argmax(np.abs(deviations)))
        return index, float(deviations[index])

    def rescale(self, factor: float) -> ShockPath:
        """Scale the shock. A two-standard-deviation shock is factor 2.

        Linear, because a VAR is linear. That is an assumption worth
        naming: it says a large shock is exactly ten times a small one,
        which rules out the nonlinearity that makes financial crises
        interesting.
        """
        if factor == 0:
            raise EngineError("scaling to zero produces no shock")

        rescaled = {
            variable: self.baselines[variable]
            + (path - self.baselines[variable]) * factor
            for variable, path in self.paths.items()
        }
        return self.model_copy(
            update={"paths": rescaled, "scale_factor": self.scale_factor * factor}
        )

    def hold(self, variable: MacroVariable) -> ShockPath:
        """Pin one variable at its baseline, removing its channel.

        For when the estimated response of that variable is not credible
        for the economy the households live in. Holding it asserts no
        response, which is weaker than asserting the wrong one, but it is
        still an assumption and the caller should record why.
        """
        path = self.get(variable)
        held = {**self.paths, variable: np.full_like(path, self.baselines[variable])}
        return self.model_copy(update={"paths": held})

    def truncate(self, horizon: int) -> ShockPath:
        """Keep periods 0..horizon and drop the rest.

        Households are simulated for as many periods as the path has, so
        this ends the simulation early rather than returning the economy
        to baseline. A horizon at or beyond the current one is a no-op.
        """
        if horizon < 0:
            raise EngineError(f"horizon must be non-negative, got {horizon}")
        if horizon >= self.horizon:
            return self
        cut = {variable: path[: horizon + 1].copy() for variable, path in self.paths.items()}
        return self.model_copy(update={"paths": cut, "horizon": horizon})

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "shock_name": self.shock_name,
            "horizon": self.horizon,
            "scale_factor": self.scale_factor,
            "scheme": self.scheme,
            "assumptions": self.assumptions,
            "diagnostics_usable": self.diagnostics_usable,
            "variables": [v.value for v in self.paths],
            "baselines": {v.value: b for v, b in self.baselines.items()},
            "peaks": {
                v.value: {"period": self.peak(v)[0], "deviation": round(self.peak(v)[1], 8)}
                for v in self.paths
            },
        }

def _check_plausible(
    variable: MacroVariable, path: np.ndarray, *, strict: bool
) -> None:
    """Reject a path whose magnitude implies a unit error.

    The seam between an impulse response and a household simulation is
    where unit errors live, and they are silent: every function succeeds,
    every type is right, and the answer is wrong by a factor of twelve.
    One such error survived to produce a consumption response of two and a
    half million percent, because nothing between the layers had an
    opinion about magnitude.

    The bounds are wide on purpose. What this catches is an already-annual
    rate multiplied by periods per year, or percentage points read as
    decimals, which is the mistake that actually happens.
    """
    low, high = PLAUSIBLE_RANGES[variable]
    below, above = float(path.min()), float(path.max())

    if below >= low and above <= high:
        return

    message = (
        f"{variable.value} path reaches [{below:.3f}, {above:.3f}], outside "
        f"the plausible range [{low}, {high}]. Values are decimals, so 0.05 "
        f"means five percent. A path this large usually means the mapping "
        f"scale is wrong. Pass check_plausibility=False to proceed anyway."
    )
    if strict:
        raise EngineError(message)
    log.warning(
        "implausible_shock_path",
        variable=variable.value,
        peak=float(np.max(np.abs(path))),
    )

def build_shock_path(
    responses: ImpulseResponse,
    shock_name: str,
    mappings: tuple[VariableMapping, ...],
    *,
    scale: float = 1.0,
    diagnostics: DiagnosticReport | None = None,
    require_usable: bool = True,
    check_plausibility: bool = True,
) -> ShockPath:
    """Turn an impulse response into a macro path households can face.

    Parameters
    ----------
    require_usable
        Refuse to build a path from a VAR that failed critical
        specification checks. This is a hard gate rather than a warning:
        a simulation built on serially correlated residuals inherits the
        problem and buries it under a million households, where nobody
        will find it.
    """
    if not mappings:
        raise EngineError("at least one variable mapping is required")
    if shock_name not in responses.shock_names:
        raise EngineError(
            f"unknown shock {shock_name!r}; have {list(responses.shock_names)}"
        )

    if diagnostics is not None and require_usable and not diagnostics.is_usable:
        raise EngineError(
            f"refusing to build a shock path from a VAR that failed critical "
            f"specification checks: {diagnostics.summary()}. Pass "
            f"require_usable=False to override, which will be recorded."
        )

    seen: set[MacroVariable] = set()
    paths: dict[MacroVariable, np.ndarray] = {}
    baselines: dict[MacroVariable, float] = {}

    for mapping in mappings:
        if mapping.macro_variable in seen:
            raise EngineError(f"{mapping.macro_variable.value} is mapped twice")
        seen.add(mapping.macro_variable)

        raw = responses.path(mapping.var_variable, shock_name)
        deviation = np.cumsum(raw) if mapping.cumulate else raw.copy()
        deviation = deviation * mapping.scale * scale

        paths[mapping.macro_variable] = mapping.baseline + deviation
        _check_plausible(
            mapping.macro_variable,
            paths[mapping.macro_variable],
            strict=check_plausibility,
        )
        baselines[mapping.macro_variable] = mapping.baseline

    path = ShockPath(
        shock_name=shock_name,
        horizon=responses.horizon,
        paths=paths,
        baselines=baselines,
        scale_factor=scale,
        scheme=responses.scheme,
        assumptions=responses.assumptions,
        diagnostics_usable=None if diagnostics is None else diagnostics.is_usable,
    )

    log.info(
        "shock_path_built",
        shock=shock_name,
        horizon=responses.horizon,
        scale=scale,
        variables=[v.value for v in paths],
        diagnostics_usable=path.diagnostics_usable,
    )
    return path


def paths_from_bands(
    bands: ConfidenceBands,
    shock_name: str,
    mappings: tuple[VariableMapping, ...],
    level: float,
    *,
    scale: float = 1.0,
) -> tuple[ShockPath, ShockPath, ShockPath]:
    """Build lower, central and upper shock paths from bootstrap bands.

    Running the simulation on all three turns a point scenario into a
    range. The resulting spread is parameter uncertainty in the estimated
    dynamics, and it is separate from, and usually smaller than, the
    uncertainty in how households respond.
    """
    if level not in bands.lower:
        raise EngineError(f"no band at level {level}; have {list(bands.levels)}")

    def build(responses: np.ndarray) -> ShockPath:
        surrogate = bands.point.model_copy(update={"responses": responses})
        return build_shock_path(surrogate, shock_name, mappings, scale=scale)

    return (
        build(bands.lower[level]),
        build(bands.point.responses),
        build(bands.upper[level]),
    )


# ---- conventional mappings ------------------------------------------------


def monetary_mappings(
    rate_variable: str,
    price_variable: str,
    output_variable: str,
    *,
    baseline_rate: float = 0.065,
    baseline_inflation: float = 0.05,
    baseline_income_growth: float = 0.06,
    periods_per_year: float = 12.0,
) -> tuple[VariableMapping, ...]:
    """Standard mappings for a three-variable monetary VAR.

    Assumes the conventional preparation: the rate differenced, prices and
    output log-differenced. Baselines are annual rates that the economy
    sits at before the shock, and the responses are converted to the same
    frequency so they can be added to them.

    Note the income growth mapping treats the output response as a proxy
    for household income growth. That is a modelling assumption, not an
    estimate: it says aggregate output and household income move together
    one for one, which understates how unevenly recessions land.
    """
    return (
        VariableMapping(
            var_variable=rate_variable,
            macro_variable=MacroVariable.POLICY_RATE,
            transformation=Transformation.DIFFERENCE,
            scale=0.01,  # percent to decimal
            cumulate=True,
            baseline=baseline_rate,
        ),
        VariableMapping(
            var_variable=price_variable,
            macro_variable=MacroVariable.INFLATION,
            transformation=Transformation.LOG_DIFFERENCE,
            scale=periods_per_year,  # per-period growth to annualised
            cumulate=False,
            baseline=baseline_inflation,
        ),
        VariableMapping(
            var_variable=output_variable,
            macro_variable=MacroVariable.INCOME_GROWTH,
            transformation=Transformation.LOG_DIFFERENCE,
            scale=periods_per_year,
            cumulate=False,
            baseline=baseline_income_growth,
        ),
    )
def yoy_percentage_point_mappings(
    rate_variable: str,
    inflation_variable: str,
    growth_variable: str,
    *,
    baseline_rate: float,
    baseline_inflation: float,
    baseline_income_growth: float,
) -> tuple[VariableMapping, ...]:
    """Mappings for a VAR whose variables are already annual rates.

    RBI publishes industrial production and CPI as year-on-year percent
    changes, so an impulse response of -0.16 means -0.16 percentage points
    per annum. The only conversion needed is percentage points to decimal.

    This differs from monetary_mappings, which assumes log-differenced
    levels and annualises by multiplying by the periods per year. Applying
    that here multiplies an already annual rate by twelve, which produces
    inflation paths in the hundreds of percent. The two mapping sets exist
    separately because the error is silent: the arithmetic succeeds and
    only the magnitude is absurd.
    """
    return (
        VariableMapping(
            var_variable=rate_variable,
            macro_variable=MacroVariable.POLICY_RATE,
            transformation=Transformation.DIFFERENCE,
            scale=0.01,
            cumulate=True,  # the VAR modelled the change in the rate
            baseline=baseline_rate,
        ),
        VariableMapping(
            var_variable=inflation_variable,
            macro_variable=MacroVariable.INFLATION,
            transformation=Transformation.NONE,
            scale=0.01,
            cumulate=False,  # already a level, in rate space
            baseline=baseline_inflation,
        ),
        VariableMapping(
            var_variable=growth_variable,
            macro_variable=MacroVariable.INCOME_GROWTH,
            transformation=Transformation.NONE,
            scale=0.01,
            cumulate=False,
            baseline=baseline_income_growth,
        ),
    )