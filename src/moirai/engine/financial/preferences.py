"""
Estimating central bank preferences from observed policy.

The weights in central_banks.py are assumed, and the sensitivity analysis
showed the Federal Reserve's output weight is the most consequential of
them: varying it across a plausible range moves the equilibrium rate by
roughly 145 basis points. That is the model's weakest link, and it is
weak precisely where it cannot be sourced, since the Fed publishes no
numeric employment target on the grounds that maximum employment is not
directly measurable.

It can, however, be estimated. Given the actual path of the policy rate
and the inflation and output data the committee saw, one can ask which
weights make those decisions loss-minimising. That is an inverse problem
and it is how central bank preferences are estimated in the literature.

Three limitations, none of which the method removes:

  * Identification is weak. Preference weights and transmission parameters
    enter the first order condition jointly, so a strong response to
    inflation is observationally consistent with caring a great deal and
    with policy being powerful. Separating them requires fixing the
    transmission from elsewhere, which imports that model's assumptions.

  * The output gap is not observed. It requires a trend, and the estimate
    depends on the detrending choice. Results are reported for more than
    one method rather than for the one that fits best.

  * The estimate is descriptive, not normative. It recovers revealed
    preferences from behaviour, including behaviour that was mistaken. A
    period of policy error produces an estimate of the preferences that
    error implies, not of the preferences the committee held.

The result is therefore DERIVED, not SOURCED.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.game import Confidence

log = get_logger(__name__)

#: Bounds on the search. Wide enough to contain any defensible value and
#: narrow enough that an unbounded optimiser cannot wander somewhere
#: meaningless.
WEIGHT_BOUNDS = (0.0, 3.0)
SMOOTHING_BOUNDS = (0.0, 2.0)


class DetrendMethod(BaseModel):
    """How the output gap was constructed.

    Recorded with the estimate because the estimate depends on it. Two
    analysts using the same rate path and different detrending will get
    different preference weights, and neither is wrong.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameter: float | None = None
    note: str = ""


def linear_gap(series: np.ndarray) -> tuple[np.ndarray, DetrendMethod]:
    """Deviation from a fitted linear trend.

    The simplest choice and the most transparent. It assumes potential
    output grows at a constant rate, which is wrong over long samples but
    has no tuning parameter to argue about.
    """
    time = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(time, series, 1)
    return series - (slope * time + intercept), DetrendMethod(
        name="linear",
        note="constant growth in potential output; no tuning parameter",
    )


def hp_gap(series: np.ndarray, lamb: float = 14_400.0) -> tuple[np.ndarray, DetrendMethod]:
    """Hodrick-Prescott filter gap.

    Standard in the literature and widely criticised. It has no
    theoretical basis for the smoothing parameter, produces spurious
    dynamics at the ends of the sample, and Hamilton (2018) argued it
    should not be used at all. Included because comparability with
    published work is worth something, not because it is defensible.

    lambda 14400 is the monthly convention; 1600 is quarterly.
    """
    n = len(series)
    if n < 5:
        raise EngineError("the HP filter needs at least five observations")

    # Second-difference penalty matrix, solved directly.
    identity = np.eye(n)
    second = np.zeros((n - 2, n))
    for i in range(n - 2):
        second[i, i : i + 3] = [1.0, -2.0, 1.0]

    trend = np.linalg.solve(identity + lamb * second.T @ second, series)
    return series - trend, DetrendMethod(
        name="hp_filter",
        parameter=lamb,
        note=(
            "no theoretical basis for lambda; spurious end-of-sample dynamics; "
            "Hamilton (2018) argues against its use"
        ),
    )


class EstimatedPreferences(BaseModel):
    """Weights fitted to an observed policy path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: str
    inflation_weight: float
    output_weight: float
    smoothing_weight: float

    inflation_target: float
    detrend: DetrendMethod
    n_observations: int
    sample_start: str = ""
    sample_end: str = ""

    residual_std_bp: float = Field(description="Standard deviation of the fit, in bp.")
    r_squared: float
    converged: bool
    confidence: Confidence = Confidence.DERIVED

    @property
    def output_to_inflation_ratio(self) -> float:
        """Output response relative to inflation response.

        Taylor's canonical values are 0.5 and 0.5, a ratio of one. A ratio
        well below one describes a bank behaving as though its mandate were
        hierarchical whatever its statute says.

        Returns infinity when the inflation response is essentially zero,
        which signals a failed estimate rather than an extreme preference.
        """
        if abs(self.inflation_weight) < 1e-6:
            return float("inf")
        return self.output_weight / self.inflation_weight
    @property
    def satisfies_taylor_principle(self) -> bool:
        """Does the nominal rate rise more than one-for-one with inflation?

        If not, the real rate falls when inflation rises, which amplifies
        shocks rather than damping them. Clarida, Gali and Gertler argued
        the pre-Volcker Fed failed this and that the failure explains the
        Great Inflation.
        """
        return self.inflation_weight > 0.0
    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "bank": self.bank,
            "inflation_weight": round(self.inflation_weight, 4),
            "output_weight": round(self.output_weight, 4),
            "smoothing_weight": round(self.smoothing_weight, 4),
            "output_to_inflation_ratio": round(self.output_to_inflation_ratio, 4),
            "inflation_target": self.inflation_target,
            "detrend": self.detrend.model_dump(),
            "n_observations": self.n_observations,
            "sample": [self.sample_start, self.sample_end],
            "residual_std_bp": round(self.residual_std_bp, 2),
            "r_squared": round(self.r_squared, 4),
            "converged": self.converged,
            "confidence": self.confidence.value,
        }


def target_rate(
    inflation: np.ndarray,
    output_gap: np.ndarray,
    *,
    inflation_target: float,
    inflation_response: float,
    output_response: float,
    neutral_real_rate: float,
) -> np.ndarray:
    """The rate a bank would set if it could adjust freely.

    This is the first order condition of the quadratic loss, and it has
    the form of a Taylor rule with coefficients derived from preferences
    rather than imposed. Taylor's original 0.5 and 0.5 were descriptive:
    he observed they fitted Fed behaviour from 1987 to 1992. Deriving them
    means the rule can answer what a bank would do under *different*
    preferences, which a fitted rule cannot.

    `neutral_real_rate` is the weakest input. Estimates for the United
    States span roughly half a percent to two, and Holston, Laubach and
    Williams revisions move it materially. It is passed in rather than
    estimated so its influence stays visible.
    """
    return (
        neutral_real_rate
        + inflation
        + inflation_response * (inflation - inflation_target)
        + output_response * output_gap
    )


def implied_rate(
    inflation: np.ndarray,
    output_gap: np.ndarray,
    previous_rate: np.ndarray,
    *,
    inflation_target: float,
    inflation_response: float,
    output_response: float,
    adjustment_speed: float,
    neutral_real_rate: float,
) -> np.ndarray:
    """Partial adjustment toward the target rate.

    Central banks move in increments rather than jumping to the optimum,
    so the observed rate adjusts a fraction theta of the way each period.

    Separating the target from the adjustment matters for estimation. An
    earlier version folded both into one equation and the optimiser found
    that predicting the rate from its own lag explains ninety percent of
    a highly persistent series, so it drove the adjustment weight to its
    bound and the policy coefficients to zero. That is a degenerate fit,
    not a preference estimate: lagging the dependent variable beat
    modelling the rule.
    """
    target = target_rate(
        inflation,
        output_gap,
        inflation_target=inflation_target,
        inflation_response=inflation_response,
        output_response=output_response,
        neutral_real_rate=neutral_real_rate,
    )
    return previous_rate + adjustment_speed * (target - previous_rate)
def to_quarterly(series: np.ndarray, *, how: str = "mean") -> np.ndarray:
    """Aggregate a monthly series to quarterly.

    Preference estimation belongs at quarterly frequency. The FOMC meets
    eight times a year and the gaps it responds to are quarterly concepts,
    so monthly rate changes are largely noise around a quarterly decision
    process. Estimating monthly gives the inflation gap almost no
    variation to explain and the coefficient collapses to zero.

    Averaging rather than sampling the last month, because the quarterly
    average is what the policy rate actually was over the quarter, and
    end-of-quarter sampling would inherit whichever meeting happened to
    fall near the boundary.
    """
    n_quarters = len(series) // 3
    if n_quarters < 1:
        raise EngineError(f"{len(series)} months is fewer than one quarter")

    trimmed = series[: n_quarters * 3].reshape(n_quarters, 3)
    if how == "mean":
        return trimmed.mean(axis=1)
    if how == "last":
        return trimmed[:, -1]
    raise EngineError(f"unknown aggregation {how!r}")
def estimate_preferences(
    bank_name: str,
    rates: np.ndarray,
    inflation: np.ndarray,
    output: np.ndarray,
    *,
    inflation_target: float,
    detrend: str = "linear",
    neutral_real_rate: float = 0.02,
    sample_start: str = "",
    sample_end: str = "",
) -> EstimatedPreferences:
    """Fit a partial-adjustment policy rule to an observed rate path.

    Three parameters: the response to the inflation gap, the response to
    the output gap, and the speed of adjustment. Fitted to the *change* in
    the rate rather than its level, so the policy terms must explain
    something a lagged dependent variable cannot.

    The Taylor principle requires the inflation response to exceed zero in
    this parameterisation, equivalently a coefficient above one on
    inflation itself. A bank failing it raises nominal rates by less than
    inflation, so the real rate falls when inflation rises, which is
    destabilising. Whether the estimate satisfies it is reported.

    Nelder-Mead rather than a gradient method: three parameters, a smooth
    objective, and bounds that matter. Methods built for high-dimensional
    non-convex problems would be slower and no more accurate.
    """
    rates = np.asarray(rates, dtype=float)
    inflation = np.asarray(inflation, dtype=float)
    output = np.asarray(output, dtype=float)

    if not (len(rates) == len(inflation) == len(output)):
        raise EngineError(
            f"lengths differ: rates {len(rates)}, inflation {len(inflation)}, "
            f"output {len(output)}"
        )
    if len(rates) < 20:
        raise EngineError(f"{len(rates)} observations is too few")
    if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(inflation)):
        raise EngineError("rates and inflation must be finite")

    gap, method = linear_gap(output) if detrend == "linear" else hp_gap(output)

    previous = rates[:-1]
    observed_change = rates[1:] - rates[:-1]
    inflation_t = inflation[1:]
    gap_t = gap[1:]

    def objective(parameters: np.ndarray) -> float:
        inflation_response, output_response, speed = parameters
        if not (
            0.0 <= inflation_response <= 3.0
            and -0.5 <= output_response <= 3.0
            and 0.01 <= speed <= 1.0
        ):
            return 1e9

        predicted = implied_rate(
            inflation_t,
            gap_t,
            previous,
            inflation_target=inflation_target,
            inflation_response=inflation_response,
            output_response=output_response,
            adjustment_speed=speed,
            neutral_real_rate=neutral_real_rate,
        )
        predicted_change = predicted - previous
        return float(np.sum((observed_change - predicted_change) ** 2))

    best = None
    for start in ([0.5, 0.5, 0.2], [1.0, 0.5, 0.1], [0.2, 1.0, 0.3]):
        candidate = minimize(
            objective, x0=np.array(start), method="Nelder-Mead", tol=1e-12
        )
        if best is None or candidate.fun < best.fun:
            best = candidate

    assert best is not None
    inflation_response, output_response, speed = (float(v) for v in best.x)

    predicted = implied_rate(
        inflation_t,
        gap_t,
        previous,
        inflation_target=inflation_target,
        inflation_response=inflation_response,
        output_response=output_response,
        adjustment_speed=speed,
        neutral_real_rate=neutral_real_rate,
    )
    residuals = observed_change - (predicted - previous)
    variation = float(np.sum((observed_change - observed_change.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / variation if variation > 0 else 0.0

    at_bound = (
        inflation_response <= 1e-6
        or inflation_response >= 3.0 - 1e-6
        or output_response >= 3.0 - 1e-6
        or speed >= 1.0 - 1e-6
        or speed <= 0.01 + 1e-6
    )

    estimate = EstimatedPreferences(
        bank=bank_name,
        inflation_weight=inflation_response,
        output_weight=output_response,
        smoothing_weight=speed,
        inflation_target=inflation_target,
        detrend=method,
        n_observations=len(observed_change),
        sample_start=sample_start,
        sample_end=sample_end,
        residual_std_bp=float(residuals.std() * 10_000),
        r_squared=r_squared,
        converged=bool(best.success) and not at_bound,
    )

    log.info(
        "preferences_estimated",
        bank=bank_name,
        inflation_response=round(inflation_response, 4),
        output_response=round(output_response, 4),
        adjustment_speed=round(speed, 4),
        r_squared=round(r_squared, 4),
        at_bound=at_bound,
    )

    if at_bound:
        log.warning(
            "estimate_at_bound",
            bank=bank_name,
            note="a parameter hit its bound, so this is not an interior estimate",
        )
    return estimate

def compare_detrending(
    bank_name: str,
    rates: np.ndarray,
    inflation: np.ndarray,
    output: np.ndarray,
    **options: Any,
) -> dict[str, EstimatedPreferences]:
    """Estimate under each detrending method.

    Reported together rather than choosing the one that fits best. If the
    two disagree materially, the output gap is doing the work and the
    estimate should be quoted as a range.
    """
    return {
        method: estimate_preferences(
            bank_name, rates, inflation, output, detrend=method, **options
        )
        for method in ("linear", "hp")
    }