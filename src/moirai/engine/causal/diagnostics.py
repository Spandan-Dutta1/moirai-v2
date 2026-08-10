"""
Specification diagnostics for an estimated VAR.

Every result in this layer rests on assumptions about the residuals. If
they are autocorrelated, the lag order is wrong, the standard errors are
wrong, and the bootstrap bands are too narrow, which makes an effect look
more certain than the data supports. That is the failure this project
exists to avoid, so the checks are here rather than left to the analyst.

What each test asks:

    Portmanteau / Ljung-Box   are residuals serially uncorrelated?
    ARCH-LM                   is the variance constant over time?
    Jarque-Bera               are residuals normal?
    Granger causality         do lags of x help predict y?
    CUSUM                     are the coefficients stable across the sample?

A note on how these are reported. Failing normality is common in macro
data and rarely fatal, since OLS remains consistent. Failing serial
correlation is serious, because it means the model has not captured the
dynamics it claims to. The verdict distinguishes the two rather than
reducing everything to a pass or a fail.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.var import VARResult, estimate_var

log = get_logger(__name__)

DEFAULT_SIGNIFICANCE = 0.05


class Severity(StrEnum):
    """How much a failed test should change what you do next.

    CRITICAL failures invalidate the inference. ADVISORY failures are
    common in macro samples and usually mean robust standard errors or a
    caveat, not a different model.
    """

    CRITICAL = "critical"
    ADVISORY = "advisory"


class CheckOutcome(BaseModel):
    """Result of one specification test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    statistic: float
    pvalue: float
    null_hypothesis: str
    severity: Severity
    significance: float = Field(default=DEFAULT_SIGNIFICANCE, gt=0, lt=1)
    detail: str = ""

    @property
    def rejects_null(self) -> bool:
        return self.pvalue < self.significance

    @property
    def passed(self) -> bool:
        """Passing means failing to reject the null, which is what we want here.

        Note this is weaker than it sounds: not rejecting is not evidence
        for the null, only absence of evidence against it. A short sample
        passes most tests because it has no power to fail them.
        """
        return not self.rejects_null

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": round(self.statistic, 6),
            "pvalue": round(self.pvalue, 6),
            "null_hypothesis": self.null_hypothesis,
            "severity": self.severity.value,
            "passed": self.passed,
            "detail": self.detail,
        }


class DiagnosticReport(BaseModel):
    """All specification checks for one estimated VAR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variables: tuple[str, ...]
    n_lags: int
    n_observations: int
    outcomes: tuple[CheckOutcome, ...]

    @property
    def critical_failures(self) -> tuple[CheckOutcome, ...]:
        return tuple(
            o for o in self.outcomes if not o.passed and o.severity is Severity.CRITICAL
        )

    @property
    def advisory_failures(self) -> tuple[CheckOutcome, ...]:
        return tuple(
            o for o in self.outcomes if not o.passed and o.severity is Severity.ADVISORY
        )

    @property
    def is_usable(self) -> bool:
        """No critical failure. Advisory failures are noted, not disqualifying."""
        return not self.critical_failures

    def summary(self) -> str:
        if self.is_usable and not self.advisory_failures:
            return "All specification checks passed."
        parts = []
        if self.critical_failures:
            names = ", ".join(o.name for o in self.critical_failures)
            parts.append(f"CRITICAL: {names}")
        if self.advisory_failures:
            names = ", ".join(o.name for o in self.advisory_failures)
            parts.append(f"advisory: {names}")
        return "; ".join(parts)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "n_lags": self.n_lags,
            "n_observations": self.n_observations,
            "is_usable": self.is_usable,
            "summary": self.summary(),
            "outcomes": [o.to_ledger_dict() for o in self.outcomes],
        }


# ---- serial correlation ---------------------------------------------------


def portmanteau_test(
    var: VARResult, n_lags: int = 12, *, significance: float = DEFAULT_SIGNIFICANCE
) -> CheckOutcome:
    """Multivariate Ljung-Box test for residual autocorrelation.

    The most important test here. Serially correlated residuals mean the
    lag order is too short: dynamics the model should have captured are
    still sitting in the error term. Every standard error and every
    confidence band downstream is then understated.

    Degrees of freedom subtract the k^2 p coefficients already estimated,
    which is what distinguishes this from applying Ljung-Box naively to a
    residual series.
    """
    residuals = var.residuals
    t, k = residuals.shape

    if n_lags >= t:
        raise EngineError(f"cannot test {n_lags} lags with {t} residual observations")

    c0 = residuals.T @ residuals / t
    try:
        c0_inverse = np.linalg.inv(c0)
    except np.linalg.LinAlgError as err:
        raise EngineError("residual covariance is singular") from err

    statistic = 0.0
    for lag in range(1, n_lags + 1):
        c_lag = residuals[lag:].T @ residuals[:-lag] / t
        term = c_lag.T @ c0_inverse @ c_lag @ c0_inverse
        statistic += float(np.trace(term)) / (t - lag)

    statistic *= t * (t + 2)

    degrees = k * k * (n_lags - var.n_lags)
    if degrees <= 0:
        raise EngineError(
            f"test lags ({n_lags}) must exceed model lags ({var.n_lags}) "
            f"for the statistic to have positive degrees of freedom"
        )

    pvalue = float(stats.chi2.sf(statistic, degrees))

    return CheckOutcome(
        name="portmanteau",
        statistic=statistic,
        pvalue=pvalue,
        null_hypothesis="residuals are serially uncorrelated up to the tested lag",
        severity=Severity.CRITICAL,
        significance=significance,
        detail=f"{n_lags} lags tested, {degrees} degrees of freedom",
    )


def ljung_box_per_equation(
    var: VARResult, n_lags: int = 12, *, significance: float = DEFAULT_SIGNIFICANCE
) -> tuple[CheckOutcome, ...]:
    """Univariate Ljung-Box on each equation's residuals.

    Complements the portmanteau test by localising a failure. A joint
    rejection says something is wrong; these say which equation.
    """
    residuals = var.residuals
    t = residuals.shape[0]
    outcomes = []

    for index, name in enumerate(var.variables):
        series = residuals[:, index]
        series = series - series.mean()
        variance = float(series @ series)

        statistic = 0.0
        for lag in range(1, n_lags + 1):
            autocov = float(series[lag:] @ series[:-lag])
            rho = autocov / variance
            statistic += rho * rho / (t - lag)
        statistic *= t * (t + 2)

        degrees = max(n_lags - var.n_lags, 1)
        outcomes.append(
            CheckOutcome(
                name=f"ljung_box[{name}]",
                statistic=statistic,
                pvalue=float(stats.chi2.sf(statistic, degrees)),
                null_hypothesis=f"residuals of {name} are serially uncorrelated",
                severity=Severity.CRITICAL,
                significance=significance,
                detail=f"{n_lags} lags, {degrees} degrees of freedom",
            )
        )
    return tuple(outcomes)


# ---- heteroskedasticity ---------------------------------------------------


def arch_lm_test(
    var: VARResult, n_lags: int = 5, *, significance: float = DEFAULT_SIGNIFICANCE
) -> tuple[CheckOutcome, ...]:
    """Engle's ARCH-LM test for conditional heteroskedasticity.

    Regresses squared residuals on their own lags. A rejection means
    volatility clusters, which is near-universal in financial data and
    common in macro data spanning the Great Moderation.

    Advisory rather than critical: OLS stays consistent under
    heteroskedasticity, the coefficient estimates are still unbiased, and
    the usual fix is robust standard errors rather than a different model.
    """
    residuals = var.residuals
    t = residuals.shape[0]
    outcomes = []

    for index, name in enumerate(var.variables):
        squared = residuals[:, index] ** 2
        effective = t - n_lags

        design = np.ones((effective, n_lags + 1))
        for lag in range(1, n_lags + 1):
            design[:, lag] = squared[n_lags - lag : t - lag]
        target = squared[n_lags:]

        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ beta
        centred = target - target.mean()

        total = float(centred @ centred)
        explained = float((fitted - target.mean()) @ (fitted - target.mean()))
        r_squared = 0.0 if total == 0 else explained / total

        statistic = effective * r_squared
        outcomes.append(
            CheckOutcome(
                name=f"arch_lm[{name}]",
                statistic=statistic,
                pvalue=float(stats.chi2.sf(statistic, n_lags)),
                null_hypothesis=f"residual variance of {name} is constant",
                severity=Severity.ADVISORY,
                significance=significance,
                detail=f"{n_lags} lags of squared residuals",
            )
        )
    return tuple(outcomes)


# ---- normality ------------------------------------------------------------


def jarque_bera_test(
    var: VARResult, *, significance: float = DEFAULT_SIGNIFICANCE
) -> tuple[CheckOutcome, ...]:
    """Jarque-Bera test on each equation's residuals.

    Advisory. Macro residuals are routinely non-normal: recessions produce
    fat tails and skew. OLS remains consistent, and the bootstrap used for
    confidence bands does not assume normality either, which is one reason
    it was preferred over analytic standard errors.
    """
    residuals = var.residuals
    t = residuals.shape[0]
    outcomes = []

    for index, name in enumerate(var.variables):
        series = residuals[:, index]
        centred = series - series.mean()
        variance = float(centred @ centred) / t
        if variance == 0:
            raise EngineError(f"residuals of {name} have zero variance")

        deviation = np.sqrt(variance)
        skewness = float((centred**3).mean()) / deviation**3
        kurtosis = float((centred**4).mean()) / deviation**4

        statistic = t / 6.0 * (skewness**2 + 0.25 * (kurtosis - 3.0) ** 2)
        outcomes.append(
            CheckOutcome(
                name=f"jarque_bera[{name}]",
                statistic=statistic,
                pvalue=float(stats.chi2.sf(statistic, 2)),
                null_hypothesis=f"residuals of {name} are normally distributed",
                severity=Severity.ADVISORY,
                significance=significance,
                detail=f"skewness {skewness:.3f}, kurtosis {kurtosis:.3f}",
            )
        )
    return tuple(outcomes)


# ---- Granger causality ----------------------------------------------------


def granger_causality(
    var: VARResult,
    cause: str,
    effect: str,
    data: np.ndarray,
    *,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> CheckOutcome:
    """Test whether lags of `cause` improve prediction of `effect`.

    The name is a persistent source of confusion. Granger causality is
    predictive precedence, not causation. If x Granger-causes y, past x
    contains information about future y beyond y's own past. That is
    consistent with x causing y, and equally consistent with both
    responding to an omitted third variable, or with agents anticipating y
    and acting on x first.

    Implemented as an F-test comparing the unrestricted equation against
    one estimated without the cause's lags.
    """
    cause_index = var.index_of(cause)
    effect_index = var.index_of(effect)
    if cause_index == effect_index:
        raise EngineError("a variable cannot Granger-cause itself")

    data = np.ascontiguousarray(data, dtype=float)
    if data.shape[1] != var.n_variables:
        raise EngineError(f"data has {data.shape[1]} columns, expected {var.n_variables}")

    p = var.n_lags
    unrestricted = var.residuals[:, effect_index]
    rss_unrestricted = float(unrestricted @ unrestricted)

    kept = [i for i in range(var.n_variables) if i != cause_index]
    restricted_data = data[:, kept]
    restricted_names = tuple(var.variables[i] for i in kept)

    restricted_var = estimate_var(
        restricted_data, restricted_names, n_lags=p, require_stable=False
    )
    restricted_index = restricted_names.index(effect)
    restricted = restricted_var.residuals[:, restricted_index]
    rss_restricted = float(restricted @ restricted)

    numerator_df = p
    denominator_df = var.n_observations - var.n_parameters
    if denominator_df <= 0:
        raise EngineError("no residual degrees of freedom for the F-test")

    statistic = ((rss_restricted - rss_unrestricted) / numerator_df) / (
        rss_unrestricted / denominator_df
    )
    pvalue = float(stats.f.sf(statistic, numerator_df, denominator_df))

    return CheckOutcome(
        name=f"granger[{cause}->{effect}]",
        statistic=statistic,
        pvalue=pvalue,
        null_hypothesis=f"lags of {cause} do not improve prediction of {effect}",
        severity=Severity.ADVISORY,
        significance=significance,
        detail=(
            f"F({numerator_df}, {denominator_df}); predictive precedence only, "
            f"not evidence of causation"
        ),
    )


# ---- parameter stability --------------------------------------------------


def cusum_test(
    var: VARResult, *, significance: float = DEFAULT_SIGNIFICANCE
) -> tuple[CheckOutcome, ...]:
    """CUSUM test for coefficient stability across the sample.

    Cumulates standardised residuals and compares the path against a
    confidence corridor. A breach suggests the relationship changed part
    way through, which for macro data usually means a regime change: a new
    policy framework, a currency reform, a crisis.

    Critical, because a VAR estimated across a structural break describes
    an average of two regimes and neither of them.
    """
    residuals = var.residuals
    t = residuals.shape[0]
    outcomes = []

    # Corridor width for the standard 5% CUSUM test.
    critical_value = 0.948

    for index, name in enumerate(var.variables):
        series = residuals[:, index]
        deviation = float(series.std(ddof=1))
        if deviation == 0:
            raise EngineError(f"residuals of {name} have zero variance")

        cumulative = np.cumsum(series) / (deviation * np.sqrt(t))
        boundary = critical_value * (1 + 2 * np.arange(1, t + 1) / t)
        breaches = int(np.sum(np.abs(cumulative) > boundary))
        max_excess = float(np.max(np.abs(cumulative) / boundary))

        # Convert the corridor breach into an approximate p-value so the
        # outcome is comparable with the other tests.
        pvalue = 1.0 if max_excess <= 1.0 else max(1e-6, float(np.exp(-2 * (max_excess - 1))))

        outcomes.append(
            CheckOutcome(
                name=f"cusum[{name}]",
                statistic=max_excess,
                pvalue=pvalue,
                null_hypothesis=f"coefficients of the {name} equation are stable",
                severity=Severity.CRITICAL,
                significance=significance,
                detail=f"{breaches} of {t} points outside the corridor",
            )
        )
    return tuple(outcomes)


# ---- full report ----------------------------------------------------------


def diagnose(
    var: VARResult,
    *,
    portmanteau_lags: int = 12,
    arch_lags: int = 5,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> DiagnosticReport:
    """Run every specification check and collect the outcomes.

    Granger causality is excluded here because it needs the original data
    matrix and answers a question about the economics rather than about
    the specification. Call it directly when relevant.
    """
    outcomes: list[CheckOutcome] = [
        portmanteau_test(var, portmanteau_lags, significance=significance)
    ]
    outcomes.extend(ljung_box_per_equation(var, portmanteau_lags, significance=significance))
    outcomes.extend(arch_lm_test(var, arch_lags, significance=significance))
    outcomes.extend(jarque_bera_test(var, significance=significance))
    outcomes.extend(cusum_test(var, significance=significance))

    report = DiagnosticReport(
        variables=var.variables,
        n_lags=var.n_lags,
        n_observations=var.n_observations,
        outcomes=tuple(outcomes),
    )

    log.info(
        "diagnostics_completed",
        variables=list(var.variables),
        n_lags=var.n_lags,
        usable=report.is_usable,
        critical=len(report.critical_failures),
        advisory=len(report.advisory_failures),
    )

    if report.critical_failures:
        log.warning(
            "specification_failure",
            failures=[o.name for o in report.critical_failures],
            note="inference from this model is not reliable",
        )
    return report