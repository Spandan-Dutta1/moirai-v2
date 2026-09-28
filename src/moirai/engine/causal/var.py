"""
Reduced-form Vector Autoregression.

A VAR lets every variable depend on lags of itself and of every other
variable in the system:

    y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t

Sims (1980) proposed this as a critique of large structural macro models,
which imposed hundreds of restrictions nobody could defend. A VAR imposes
almost none, then asks what minimal, explicit assumption is needed to read
causality into the result.

That last step is not here. The residuals u_t are correlated across
equations, so no single one can be shocked in isolation. Turning u_t into
interpretable structural shocks requires an identifying assumption the
data cannot test, and that lives in identification.py.

Estimation is statistics. Identification is economics. This file is the
statistics half, and it makes no causal claim.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger

log = get_logger(__name__)

#: Refuse to estimate when observations per parameter falls below this.
MIN_OBSERVATIONS_PER_PARAMETER = 5.0

#: Eigenvalue modulus above which the system is treated as explosive.
STABILITY_TOLERANCE = 1.0


class InformationCriterion(StrEnum):
    """Criteria for choosing lag order.

    AIC is efficient for forecasting but not consistent: it tends to
    over-select as the sample grows. BIC and HQIC are consistent and
    choose shorter lags, which is usually preferred for structural work
    because each extra lag costs k^2 parameters.
    """

    AIC = "aic"
    BIC = "bic"
    HQIC = "hqic"


class LagSelection(BaseModel):
    """Information criteria across candidate lag orders.

    All three criteria are reported rather than one being silently chosen.
    When they disagree, that disagreement is a real fact about the sample
    and the analyst should see it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[int, ...]
    aic: tuple[float, ...]
    bic: tuple[float, ...]
    hqic: tuple[float, ...]

    def best(self, criterion: InformationCriterion = InformationCriterion.BIC) -> int:
        values = getattr(self, criterion.value)
        return self.candidates[int(np.argmin(values))]

    @property
    def criteria_agree(self) -> bool:
        chosen = {self.best(criterion) for criterion in InformationCriterion}
        return len(chosen) == 1

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "aic": [round(v, 6) for v in self.aic],
            "bic": [round(v, 6) for v in self.bic],
            "hqic": [round(v, 6) for v in self.hqic],
            "best_aic": self.best(InformationCriterion.AIC),
            "best_bic": self.best(InformationCriterion.BIC),
            "best_hqic": self.best(InformationCriterion.HQIC),
            "criteria_agree": self.criteria_agree,
        }


class StabilityResult(BaseModel):
    """Eigenvalues of the companion matrix.

    A VAR is stable when every eigenvalue lies strictly inside the unit
    circle. If any modulus reaches one, shocks never die out: impulse
    responses diverge instead of decaying, and every downstream number is
    meaningless. This is checked rather than assumed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    moduli: tuple[float, ...]
    tolerance: float = STABILITY_TOLERANCE

    @property
    def max_modulus(self) -> float:
        return max(self.moduli) if self.moduli else 0.0

    @property
    def is_stable(self) -> bool:
        return self.max_modulus < self.tolerance

    @property
    def half_life(self) -> float | None:
        """Periods for the most persistent component to decay by half.

        This is a property of the system, not of any particular shock: it
        is log(0.5) / log(max modulus) for the slowest eigenmode of the
        companion matrix, and bounds how fast every impulse response
        eventually dies out. It is not the half-life of the policy rate's
        response to a policy shock, which can be much longer. In the
        1985-2007 US specification this is 12.7 months while the cumulative
        rate response has not halved from its peak by month 60.

        None when the system is not stable, since nothing decays.
        """
        if not self.is_stable or self.max_modulus <= 0:
            return None
        return float(np.log(0.5) / np.log(self.max_modulus))

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "max_modulus": round(self.max_modulus, 6),
            "is_stable": self.is_stable,
            "half_life": None if self.half_life is None else round(self.half_life, 3),
            "n_eigenvalues": len(self.moduli),
        }


class VARResult(BaseModel):
    """An estimated reduced-form VAR.

    Coefficients are stored as (p, k, k): coefficients[l][i][j] is the
    effect of variable j at lag l+1 on variable i today.

    sigma_u is the residual covariance. Its off-diagonal terms are exactly
    why this model cannot yet answer causal questions: the shocks move
    together, so no one of them can be varied alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    variables: tuple[str, ...]
    n_lags: int = Field(ge=1)
    n_observations: int = Field(gt=0)
    periods: tuple[date, ...] = ()

    intercept: Any = Field(description="(k,) array of constants.")
    coefficients: Any = Field(description="(p, k, k) array of lag coefficients.")
    residuals: Any = Field(description="(T-p, k) array of reduced-form residuals.")
    sigma_u: Any = Field(description="(k, k) residual covariance.")

    stability: StabilityResult
    lag_selection: LagSelection | None = None

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_parameters(self) -> int:
        """Per equation: one constant plus k coefficients per lag."""
        return 1 + self.n_variables * self.n_lags

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_observations - self.n_parameters

    def index_of(self, variable: str) -> int:
        try:
            return self.variables.index(variable)
        except ValueError:
            raise EngineError(
                f"variable {variable!r} is not in this VAR; have {list(self.variables)}"
            ) from None

    def companion_matrix(self) -> np.ndarray:
        """Stack the VAR into its first-order companion form."""
        return _companion(self.coefficients)

    def ma_coefficients(self, horizon: int) -> np.ndarray:
        """Moving-average coefficients Psi_h for h = 0..horizon.

        Computed by iterating the companion matrix rather than forming A^h
        directly: repeated multiplication of a k*p square matrix is cheaper
        and numerically better behaved than exponentiation for the horizons
        used in practice. Lives here rather than with the impulse responses
        because identification needs it too, when a sign restriction
        applies beyond the impact period.
        """
        if horizon < 0:
            raise EngineError("horizon must not be negative")
        k, p = self.n_variables, self.n_lags
        companion = self.companion_matrix()

        ma = np.zeros((horizon + 1, k, k), dtype=float)
        ma[0] = np.eye(k)

        power = np.eye(k * p)
        for h in range(1, horizon + 1):
            power = companion @ power
            ma[h] = power[:k, :k]
        return ma

    def log_likelihood(self) -> float:
        """Gaussian log likelihood at the estimated parameters."""
        k, t = self.n_variables, self.n_observations
        sign, log_det = np.linalg.slogdet(self.sigma_u)
        if sign <= 0:
            raise EngineError("residual covariance is not positive definite")
        return float(-0.5 * t * (k * np.log(2 * np.pi) + log_det + k))

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "n_lags": self.n_lags,
            "n_observations": self.n_observations,
            "n_parameters_per_equation": self.n_parameters,
            "degrees_of_freedom": self.degrees_of_freedom,
            "sample_start": str(self.periods[0]) if self.periods else None,
            "sample_end": str(self.periods[-1]) if self.periods else None,
            "stability": self.stability.to_ledger_dict(),
            "lag_selection": (
                self.lag_selection.to_ledger_dict() if self.lag_selection else None
            ),
        }


# ---- internals ------------------------------------------------------------


def _build_design(data: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the regressor matrix X and target matrix Y.

    Row t of X is [1, y_{t-1}, ..., y_{t-p}] flattened. The first p
    observations are consumed as initial conditions.
    """
    t_total, k = data.shape
    effective = t_total - n_lags

    y = data[n_lags:]
    x = np.empty((effective, 1 + k * n_lags), dtype=float)
    x[:, 0] = 1.0
    for lag in range(1, n_lags + 1):
        start = 1 + (lag - 1) * k
        x[:, start : start + k] = data[n_lags - lag : t_total - lag]
    return x, y


def _companion(coefficients: np.ndarray) -> np.ndarray:
    """Companion matrix: rewrites a VAR(p) as a VAR(1) in stacked form.

    The top block row holds the lag coefficients; the subdiagonal identity
    blocks shift the state. Its eigenvalues govern the system's dynamics.
    """
    p, k, _ = coefficients.shape
    companion = np.zeros((k * p, k * p), dtype=float)
    companion[:k] = np.hstack([coefficients[lag] for lag in range(p)])
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return companion


def check_stability(
    coefficients: np.ndarray, tolerance: float = STABILITY_TOLERANCE
) -> StabilityResult:
    """Eigenvalue moduli of the companion matrix, descending."""
    eigenvalues = np.linalg.eigvals(_companion(coefficients))
    moduli = tuple(sorted((float(abs(v)) for v in eigenvalues), reverse=True))
    return StabilityResult(moduli=moduli, tolerance=tolerance)


def _fit_ols(data: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a VAR by OLS. Returns (intercept, coefficients, residuals).

    Every equation has identical regressors, so equation-by-equation OLS
    is exactly the SUR/GLS estimator here. Solving all equations at once
    with lstsq is both simpler and numerically better conditioned than
    forming the normal equations.
    """
    x, y = _build_design(data, n_lags)
    k = data.shape[1]

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)  # (1 + k*p, k)
    residuals = y - x @ beta

    intercept = beta[0].copy()
    coefficients = np.empty((n_lags, k, k), dtype=float)
    for lag in range(n_lags):
        block = beta[1 + lag * k : 1 + (lag + 1) * k]  # (k, k)
        coefficients[lag] = block.T  # transpose: rows become equations
    return intercept, coefficients, residuals


def _sigma_u(residuals: np.ndarray, n_parameters: int, *, corrected: bool = True) -> np.ndarray:
    """Residual covariance.

    The degrees-of-freedom correction matters in short macro samples,
    where the difference between T and T - (1 + kp) can be substantial.
    """
    t = residuals.shape[0]
    denominator = (t - n_parameters) if corrected else t
    if denominator <= 0:
        raise EngineError("not enough observations to estimate the residual covariance")
    return (residuals.T @ residuals) / denominator


def select_lag_order(
    data: np.ndarray,
    max_lags: int = 12,
    *,
    variables: tuple[str, ...] | None = None,
) -> LagSelection:
    """Compute AIC, BIC and HQIC across lag orders 1..max_lags.

    Every candidate is estimated on the same sample, dropping max_lags
    initial observations throughout. Comparing criteria computed on
    different samples is a common and invalidating mistake.
    """
    t_total, k = data.shape
    if max_lags < 1:
        raise EngineError("max_lags must be at least 1")

    effective = t_total - max_lags
    if effective < k * max_lags + 1:
        raise EngineError(
            f"cannot evaluate up to {max_lags} lags: {effective} usable observations "
            f"for {k * max_lags + 1} parameters per equation"
        )

    candidates, aic, bic, hqic = [], [], [], []
    for p in range(1, max_lags + 1):
        trimmed = data[max_lags - p :]
        _, _, residuals = _fit_ols(trimmed, p)
        sigma = _sigma_u(residuals, 1 + k * p, corrected=False)

        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise EngineError(f"singular residual covariance at {p} lags")

        n_free = k * (1 + k * p)  # parameters across the whole system
        candidates.append(p)
        aic.append(log_det + 2 * n_free / effective)
        bic.append(log_det + np.log(effective) * n_free / effective)
        hqic.append(log_det + 2 * np.log(np.log(effective)) * n_free / effective)

    selection = LagSelection(
        candidates=tuple(candidates),
        aic=tuple(aic),
        bic=tuple(bic),
        hqic=tuple(hqic),
    )

    log.info(
        "lag_order_selected",
        variables=list(variables) if variables else None,
        best_aic=selection.best(InformationCriterion.AIC),
        best_bic=selection.best(InformationCriterion.BIC),
        best_hqic=selection.best(InformationCriterion.HQIC),
        agree=selection.criteria_agree,
    )
    return selection


# ---- public estimation ----------------------------------------------------


def estimate_var(
    data: np.ndarray,
    variables: tuple[str, ...],
    *,
    n_lags: int | None = None,
    max_lags: int = 12,
    criterion: InformationCriterion = InformationCriterion.BIC,
    periods: tuple[date, ...] = (),
    require_stable: bool = True,
) -> VARResult:
    """Estimate a reduced-form VAR by OLS.

    Parameters
    ----------
    data
        (T, k) matrix, one column per variable, rows in time order.
    n_lags
        Fixed lag order. When omitted, chosen by `criterion`.
    require_stable
        Raise if the estimated system is explosive. Leave this on unless
        instability is itself the object of study: impulse responses from
        an unstable VAR diverge and mean nothing.
    """
    data = np.ascontiguousarray(data, dtype=float)
    if data.ndim != 2:
        raise EngineError(f"data must be 2-dimensional, got shape {data.shape}")

    t_total, k = data.shape
    if k != len(variables):
        raise EngineError(f"{k} columns but {len(variables)} variable names")
    if len(set(variables)) != k:
        raise EngineError(f"variable names must be unique, got {list(variables)}")
    if not np.all(np.isfinite(data)):
        raise EngineError("data contains NaN or infinity; prepare the series first")
    if periods and len(periods) != t_total:
        raise EngineError(f"{len(periods)} periods for {t_total} rows")

    selection: LagSelection | None = None
    if n_lags is None:
        usable_max = min(max_lags, max(1, (t_total - 1) // (k + 1)))
        selection = select_lag_order(data, usable_max, variables=variables)
        n_lags = selection.best(criterion)
    elif n_lags < 1:
        raise EngineError("n_lags must be at least 1")

    effective = t_total - n_lags
    n_parameters = 1 + k * n_lags
    if effective <= n_parameters:
        raise EngineError(
            f"{effective} usable observations cannot identify {n_parameters} "
            f"parameters per equation at {n_lags} lags"
        )

    ratio = effective / n_parameters
    if ratio < MIN_OBSERVATIONS_PER_PARAMETER:
        log.warning(
            "var_overparameterised",
            observations_per_parameter=round(ratio, 2),
            n_lags=n_lags,
            n_variables=k,
        )

    intercept, coefficients, residuals = _fit_ols(data, n_lags)
    sigma = _sigma_u(residuals, n_parameters)
    stability = check_stability(coefficients)

    if require_stable and not stability.is_stable:
        raise EngineError(
            f"VAR is not stable: largest eigenvalue modulus is "
            f"{stability.max_modulus:.4f}. Impulse responses would diverge. "
            f"Check for a missed unit root or reduce the lag order."
        )

    result = VARResult(
        variables=tuple(variables),
        n_lags=n_lags,
        n_observations=effective,
        periods=periods[n_lags:] if periods else (),
        intercept=intercept,
        coefficients=coefficients,
        residuals=residuals,
        sigma_u=sigma,
        stability=stability,
        lag_selection=selection,
    )

    log.info(
        "var_estimated",
        variables=list(variables),
        n_lags=n_lags,
        observations=effective,
        max_modulus=round(stability.max_modulus, 4),
        stable=stability.is_stable,
    )
    return result


def forecast(result: VARResult, horizon: int, history: np.ndarray) -> np.ndarray:
    """Iterated point forecast for `horizon` periods.

    Parameters
    ----------
    history
        At least (p, k) rows of recent data, most recent last.

    Notes
    -----
    Point forecasts only. Forecast uncertainty needs the moving-average
    representation and is deferred to the impulse response module, which
    builds it anyway.
    """
    if horizon < 1:
        raise EngineError("horizon must be at least 1")

    history = np.ascontiguousarray(history, dtype=float)
    p, k = result.n_lags, result.n_variables
    if history.shape[1] != k:
        raise EngineError(f"history has {history.shape[1]} columns, expected {k}")
    if history.shape[0] < p:
        raise EngineError(f"history needs at least {p} rows, got {history.shape[0]}")

    window = list(history[-p:])
    predictions = np.empty((horizon, k), dtype=float)

    for step in range(horizon):
        value = result.intercept.copy()
        for lag in range(p):
            value = value + result.coefficients[lag] @ window[-(lag + 1)]
        predictions[step] = value
        window.append(value)

    return predictions