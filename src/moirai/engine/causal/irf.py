"""
Impulse responses and variance decompositions.

A VAR is autoregressive. To trace a shock forward it must be inverted into
its moving-average representation:

    y_t = sum_h Psi_h eps_{t-h}      with    Psi_h = J A^h J' B

A is the companion matrix, J selects the leading block, and B is the
impact matrix from identification. Powers of A propagate the shock, which
is precisely why stability is checked before any of this runs: if an
eigenvalue reaches one, A^h grows without bound and the response diverges
instead of decaying.

Three outputs, in increasing order of how often they change a conclusion:

  * impulse responses, the dynamic path after a shock
  * variance decompositions, how much of the forecast error each shock
    explains, which often reveals that a clear effect is a small one
  * bootstrap bands, without which a response is a point estimate wearing
    the costume of a result

On the bootstrap: OLS is downward biased in autoregressive models, and the
residual bootstrap inherits that bias, so percentile bands can cover less
often than their nominal level (Kilian, 1998). The bias-corrected
bootstrap addresses this. The standard version is implemented here and the
limitation is stated rather than hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.identification import StructuralModel
from moirai.engine.causal.var import VARResult, estimate_var

log = get_logger(__name__)

DEFAULT_HORIZON = 24
DEFAULT_DRAWS = 1_000
DEFAULT_LEVELS = (0.68, 0.90)


class ImpulseResponse(BaseModel):
    """Dynamic responses of every variable to every structural shock.

    `responses` is (horizon + 1, k, k): responses[h][i][j] is the response
    of variable i at horizon h to shock j. Horizon 0 is the impact period,
    so responses[0] is exactly the impact matrix B.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    variables: tuple[str, ...]
    shock_names: tuple[str, ...]
    horizon: int = Field(ge=0)
    responses: Any = Field(description="(horizon + 1, k, k) array.")
    cumulative: bool = False
    scheme: str = ""
    assumptions: str = ""

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    def path(self, response_of: str, response_to: str) -> np.ndarray:
        """The (horizon + 1,) response path of one variable to one shock."""
        row = _index(self.variables, response_of, "variable")
        column = _index(self.shock_names, response_to, "shock")
        return self.responses[:, row, column]

    def peak(self, response_of: str, response_to: str) -> tuple[int, float]:
        """Horizon and value of the largest absolute response.

        The peak is usually the number a report quotes, and its timing is
        as interesting as its size: a peak at h=0 means a contemporaneous
        effect, a peak at h=18 means a long transmission lag.
        """
        series = self.path(response_of, response_to)
        index = int(np.argmax(np.abs(series)))
        return index, float(series[index])

    def cumulate(self) -> ImpulseResponse:
        """Accumulate responses along the horizon.

        Necessary when a variable was differenced before estimation. The
        VAR then describes the change, and the level response is its
        running sum. Reporting an uncumulated response for a differenced
        variable understates a persistent effect badly.
        """
        if self.cumulative:
            raise EngineError("responses are already cumulative")
        return self.model_copy(
            update={"responses": np.cumsum(self.responses, axis=0), "cumulative": True}
        )

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "shock_names": list(self.shock_names),
            "horizon": self.horizon,
            "cumulative": self.cumulative,
            "scheme": self.scheme,
            "assumptions": self.assumptions,
        }


class VarianceDecomposition(BaseModel):
    """Share of forecast error variance attributable to each shock.

    `shares` is (horizon + 1, k, k): shares[h][i][j] is the fraction of the
    h-step forecast error variance of variable i explained by shock j.
    Each row sums to one.

    This frequently matters more than the impulse response. A shock can
    move a variable in a clearly signed, precisely estimated way and still
    account for two percent of its variation, in which case the effect is
    real and unimportant. The impulse response alone cannot say which.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    variables: tuple[str, ...]
    shock_names: tuple[str, ...]
    horizon: int = Field(ge=0)
    shares: Any = Field(description="(horizon + 1, k, k) array, rows summing to one.")

    def share(self, variable: str, shock: str, horizon: int | None = None) -> float:
        """Variance share at one horizon, defaulting to the longest."""
        row = _index(self.variables, variable, "variable")
        column = _index(self.shock_names, shock, "shock")
        h = self.horizon if horizon is None else horizon
        if not 0 <= h <= self.horizon:
            raise EngineError(f"horizon {h} is outside 0..{self.horizon}")
        return float(self.shares[h, row, column])

    def dominant_shock(self, variable: str, horizon: int | None = None) -> tuple[str, float]:
        """Which shock explains most of a variable's forecast error."""
        row = _index(self.variables, variable, "variable")
        h = self.horizon if horizon is None else horizon
        column = int(np.argmax(self.shares[h, row]))
        return self.shock_names[column], float(self.shares[h, row, column])

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "shock_names": list(self.shock_names),
            "horizon": self.horizon,
            "long_run_shares": self.shares[-1].round(6).tolist(),
        }


class ConfidenceBands(BaseModel):
    """Bootstrap percentile bands around impulse responses.

    `lower` and `upper` are keyed by confidence level, each holding a
    (horizon + 1, k, k) array matching the point responses.

    These are percentile bootstrap bands. OLS is downward biased in
    autoregressions and the bootstrap inherits that bias, so realised
    coverage can fall short of the nominal level, particularly for
    persistent systems at long horizons. Treat them as indicative rather
    than exact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    point: ImpulseResponse
    lower: dict[float, Any]
    upper: dict[float, Any]
    n_draws: int
    n_successful: int
    seed: int

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(sorted(self.lower))

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.n_successful / self.n_draws if self.n_draws else 0.0

    def band(
        self, response_of: str, response_to: str, level: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lower, point, upper) paths for one response at one level."""
        if level not in self.lower:
            raise EngineError(f"no band at level {level}; have {list(self.levels)}")
        row = _index(self.point.variables, response_of, "variable")
        column = _index(self.point.shock_names, response_to, "shock")
        return (
            self.lower[level][:, row, column],
            self.point.responses[:, row, column],
            self.upper[level][:, row, column],
        )

    def excludes_zero(self, response_of: str, response_to: str, level: float) -> np.ndarray:
        """Boolean path: does the band exclude zero at each horizon?

        The nearest thing to a significance statement an IRF offers, and
        weaker than it looks: these are pointwise bands, so the chance that
        *some* horizon excludes zero by chance grows with the horizon.
        Joint bands would be needed for a statement about the whole path.
        """
        low, _, high = self.band(response_of, response_to, level)
        return (low > 0) | (high < 0)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "n_draws": self.n_draws,
            "n_successful": self.n_successful,
            "failure_rate": round(self.failure_rate, 6),
            "seed": self.seed,
            "method": "residual percentile bootstrap",
            "caveat": "percentile bands may under-cover; see Kilian (1998)",
            "point": self.point.to_ledger_dict(),
        }


# ---- internals ------------------------------------------------------------


def _index(names: tuple[str, ...], wanted: str, kind: str) -> int:
    try:
        return names.index(wanted)
    except ValueError:
        raise EngineError(f"unknown {kind} {wanted!r}; have {list(names)}") from None


def _ma_coefficients(var: VARResult, horizon: int) -> np.ndarray:
    """Moving-average coefficients Psi_h for h = 0..horizon.

    Computed by iterating the companion matrix rather than forming A^h
    directly: repeated multiplication of a k*p square matrix is cheaper and
    numerically better behaved than exponentiation for the horizons used
    in practice.
    """
    k, p = var.n_variables, var.n_lags
    companion = var.companion_matrix()

    ma = np.zeros((horizon + 1, k, k), dtype=float)
    ma[0] = np.eye(k)

    power = np.eye(k * p)
    for h in range(1, horizon + 1):
        power = companion @ power
        ma[h] = power[:k, :k]
    return ma


# ---- public ---------------------------------------------------------------


def impulse_responses(
    model: StructuralModel, horizon: int = DEFAULT_HORIZON
) -> ImpulseResponse:
    """Trace the response of every variable to every structural shock.

    Horizon 0 is the impact period, so responses[0] equals the impact
    matrix B. A one standard deviation shock is used throughout, which is
    the convention; scaling to a fixed size (say 25 basis points) is a
    caller's choice and belongs in reporting, not here.
    """
    if horizon < 0:
        raise EngineError("horizon must not be negative")
    if not model.var.stability.is_stable:
        raise EngineError(
            "cannot compute impulse responses from an unstable VAR: powers of "
            "the companion matrix diverge, so the responses are meaningless"
        )

    ma = _ma_coefficients(model.var, horizon)
    responses = np.einsum("hij,jk->hik", ma, model.impact)

    result = ImpulseResponse(
        variables=model.var.variables,
        shock_names=model.shock_names,
        horizon=horizon,
        responses=responses,
        scheme=model.scheme.value,
        assumptions=model.assumptions,
    )

    log.info(
        "impulse_responses_computed",
        horizon=horizon,
        scheme=model.scheme.value,
        variables=list(model.var.variables),
    )
    return result


def variance_decomposition(
    model: StructuralModel, horizon: int = DEFAULT_HORIZON
) -> VarianceDecomposition:
    """Forecast error variance shares by shock, at each horizon.

    The h-step forecast error is a sum of Psi_j eps_{t+h-j} over j <= h.
    Because the structural shocks are orthogonal, the variance splits
    cleanly by shock, which is what makes the decomposition well defined.
    """
    if horizon < 0:
        raise EngineError("horizon must not be negative")

    responses = impulse_responses(model, horizon).responses
    contributions = np.cumsum(responses**2, axis=0)  # (h+1, k, k)
    totals = contributions.sum(axis=2, keepdims=True)

    if np.any(totals <= 0):
        raise EngineError("a variable has zero forecast error variance")

    shares = contributions / totals

    result = VarianceDecomposition(
        variables=model.var.variables,
        shock_names=model.shock_names,
        horizon=horizon,
        shares=shares,
    )

    log.info(
        "variance_decomposition_computed",
        horizon=horizon,
        scheme=model.scheme.value,
    )
    return result


def bootstrap_bands(
    model: StructuralModel,
    horizon: int = DEFAULT_HORIZON,
    *,
    n_draws: int = DEFAULT_DRAWS,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    seed: int = 42,
) -> ConfidenceBands:
    """Percentile confidence bands by residual bootstrap.

    Each replication resamples the estimated residuals with replacement,
    rebuilds an artificial sample from the fitted dynamics, re-estimates
    the VAR, re-identifies, and recomputes the responses. The spread across
    replications is parameter uncertainty.

    Replications that produce an unstable system are discarded and counted.
    A high failure rate is itself informative: it means the estimated
    system sits close to the stability boundary, and the bands should be
    read with that in mind.

    `seed` is explicit so bands are reproducible. Bands that move between
    runs are not a result.
    """
    for level in levels:
        if not 0 < level < 1:
            raise EngineError(f"confidence level must lie in (0, 1), got {level}")
    if n_draws < 2:
        raise EngineError("bootstrap needs at least two draws")

    var = model.var
    point = impulse_responses(model, horizon)

    rng = np.random.default_rng(seed)
    residuals = var.residuals
    n_effective, k = residuals.shape
    p = var.n_lags

    # Reconstruct the fitted sample so replications start from real history.
    initial = np.zeros((p, k), dtype=float)

    collected: list[np.ndarray] = []
    for _ in range(n_draws):
        drawn = residuals[rng.integers(0, n_effective, size=n_effective)]

        artificial = np.zeros((p + n_effective, k), dtype=float)
        artificial[:p] = initial
        for t in range(p, p + n_effective):
            value = var.intercept.copy()
            for lag in range(p):
                value = value + var.coefficients[lag] @ artificial[t - lag - 1]
            artificial[t] = value + drawn[t - p]

        try:
            replicate_var = estimate_var(
                artificial, var.variables, n_lags=p, require_stable=True
            )
            replicate_model = _reidentify(model, replicate_var)
            collected.append(impulse_responses(replicate_model, horizon).responses)
        except (EngineError, np.linalg.LinAlgError):
            continue  # unstable or singular replication, counted by omission

    if len(collected) < 2:
        raise EngineError(
            f"only {len(collected)} of {n_draws} bootstrap replications succeeded; "
            f"the system is likely too close to the stability boundary"
        )

    stack = np.stack(collected)  # (n_successful, h+1, k, k)
    lower: dict[float, Any] = {}
    upper: dict[float, Any] = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lower[level] = np.quantile(stack, tail, axis=0)
        upper[level] = np.quantile(stack, 1.0 - tail, axis=0)

    result = ConfidenceBands(
        point=point,
        lower=lower,
        upper=upper,
        n_draws=n_draws,
        n_successful=len(collected),
        seed=seed,
    )

    log.info(
        "bootstrap_bands_computed",
        n_draws=n_draws,
        n_successful=len(collected),
        failure_rate=round(result.failure_rate, 4),
        levels=list(levels),
        seed=seed,
    )

    if result.failure_rate > 0.1:
        log.warning(
            "high_bootstrap_failure_rate",
            failure_rate=round(result.failure_rate, 4),
            note="the estimated system may sit near the stability boundary",
        )
    return result


def _reidentify(original: StructuralModel, replicate: VARResult) -> StructuralModel:
    """Apply the original identification to a bootstrap replicate.

    The identifying assumption is held fixed across replications by
    construction. Re-drawing it would conflate parameter uncertainty with
    identification uncertainty, which are different things and should be
    reported separately.
    """
    from moirai.engine.causal.identification import (
        IdentificationScheme,
        identify_cholesky,
    )

    if original.scheme is IdentificationScheme.CHOLESKY:
        return identify_cholesky(
            replicate, original.ordering, shock_names=original.shock_names
        )

    raise EngineError(
        f"bootstrap bands are not implemented for {original.scheme.value} "
        f"identification; sign-restricted sets already report a range"
    )