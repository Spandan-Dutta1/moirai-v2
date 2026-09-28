"""
Structural identification of a reduced-form VAR.

A VAR gives residuals u_t with covariance Sigma. What is wanted are
structural shocks eps_t that are economically meaningful and mutually
independent:

    u_t = B eps_t        with     Sigma = B B'

Sigma is symmetric, so it carries k(k+1)/2 distinct numbers. B has k^2
unknowns. For three variables that is six equations and nine unknowns.
The system is underdetermined, and no quantity of additional data closes
the gap: the missing k(k-1)/2 restrictions must come from outside the
data entirely.

That is what identification means. Every scheme here is a different
answer to the question "what am I willing to assume?", and in every case
the assumption is untestable. The honest response is not to hide it but
to state it, and then to show how much the conclusion depends on it,
which is what ordering_sensitivity does.
"""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import permutations
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.causal.var import VARResult

log = get_logger(__name__)

#: Largest system for which every ordering can be enumerated. 8! is 40320.
MAX_VARIABLES_FOR_FULL_SENSITIVITY = 8


class IdentificationScheme(StrEnum):
    """How structural shocks were recovered from reduced-form residuals."""

    #: Recursive ordering. Point-identified, ordering-dependent.
    CHOLESKY = "cholesky"
    #: Directional restrictions on impact responses. Set-identified.
    SIGN_RESTRICTIONS = "sign_restrictions"
    #: Caller supplied B directly. Provenance depends on the caller.
    EXTERNAL = "external"


class Sign(StrEnum):
    """Direction a response must take on impact."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNRESTRICTED = "unrestricted"


class StructuralModel(BaseModel):
    """A VAR plus one identification of its shocks.

    `impact` is B: column j holds the contemporaneous response of every
    variable to a one standard deviation shock j. `shocks` are the implied
    structural disturbances, which should be close to orthogonal.

    The scheme and its assumptions are carried alongside deliberately. An
    impulse response without its identifying assumption is not a result,
    it is a number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    var: VARResult
    scheme: IdentificationScheme
    impact: Any = Field(description="(k, k) matrix B such that Sigma = B B'.")
    shocks: Any = Field(description="(T, k) structural shocks.")
    shock_names: tuple[str, ...]
    assumptions: str = Field(description="The untestable claim this rests on.")
    ordering: tuple[str, ...] | None = None

    @property
    def n_variables(self) -> int:
        return self.var.n_variables

    def reconstruction_error(self) -> float:
        """Largest absolute deviation in B B' - Sigma.

        Any identification must reproduce the observed covariance exactly.
        A non-trivial error means the decomposition is wrong, not merely
        differently assumed.
        """
        return float(np.abs(self.impact @ self.impact.T - self.var.sigma_u).max())

    def shock_correlation(self) -> np.ndarray:
        """Correlation matrix of the recovered shocks, ideally the identity."""
        scale = self.shocks.std(axis=0, ddof=1)
        if np.any(scale == 0):
            raise EngineError("a structural shock has zero variance")
        standardised = self.shocks / scale
        return np.corrcoef(standardised, rowvar=False)

    def max_off_diagonal_correlation(self) -> float:
        correlation = self.shock_correlation()
        off_diagonal = correlation - np.eye(self.n_variables)
        return float(np.abs(off_diagonal).max())

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme.value,
            "assumptions": self.assumptions,
            "ordering": list(self.ordering) if self.ordering else None,
            "shock_names": list(self.shock_names),
            "reconstruction_error": round(self.reconstruction_error(), 12),
            "max_shock_correlation": round(self.max_off_diagonal_correlation(), 6),
            "var": self.var.to_ledger_dict(),
        }


# ---- Cholesky -------------------------------------------------------------


def _validate_ordering(var: VARResult, ordering: tuple[str, ...] | None) -> list[int]:
    """Turn a variable ordering into column indices."""
    if ordering is None:
        return list(range(var.n_variables))

    if len(ordering) != var.n_variables:
        raise EngineError(
            f"ordering has {len(ordering)} names for {var.n_variables} variables"
        )
    if set(ordering) != set(var.variables):
        missing = set(var.variables) - set(ordering)
        extra = set(ordering) - set(var.variables)
        raise EngineError(f"ordering mismatch; missing {missing}, unexpected {extra}")

    return [var.index_of(name) for name in ordering]


def identify_cholesky(
    var: VARResult,
    ordering: tuple[str, ...] | None = None,
    *,
    shock_names: tuple[str, ...] | None = None,
) -> StructuralModel:
    """Recursive identification by Cholesky decomposition.

    The assumption is a contemporaneous causal ordering: the first variable
    responds to nothing else within the period, the second responds only to
    the first, and so on. That is exactly k(k-1)/2 restrictions, which is
    precisely the number the system is short.

    In a monetary VAR ordered (output, prices, policy rate), the claim is
    that the central bank observes output and prices when setting the rate,
    while output and prices respond to the rate only with a lag. Defensible
    for monthly data. Much less so if a financial variable is involved,
    since asset prices move within minutes.

    Results depend on the ordering. Use ordering_sensitivity to find out
    how much.
    """
    indices = _validate_ordering(var, ordering)
    resolved_ordering = tuple(var.variables[i] for i in indices)

    permuted = var.sigma_u[np.ix_(indices, indices)]

    try:
        lower = np.linalg.cholesky(permuted)
    except np.linalg.LinAlgError as err:
        raise EngineError(
            "residual covariance is not positive definite, so no Cholesky "
            "factor exists. This usually means collinear variables or more "
            "parameters than the sample can support."
        ) from err

    # Undo the permutation so B is expressed in the VAR's own variable order.
    inverse = np.argsort(indices)
    impact = lower[np.ix_(inverse, inverse)]

    # eps_t = B^-1 u_t, solved rather than inverted for numerical stability.
    shocks = np.linalg.solve(impact, var.residuals.T).T

    names = shock_names or tuple(f"{name}_shock" for name in var.variables)
    if len(names) != var.n_variables:
        raise EngineError(f"{len(names)} shock names for {var.n_variables} variables")

    model = StructuralModel(
        var=var,
        scheme=IdentificationScheme.CHOLESKY,
        impact=impact,
        shocks=shocks,
        shock_names=names,
        ordering=resolved_ordering,
        assumptions=(
            "Contemporaneous causal ordering "
            + " -> ".join(resolved_ordering)
            + ". Each variable responds within the period only to those before it. "
            "Untestable from the data."
        ),
    )

    log.info(
        "identified_cholesky",
        ordering=list(resolved_ordering),
        reconstruction_error=round(model.reconstruction_error(), 12),
    )
    return model


# ---- ordering sensitivity -------------------------------------------------


class OrderingSensitivity(BaseModel):
    """How much an impact response moves across Cholesky orderings.

    Most published VARs report a single ordering. That is defensible only
    if the conclusion is insensitive to it, and the way to know is to look.
    A result that survives every ordering is robust. A result that flips
    sign means the ordering is the finding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_of: str
    response_to: str
    n_orderings: int
    minimum: float
    maximum: float
    median: float
    sign_flips: bool
    values_by_ordering: tuple[tuple[tuple[str, ...], float], ...]

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum

    @property
    def relative_spread(self) -> float:
        """Spread as a multiple of the median magnitude.

        Scale-free, so comparable across variables measured in different
        units. Above roughly 0.5 the ordering is doing serious work.
        """
        scale = abs(self.median)
        return float("inf") if scale == 0 else self.spread / scale

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "response_of": self.response_of,
            "response_to": self.response_to,
            "n_orderings": self.n_orderings,
            "minimum": round(self.minimum, 8),
            "maximum": round(self.maximum, 8),
            "median": round(self.median, 8),
            "spread": round(self.spread, 8),
            "sign_flips": self.sign_flips,
        }


def ordering_sensitivity(
    var: VARResult, response_of: str, response_to: str
) -> OrderingSensitivity:
    """Impact response of one variable to another, across every ordering.

    Enumerates all k! Cholesky orderings and records the impact response
    under each. This is the honest companion to reporting a single
    ordering: it converts an unstated dependence into a measured one.
    """
    if var.n_variables > MAX_VARIABLES_FOR_FULL_SENSITIVITY:
        raise EngineError(
            f"{var.n_variables} variables means "
            f"{math.factorial(var.n_variables)} orderings; enumeration is "
            f"limited to {MAX_VARIABLES_FOR_FULL_SENSITIVITY}"
        )

    row = var.index_of(response_of)
    column = var.index_of(response_to)

    collected: list[tuple[tuple[str, ...], float]] = []
    for candidate in permutations(var.variables):
        model = identify_cholesky(var, candidate)
        collected.append((candidate, float(model.impact[row, column])))

    values = np.array([value for _, value in collected])
    signs = np.sign(values[np.abs(values) > 1e-12])

    result = OrderingSensitivity(
        response_of=response_of,
        response_to=response_to,
        n_orderings=len(collected),
        minimum=float(values.min()),
        maximum=float(values.max()),
        median=float(np.median(values)),
        sign_flips=bool(signs.size > 0 and not np.all(signs == signs[0])),
        values_by_ordering=tuple(collected),
    )

    log.info(
        "ordering_sensitivity_computed",
        response_of=response_of,
        response_to=response_to,
        n_orderings=result.n_orderings,
        spread=round(result.spread, 8),
        sign_flips=result.sign_flips,
    )
    return result


# ---- sign restrictions ----------------------------------------------------


class SignRestriction(BaseModel):
    """A required direction for one variable's response to one shock.

    By default the restriction applies on impact only. `horizons` extends
    it to later periods, so "does not raise output for three months" is
    horizons (0, 1, 2). Responses beyond impact depend on the VAR's
    dynamics as well as on the rotation, so a horizon restriction says
    more than an impact one and generally accepts fewer draws.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variable: str
    shock: str
    sign: Sign
    horizons: tuple[int, ...] = (0,)

    @field_validator("horizons")
    @classmethod
    def _horizons_are_distinct_and_ordered(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("a restriction needs at least one horizon")
        if any(h < 0 for h in value):
            raise ValueError("horizons must not be negative")
        if list(value) != sorted(set(value)):
            raise ValueError("horizons must be strictly increasing")
        return value

    def is_satisfied(self, value: float, tolerance: float = 0.0) -> bool:
        if self.sign is Sign.UNRESTRICTED:
            return True
        if self.sign is Sign.POSITIVE:
            return value > tolerance
        return value < -tolerance


def _random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random orthogonal matrix from the Haar measure.

    QR of a Gaussian matrix gives a uniform orthogonal draw once the signs
    of R's diagonal are normalised. Skipping that normalisation biases the
    draw, which quietly biases the identified set.
    """
    gaussian = rng.standard_normal((k, k))
    q, r = np.linalg.qr(gaussian)
    return q * np.sign(np.diag(r))


class SignIdentifiedSet(BaseModel):
    """The set of impact matrices consistent with the sign restrictions.

    Sign restrictions are set-identifying, not point-identifying: many
    matrices satisfy the same directional claims. Reporting one draw as
    though it were the answer misrepresents what was learned. The set is
    the result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    var: VARResult
    restrictions: tuple[SignRestriction, ...]
    accepted: Any = Field(description="(n_accepted, k, k) stack of admissible B matrices.")
    n_draws: int
    shock_names: tuple[str, ...]

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.shape[0])

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_draws if self.n_draws else 0.0

    def response_paths(self, response_of: str, response_to: str, horizon: int) -> np.ndarray:
        """Every accepted draw's response path, shape (n_accepted, horizon + 1).

        One standard deviation shocks, as in the impulse responses. The set
        is conditional on the VAR's point estimates: the rotations vary,
        the reduced form does not.
        """
        row = self.var.index_of(response_of)
        column = self.shock_names.index(response_to)
        ma = self.var.ma_coefficients(horizon)
        paths: np.ndarray = np.einsum("hj,nj->nh", ma[:, row, :], self.accepted[:, :, column])
        return paths

    def impact_quantiles(
        self, response_of: str, response_to: str, quantiles: tuple[float, ...] = (0.16, 0.5, 0.84)
    ) -> tuple[float, ...]:
        """Quantiles of one impact response across the identified set."""
        row = self.var.index_of(response_of)
        column = self.shock_names.index(response_to)
        values = self.accepted[:, row, column]
        return tuple(float(v) for v in np.quantile(values, quantiles))

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "scheme": IdentificationScheme.SIGN_RESTRICTIONS.value,
            "n_draws": self.n_draws,
            "n_accepted": self.n_accepted,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "restrictions": [
                {
                    "variable": r.variable,
                    "shock": r.shock,
                    "sign": r.sign.value,
                    "horizons": list(r.horizons),
                }
                for r in self.restrictions
            ],
            "var": self.var.to_ledger_dict(),
        }


def identify_sign_restrictions(
    var: VARResult,
    restrictions: tuple[SignRestriction, ...],
    shock_names: tuple[str, ...],
    *,
    n_draws: int = 2_000,
    seed: int = 42,
    tolerance: float = 0.0,
) -> SignIdentifiedSet:
    """Identify shocks by the direction of their responses.

    Every candidate B = P Q, where P is a Cholesky factor and Q is a random
    orthogonal matrix, reproduces Sigma exactly. The restrictions then keep
    only those candidates whose responses point the required way at every
    restricted horizon. The response at horizon h is Psi_h B, so a
    restriction beyond impact also depends on the VAR's dynamics.

    The assumption is weaker than a recursive ordering, and often easier to
    defend: a contractionary monetary shock raises the policy rate and
    lowers output, whatever the timing of information flows. The price is
    that the answer is a set rather than a point.

    `seed` is fixed so the draw is reproducible. An identified set that
    changes between runs is not a result.
    """
    if not restrictions:
        raise EngineError("sign identification needs at least one restriction")
    if len(shock_names) != var.n_variables:
        raise EngineError(f"{len(shock_names)} shock names for {var.n_variables} variables")
    if len(set(shock_names)) != len(shock_names):
        raise EngineError("shock names must be unique")

    for restriction in restrictions:
        var.index_of(restriction.variable)  # raises if unknown
        if restriction.shock not in shock_names:
            raise EngineError(
                f"restriction refers to unknown shock {restriction.shock!r}; "
                f"have {list(shock_names)}"
            )

    try:
        base = np.linalg.cholesky(var.sigma_u)
    except np.linalg.LinAlgError as err:
        raise EngineError("residual covariance is not positive definite") from err

    rng = np.random.default_rng(seed)
    ma = var.ma_coefficients(max(h for r in restrictions for h in r.horizons))
    indexed = [
        (var.index_of(r.variable), shock_names.index(r.shock), r) for r in restrictions
    ]

    accepted: list[np.ndarray] = []
    for _ in range(n_draws):
        candidate = base @ _random_orthogonal(var.n_variables, rng)
        if all(
            restriction.is_satisfied(float(ma[h, row, :] @ candidate[:, column]), tolerance)
            for row, column, restriction in indexed
            for h in restriction.horizons
        ):
            accepted.append(candidate)

    if not accepted:
        raise EngineError(
            f"no draw satisfied the restrictions in {n_draws} attempts. The "
            f"restrictions may be mutually inconsistent, or too tight for this "
            f"covariance."
        )

    result = SignIdentifiedSet(
        var=var,
        restrictions=restrictions,
        accepted=np.stack(accepted),
        n_draws=n_draws,
        shock_names=shock_names,
    )

    log.info(
        "identified_sign_restrictions",
        n_draws=n_draws,
        n_accepted=result.n_accepted,
        acceptance_rate=round(result.acceptance_rate, 4),
        n_restrictions=len(restrictions),
    )

    if result.acceptance_rate < 0.01:
        log.warning(
            "low_acceptance_rate",
            acceptance_rate=round(result.acceptance_rate, 5),
            note="the identified set may be poorly explored; consider more draws",
        )
    return result


# ---- external -------------------------------------------------------------


def identify_external(
    var: VARResult, impact: np.ndarray, shock_names: tuple[str, ...], assumptions: str
) -> StructuralModel:
    """Accept an impact matrix computed elsewhere.

    Used for schemes not implemented here, such as long-run or narrative
    identification. The reconstruction error is checked, because a B that
    does not reproduce Sigma is not an identification of this VAR.
    """
    impact = np.ascontiguousarray(impact, dtype=float)
    k = var.n_variables

    if impact.shape != (k, k):
        raise EngineError(f"impact matrix is {impact.shape}, expected ({k}, {k})")
    if len(shock_names) != k:
        raise EngineError(f"{len(shock_names)} shock names for {k} variables")

    error = float(np.abs(impact @ impact.T - var.sigma_u).max())
    if error > 1e-6:
        raise EngineError(
            f"impact matrix does not reproduce the residual covariance "
            f"(max deviation {error:.2e}); B B' must equal Sigma"
        )

    shocks = np.linalg.solve(impact, var.residuals.T).T
    return StructuralModel(
        var=var,
        scheme=IdentificationScheme.EXTERNAL,
        impact=impact,
        shocks=shocks,
        shock_names=shock_names,
        assumptions=assumptions,
    )