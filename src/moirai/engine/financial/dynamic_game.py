"""A dynamic game between central banks: policy that works with lags.

The network game in network.py is static. A rate move lowers inflation and
output in the same period it is made, and each bank chooses one rate. Real
policy works with lags: a rate change reaches output over the following
quarters and inflation mostly through output after that, which is why
central banks set rules for how they will respond to the economy rather
than choosing one number once. This module solves that version.

Transmission, quarterly, in deviations. For bank i with rate move u_i from
its current rate:

    y_i[t+1] = a_y y_i[t] - s u_i[t] - sum_j d_ij u_j[t]
    pi_i[t+1] = a_pi pi_i[t] + k y_i[t] - sum_j e_ij (u_i[t] - u_j[t])

The structure is the backward-looking policy model of Rudebusch and
Svensson (1999): rates reach output after a quarter, and inflation through
output after that, with the exchange rate acting on inflation after a
quarter. The coefficients are anchored to the static model rather than
estimated separately: s, k and the spillover terms are set so that a
permanent rate move has, in the long run, the static game's own output,
own inflation, exchange and demand effects.

One long-run effect differs, and it is the static game that is
inconsistent: there, a foreign rate move lowers home output through demand
without lowering home inflation. Here inflation responds to the output
gap whatever caused it, so the demand spillover also reaches inflation,
by k / (1 - a_pi) = 0.67 of its output effect. With the defaults a_y = 0.9 and a_pi = 0.8 this
gives k = 0.133 and s = 0.12 per quarter, close to Rudebusch and
Svensson's own estimates of 0.14 and 0.10 for the United States. The
persistence parameters are assumed; ADR 023 records their provenance and
the result's sensitivity to them.

Each bank minimises a discounted sum of the loss it carries in the static
game:

    w_pi pi_i^2 + w_y y_i^2 + w_s u_i^2 + w_e (u_i - mean_j u_j)^2

Smoothing penalises distance from the current rate, as in the static
game. The external term uses the gap in moves (the static game uses
levels; in the static game the difference is a constant that cancels from
every caused move, ADR 011).

The solution concept is feedback Nash: each bank chooses a linear rule
u_i = -F_i x, where x stacks every economy's inflation and output gaps,
taking the other banks' rules as given. It is found by iterating each
bank's discounted Riccati equation against the others' current rules until
no rule changes (the method of QuantEcon's nnash, extended to n players).
The external term depends on the others' moves, which through their rules
are functions of the state, so it enters each bank's problem exactly as a
state cost and a cross term.

This module adds a solver. It changes nothing the static game, the
scenarios or the pipeline compute.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.central_banks import CentralBank
from moirai.engine.financial.network import SpilloverMatrix

log = get_logger(__name__)

#: Quarterly persistence of the output gap. Rudebusch and Svensson (1999)
#: estimate two lags summing to 0.91 for the United States.
OUTPUT_PERSISTENCE = 0.90

#: Quarterly persistence of inflation. Rudebusch and Svensson impose an
#: accelerationist sum of one; 0.8 keeps inflation stationary under any
#: policy, which a game needs for its losses to be finite, and makes a
#: permanent rate change's effect settle in about two years.
INFLATION_PERSISTENCE = 0.80

#: Quarterly discount factor.
DISCOUNT = 0.99

#: Policy inertia calibrated so the Fed's equilibrium rule puts 0.79 on its
#: own previous rate, the smoothing coefficient Clarida, Gali and Gertler
#: (2000) estimate for the Volcker-Greenspan Fed at quarterly frequency.
#: Calibrated on the Fed, a published estimate, and checked against the RBI,
#: whose measured response to Fed surprises (ADR 022) was not used to choose
#: it. ADR 026.
CALIBRATED_INERTIA_WEIGHT = 6.97

#: The smoothing coefficient the calibration targets.
FED_SMOOTHING_TARGET = 0.79

#: The RBI's measured response per point of expected US policy, the floor
#: of the range ADR 027 measures: repo rate response over two-year Treasury
#: yield response to the same Fed surprises, at the horizons where both are
#: significant.
RBI_FOLLOWING_FLOOR = 0.47

#: The RBI's external weight at which, with its external objective weighted
#: by invoicing currency and the calibrated inertia, its peak caused move is
#: RBI_FOLLOWING_FLOOR times the Fed's. The validated RBI default, 0.40, is
#: unchanged; this is used only by the calibrated dynamic game. ADR 027.
CALIBRATED_RBI_EXTERNAL_WEIGHT = 1.92


class DynamicParameters(BaseModel):
    """Timing of transmission. The long-run effects come from the matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_persistence: float = Field(default=OUTPUT_PERSISTENCE, ge=0.0, lt=1.0)
    inflation_persistence: float = Field(default=INFLATION_PERSISTENCE, ge=0.0, lt=1.0)
    discount: float = Field(default=DISCOUNT, gt=0.0, lt=1.0)
    inertia_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the squared change in a bank's rate from the previous "
            "quarter: policy inertia, as in estimated inertial Taylor rules. "
            "Zero reproduces ADR 023 exactly. ADR 026 calibrates it to the "
            "RBI's measured response to Fed surprises."
        ),
    )
    tolerance: float = Field(default=1e-10, gt=0)
    max_iterations: int = Field(default=5_000, gt=0)


class DynamicEquilibrium(BaseModel):
    """Feedback rules and the rate paths they produce from a starting state."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    names: tuple[str, ...]
    rules: Any = Field(description="(n, 2n) array: row i is bank i's F_i, u_i = -F_i x.")
    current_rates: tuple[float, ...]
    moves: Any = Field(description="(horizon, n) array of rate moves from the current rate.")
    inflation_gaps: Any = Field(description="(horizon + 1, n) array.")
    output_gaps: Any = Field(description="(horizon + 1, n) array.")
    iterations: int
    parameters: DynamicParameters

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise EngineError(f"no bank named {name!r}; have {list(self.names)}") from None

    def rate_path(self, name: str) -> np.ndarray:
        i = self.index_of(name)
        return self.current_rates[i] + np.asarray(self.moves)[:, i]

    def move_bp(self, name: str, quarter: int = 0) -> float:
        return float(np.asarray(self.moves)[quarter, self.index_of(name)] * 10_000)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "first_quarter_moves_bp": {n: round(self.move_bp(n), 3) for n in self.names},
            "iterations": self.iterations,
            "parameters": self.parameters.model_dump(),
        }


def transition(
    banks: Sequence[CentralBank], spillovers: SpilloverMatrix, parameters: DynamicParameters
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The law of motion x[t+1] = A x[t] + sum_i B_i u_i[t], x = (pi, y).

    Coefficients are scaled so a permanent move has the static game's
    long-run effect: dividing a flow coefficient by (1 - persistence) gives
    back the static coefficient.
    """
    n = len(banks)
    a_pi, a_y = parameters.inflation_persistence, parameters.output_persistence
    demand = np.asarray(spillovers.demand, dtype=float)
    exchange = np.asarray(spillovers.exchange, dtype=float)

    # Long run of the output equation: y = -(s / (1 - a_y)) u, so s gives
    # the static own output effect.
    s = spillovers.own_output_effect * (1.0 - a_y)
    # Long run of inflation through output: pi = k y / (1 - a_pi) = -(k s /
    # ((1 - a_pi)(1 - a_y))) u, which must equal the static own inflation
    # effect.
    k = spillovers.own_inflation_effect * (1.0 - a_pi) / spillovers.own_output_effect

    A = np.zeros((2 * n, 2 * n))
    A[:n, :n] = a_pi * np.eye(n)
    A[:n, n:] = k * np.eye(n)
    A[n:, n:] = a_y * np.eye(n)

    B = []
    for i in range(n):
        b = np.zeros(2 * n)
        b[n + i] -= s
        for j in range(n):
            if j == i:
                continue
            # Bank i's move reaches j's output through demand, and moves the
            # i-j exchange rate: i's inflation falls, j's rises.
            b[n + j] -= demand[j, i] * (1.0 - a_y)
            b[i] -= exchange[i, j] * (1.0 - a_pi)
            b[j] += exchange[j, i] * (1.0 - a_pi)
        B.append(b.reshape(-1, 1))
    return A, B


def _augmented(
    A: np.ndarray, B: list[np.ndarray], n: int, parameters: DynamicParameters
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Add last quarter's moves to the state when inertia is on.

    The state becomes (pi, y, u_lag); each bank's move this quarter is next
    quarter's u_lag. Without inertia the state is returned unchanged, so
    ADR 023's solution is reproduced exactly rather than approximately.
    """
    if parameters.inertia_weight == 0.0:
        return A, B, 2 * n
    size = 3 * n
    A_aug = np.zeros((size, size))
    A_aug[: 2 * n, : 2 * n] = A
    B_aug = []
    for i in range(n):
        b = np.zeros((size, 1))
        b[: 2 * n] = B[i]
        b[2 * n + i, 0] = 1.0
        B_aug.append(b)
    return A_aug, B_aug, size


def _solve_lq(A, B, Q, R, N, discount, tolerance, max_iterations):
    """Discounted LQ regulator with a cross term, by value iteration.

    Minimise sum beta^t (x'Qx + u'Ru + 2 x'N u) subject to x' = A x + B u.
    Returns F with u = -F x.
    """
    P = Q.copy()
    for _ in range(max_iterations):
        S = R + discount * B.T @ P @ B
        K = discount * B.T @ P @ A + N.T
        F = np.linalg.solve(S, K)
        P_next = Q + discount * A.T @ P @ A - K.T @ F
        P_next = (P_next + P_next.T) / 2
        if np.max(np.abs(P_next - P)) < tolerance:
            return np.linalg.solve(
                R + discount * B.T @ P_next @ B, discount * B.T @ P_next @ A + N.T
            )
        P = P_next
    raise EngineError("a bank's Riccati equation did not converge")


def dynamic_nash(
    banks: Sequence[CentralBank],
    spillovers: SpilloverMatrix,
    *,
    parameters: DynamicParameters | None = None,
    horizon: int = 12,
) -> DynamicEquilibrium:
    """Feedback Nash equilibrium of the dynamic game, and its rate paths.

    The starting state is each bank's current inflation gap and output gap.
    Paths are simulated from it under the equilibrium rules.
    """
    parameters = parameters or DynamicParameters()
    names = tuple(b.name for b in banks)
    if set(names) != set(spillovers.names):
        raise EngineError(
            f"banks {sorted(names)} do not match the spillover matrix {sorted(spillovers.names)}"
        )
    ordered = tuple(next(b for b in banks if b.name == n) for n in spillovers.names)
    n = len(ordered)
    if n < 2:
        raise EngineError("a game needs at least two banks")

    A, B = transition(ordered, spillovers, parameters)
    A, B, size = _augmented(A, B, n, parameters)
    w_inertia = parameters.inertia_weight
    F = np.zeros((n, size))

    iterations = 0
    for _ in range(parameters.max_iterations):
        iterations += 1
        previous = F.copy()
        for i, bank in enumerate(ordered):
            others = [j for j in range(n) if j != i]
            A_i = A - sum(B[j] @ F[j : j + 1] for j in others)
            # The mean of the others' moves is -G_i x under their rules.
            if spillovers.external_reference is None:
                G_i = F[others].mean(axis=0, keepdims=True)
            else:
                G_i = spillovers.reference_weights()[i : i + 1] @ F
            we = bank.external_weight
            Q = np.zeros((size, size))
            Q[i, i] = bank.inflation_weight
            Q[n + i, n + i] = bank.output_weight
            Q += we * G_i.T @ G_i
            R = np.array([[bank.smoothing_weight + we + w_inertia]])
            N = we * G_i.T
            if w_inertia > 0:
                # w (u_i - u_lag_i)^2 = w u_i^2 - 2 w u_i u_lag_i + w u_lag_i^2
                lag = 2 * n + i
                Q[lag, lag] += w_inertia
                N[lag, 0] -= w_inertia
            F[i] = _solve_lq(
                A_i,
                B[i],
                Q,
                R,
                N,
                parameters.discount,
                parameters.tolerance,
                parameters.max_iterations,
            )[0]
        if np.max(np.abs(F - previous)) < parameters.tolerance:
            break
    else:
        raise EngineError("the dynamic game did not converge to a feedback Nash equilibrium")

    closed = A - sum(B[i] @ F[i : i + 1] for i in range(n))
    if np.max(np.abs(np.linalg.eigvals(closed))) >= 1.0:
        raise EngineError("the equilibrium rules leave the economy unstable")

    x = np.concatenate(
        [
            [b.current_inflation - b.inflation_target for b in ordered],
            [b.current_output_gap for b in ordered],
            np.zeros(size - 2 * n),
        ]
    )
    moves, states = [], [x]
    for _ in range(horizon):
        u = -F @ x
        moves.append(u)
        x = closed @ x
        states.append(x)
    states_arr = np.array(states)

    equilibrium = DynamicEquilibrium(
        names=spillovers.names,
        rules=F,
        current_rates=tuple(b.current_rate for b in ordered),
        moves=np.array(moves),
        inflation_gaps=states_arr[:, :n],
        output_gaps=states_arr[:, n : 2 * n],
        iterations=iterations,
        parameters=parameters,
    )
    log.info(
        "dynamic_nash_solved",
        iterations=iterations,
        spectral_radius=round(float(np.max(np.abs(np.linalg.eigvals(closed)))), 4),
    )
    return equilibrium


def discounted_loss(
    bank_index: int,
    banks: Sequence[CentralBank],
    spillovers: SpilloverMatrix,
    rules: np.ndarray,
    *,
    parameters: DynamicParameters | None = None,
    periods: int = 2_000,
) -> float:
    """Bank i's discounted loss when every bank follows `rules`, by simulation.

    Used to check that an equilibrium rule is a best response: perturbing
    one bank's rule while the others keep theirs must not lower its loss.
    """
    parameters = parameters or DynamicParameters()
    ordered = tuple(next(b for b in banks if b.name == n) for n in spillovers.names)
    n = len(ordered)
    A, B = transition(ordered, spillovers, parameters)
    A, B, size = _augmented(A, B, n, parameters)
    closed = A - sum(B[i] @ rules[i : i + 1] for i in range(n))
    bank = ordered[bank_index]
    x = np.concatenate(
        [
            [b.current_inflation - b.inflation_target for b in ordered],
            [b.current_output_gap for b in ordered],
            np.zeros(size - 2 * n),
        ]
    )
    previous = np.zeros(n)
    total, weight = 0.0, 1.0
    others = [j for j in range(n) if j != bank_index]
    for _ in range(periods):
        u = -rules @ x
        if spillovers.external_reference is None:
            gap = u[bank_index] - u[others].mean()
        else:
            gap = u[bank_index] - spillovers.reference_weights()[bank_index] @ u
        total += weight * (
            bank.inflation_weight * x[bank_index] ** 2
            + bank.output_weight * x[n + bank_index] ** 2
            + bank.smoothing_weight * u[bank_index] ** 2
            + bank.external_weight * gap**2
            + parameters.inertia_weight * (u[bank_index] - previous[bank_index]) ** 2
        )
        previous = u
        weight *= parameters.discount
        x = closed @ x
    return total


def smoothing_coefficient(equilibrium: DynamicEquilibrium, name: str) -> float:
    """The weight a bank's equilibrium rule puts on its own previous move.

    Comparable to the coefficient on the lagged rate in an estimated
    inertial Taylor rule. Zero when the game has no inertia, since the
    previous move is then not part of the state.
    """
    i = equilibrium.index_of(name)
    rules = np.asarray(equilibrium.rules)
    n = len(equilibrium.names)
    if rules.shape[1] == 2 * n:
        return 0.0
    return float(-rules[i, 2 * n + i])
