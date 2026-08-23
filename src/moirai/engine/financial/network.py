"""
A network of central banks rather than a pair.

The two-player case is a special case of a system. What makes the system
interesting is that it is not symmetric: a Federal Reserve move reaches
every other economy, while an RBI move reaches almost none. Modelling five
banks with identical spillovers would erase the one feature that makes the
international monetary system worth modelling at all.

Spillovers are therefore tiered by the economy's systemic weight. That is
an assumption, and it is marked as one, but it encodes something real: the
dollar's reserve status means the Fed exports its policy stance whether it
intends to or not, and the literature on the global financial cycle
documents the asymmetry directly.

The tiering is a placeholder for estimation. `SpilloverMatrix.from_var`
accepts magnitudes recovered from a multi-country VAR, and any matrix
built that way is marked DERIVED rather than ASSUMED. Until that
estimation exists, the tiers are a structured guess and the confidence
flag says so.

On solution concepts at N players:

  * Nash generalises exactly. Each bank's first order condition is linear
    in every rate, so the equilibrium solves an N by N linear system. That
    is exact and instant regardless of N, unlike enumeration, which for
    five banks on a twenty five basis point grid would be thirty nine
    million profiles.

  * Stackelberg does not generalise cleanly. With three or more players a
    move order has to be stated, and "the Fed leads, everyone else follows
    simultaneously" is a different game from a full sequential ordering.
    Only the former is implemented, because it matches the structure of
    the actual system and can be defended.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.central_banks import CentralBank
from moirai.engine.financial.game import Confidence

log = get_logger(__name__)

#: Above this, the reaction system is near-singular and the equilibrium is
#: extremely sensitive to the assumed weights.
MAX_CONDITION_NUMBER = 1e4


class SystemicTier(StrEnum):
    """How far an economy's policy reaches beyond its own borders.

    Not a judgement about importance. It is a statement about monetary
    spillover, which tracks reserve currency status and financial openness
    rather than the size of the economy. China is the clearest case: the
    second largest economy, and a comparatively contained monetary
    spillover because of capital account management.
    """

    #: Issues the dominant reserve currency. Policy is exported globally.
    ANCHOR = "anchor"
    #: A major reserve currency with wide regional reach.
    MAJOR = "major"
    #: Significant regionally, limited globally.
    REGIONAL = "regional"
    #: A price taker in the international monetary system.
    RECIPIENT = "recipient"

    @property
    def outward_strength(self) -> float:
        """Multiplier on spillovers this economy sends."""
        return _OUTWARD[self]

    @property
    def inward_sensitivity(self) -> float:
        """Multiplier on spillovers this economy receives.

        Inversely related to outward strength. An economy that exports its
        policy stance is largely insulated from others; one that exports
        nothing absorbs everything.
        """
        return _INWARD[self]


_OUTWARD: dict[SystemicTier, float] = {
    SystemicTier.ANCHOR: 1.00,
    SystemicTier.MAJOR: 0.45,
    SystemicTier.REGIONAL: 0.20,
    SystemicTier.RECIPIENT: 0.05,
}

_INWARD: dict[SystemicTier, float] = {
    SystemicTier.ANCHOR: 0.10,
    SystemicTier.MAJOR: 0.50,
    SystemicTier.REGIONAL: 0.85,
    SystemicTier.RECIPIENT: 1.00,
}


class SpilloverMatrix(BaseModel):
    """How each bank's rate reaches every other economy.

    Three channels, each an N by N matrix where entry (i, j) is the effect
    of bank j's tightening on economy i:

        demand      weaker foreign demand lowers the home output gap
        exchange    a rate differential moves the currency and import costs
        own         the domestic effect of a bank's own move

    Asymmetry is the point. The Fed to India entry is large; India to the
    Fed is near zero. A symmetric matrix would be a different, less
    interesting, and less accurate model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    names: tuple[str, ...] = Field(min_length=2)
    demand: Any = Field(description="(n, n) demand spillover matrix.")
    exchange: Any = Field(description="(n, n) exchange rate passthrough matrix.")
    own_output_effect: float = Field(default=1.20, ge=0)
    own_inflation_effect: float = Field(default=0.80, ge=0)
    confidence: Confidence = Confidence.ASSUMED
    note: str = ""

    @model_validator(mode="after")
    def _matrices_are_square_and_matched(self) -> SpilloverMatrix:
        n = len(self.names)
        for label, matrix in (("demand", self.demand), ("exchange", self.exchange)):
            array = np.asarray(matrix, dtype=float)
            if array.shape != (n, n):
                raise ValueError(f"{label} is {array.shape}, expected ({n}, {n})")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{label} contains NaN or infinity")
        if len(set(self.names)) != n:
            raise ValueError(f"names must be unique, got {list(self.names)}")
        return self

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise EngineError(
                f"{name!r} is not in this network; have {list(self.names)}"
            ) from None

    @property
    def is_symmetric(self) -> bool:
        """A symmetric matrix would mean every economy spills over equally.

        Reported because it should be false. If it is true, the tiering has
        been lost and the model no longer represents a hierarchy.
        """
        return bool(np.allclose(self.demand, self.demand.T))

    def outward_influence(self) -> dict[str, float]:
        """Total spillover each bank sends, excluding its own effect."""
        return {
            name: float(self.demand[:, j].sum() - self.demand[j, j])
            for j, name in enumerate(self.names)
        }

    def inward_exposure(self) -> dict[str, float]:
        """Total spillover each economy receives from the others."""
        return {
            name: float(self.demand[i, :].sum() - self.demand[i, i])
            for i, name in enumerate(self.names)
        }

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "confidence": self.confidence.value,
            "note": self.note,
            "is_symmetric": self.is_symmetric,
            "outward_influence": {
                k: round(v, 4) for k, v in self.outward_influence().items()
            },
            "inward_exposure": {
                k: round(v, 4) for k, v in self.inward_exposure().items()
            },
            "demand": np.asarray(self.demand).round(4).tolist(),
            "exchange": np.asarray(self.exchange).round(4).tolist(),
        }

    @classmethod
    def from_tiers(
        cls,
        tiers: dict[str, SystemicTier],
        *,
        base_demand: float = 0.30,
        base_exchange: float = 0.35,
        own_output_effect: float = 1.20,
        own_inflation_effect: float = 0.80,
    ) -> SpilloverMatrix:
        """Build a matrix from systemic tiers.

        The spillover from j to i is the product of j's outward strength
        and i's inward sensitivity, scaled by a base magnitude. So an
        anchor economy reaching a recipient gets the full base, while a
        recipient reaching an anchor gets base times 0.05 times 0.10,
        which is effectively nothing.
        """
        names = tuple(tiers)
        n = len(names)
        if n < 2:
            raise EngineError("a network needs at least two banks")

        demand = np.zeros((n, n))
        exchange = np.zeros((n, n))

        for i, receiver in enumerate(names):
            for j, sender in enumerate(names):
                if i == j:
                    continue
                strength = (
                    tiers[sender].outward_strength * tiers[receiver].inward_sensitivity
                )
                demand[i, j] = base_demand * strength
                exchange[i, j] = base_exchange * strength

        return cls(
            names=names,
            demand=demand,
            exchange=exchange,
            own_output_effect=own_output_effect,
            own_inflation_effect=own_inflation_effect,
            confidence=Confidence.ASSUMED,
            note=(
                "Built from systemic tiers rather than estimated. The tier "
                "assignment encodes the hierarchy documented in the global "
                "financial cycle literature, but the magnitudes are a "
                "structured guess. Replace with from_var once a "
                "multi-country VAR is available."
            ),
        )

    @classmethod
    def from_var(
        cls,
        names: tuple[str, ...],
        demand: np.ndarray,
        exchange: np.ndarray,
        *,
        source: str,
        own_output_effect: float = 1.20,
        own_inflation_effect: float = 0.80,
    ) -> SpilloverMatrix:
        """Build a matrix from estimated impulse responses.

        The honest source. Entry (i, j) should be the peak response of
        economy i's output gap to a one standard deviation policy shock in
        economy j, taken from a multi-country VAR.
        """
        if not source.strip():
            raise EngineError("an estimated matrix must cite its source")

        return cls(
            names=names,
            demand=np.asarray(demand, dtype=float),
            exchange=np.asarray(exchange, dtype=float),
            own_output_effect=own_output_effect,
            own_inflation_effect=own_inflation_effect,
            confidence=Confidence.DERIVED,
            note=f"estimated: {source}",
        )


#: Tier assignment for the five banks currently modelled.
#:
#: The Fed is the anchor: dollar invoicing, dollar funding markets and the
#: global financial cycle mean its stance is exported whether or not that
#: is intended. The ECB and BoJ issue major reserve currencies with wide
#: regional reach. The Bank of England is regional: significant in European
#: financial markets, limited globally. The RBI is a recipient, which is
#: precisely why its published objective function includes capital flow
#: management while the Fed's does not.
DEFAULT_TIERS: dict[str, SystemicTier] = {
    "Federal Reserve": SystemicTier.ANCHOR,
    "European Central Bank": SystemicTier.MAJOR,
    "Bank of Japan": SystemicTier.MAJOR,
    "Bank of England": SystemicTier.REGIONAL,
    "Reserve Bank of India": SystemicTier.RECIPIENT,
}


class NetworkEquilibrium(BaseModel):
    """An equilibrium across the whole network."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rates: dict[str, float]
    losses: dict[str, float]
    concept: str
    condition_number: float
    is_well_conditioned: bool
    leader: str | None = None
    note: str = ""

    @property
    def total_loss(self) -> float:
        return sum(self.losses.values())

    def moves_bp(self, banks: dict[str, CentralBank]) -> dict[str, float]:
        """Basis points each bank moves from its current rate."""
        return {
            name: (rate - banks[name].current_rate) * 10_000
            for name, rate in self.rates.items()
        }

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "leader": self.leader,
            "rates": {k: round(v, 6) for k, v in self.rates.items()},
            "losses": {k: round(v, 8) for k, v in self.losses.items()},
            "total_loss": round(self.total_loss, 8),
            "condition_number": round(self.condition_number, 2),
            "is_well_conditioned": self.is_well_conditioned,
            "note": self.note,
        }


def _reaction_system(
    banks: tuple[CentralBank, ...], spillovers: SpilloverMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the linear first order conditions for every bank.

    Bank i's loss is quadratic in every rate, so its first order condition
    is linear. Stacking them gives A r = b, and the equilibrium is the
    solution. This is the N player generalisation of the two by two system
    solved in central_banks.analytic_nash.
    """
    n = len(banks)
    demand = np.asarray(spillovers.demand, dtype=float)
    exchange = np.asarray(spillovers.exchange, dtype=float)

    matrix = np.zeros((n, n))
    rhs = np.zeros(n)

    for i, bank in enumerate(banks):
        wi, wo = bank.inflation_weight, bank.output_weight
        we, ws = bank.external_weight, bank.smoothing_weight

        # Inflation of economy i as a linear function of the rate vector.
        # inflation_i = pi0_i - own*(r_i - r0_i) - sum_j x_ij*((r_i - r0_i) - (r_j - r0_j))
        inf_coefficients = np.zeros(n)
        inf_coefficients[i] = -spillovers.own_inflation_effect
        if we > 0:
            for j in range(n):
                if j == i:
                    continue
                inf_coefficients[i] -= exchange[i, j]
                inf_coefficients[j] += exchange[i, j]

        inf_constant = bank.current_inflation - bank.inflation_target
        inf_constant -= float(inf_coefficients @ np.array([b.current_rate for b in banks]))

        # Output gap of economy i.
        out_coefficients = np.zeros(n)
        out_coefficients[i] = -spillovers.own_output_effect
        for j in range(n):
            if j != i:
                out_coefficients[j] = -demand[i, j]

        out_constant = -float(
            out_coefficients @ np.array([b.current_rate for b in banks])
        )

        # d/dr_i of the quadratic loss, collected into a linear equation.
        for j in range(n):
            matrix[i, j] = (
                2 * wi * inf_coefficients[i] * inf_coefficients[j]
                + 2 * wo * out_coefficients[i] * out_coefficients[j]
            )
        matrix[i, i] += 2 * (we + ws)

        rhs[i] = -(
            2 * wi * inf_coefficients[i] * inf_constant
            + 2 * wo * out_coefficients[i] * out_constant
            - 2 * ws * bank.current_rate
        )

        # The external term penalises the differential against every other
        # bank, which contributes to both the diagonal and the off-diagonal.
        if we > 0:
            for j in range(n):
                if j != i:
                    matrix[i, j] -= 2 * we / max(n - 1, 1)

    return matrix, rhs


def _losses_at(
    banks: tuple[CentralBank, ...], rates: np.ndarray, spillovers: SpilloverMatrix
) -> dict[str, float]:
    """Realised loss for every bank at a given rate vector."""
    n = len(banks)
    demand = np.asarray(spillovers.demand, dtype=float)
    exchange = np.asarray(spillovers.exchange, dtype=float)
    moves = np.array([rates[i] - banks[i].current_rate for i in range(n)])

    losses: dict[str, float] = {}
    for i, bank in enumerate(banks):
        inflation = bank.current_inflation - spillovers.own_inflation_effect * moves[i]
        if bank.external_weight > 0:
            for j in range(n):
                if j != i:
                    inflation -= exchange[i, j] * (moves[i] - moves[j])

        output = -spillovers.own_output_effect * moves[i]
        for j in range(n):
            if j != i:
                output -= demand[i, j] * moves[j]

        differential = (
            float(np.mean([abs(rates[i] - rates[j]) for j in range(n) if j != i]))
            if bank.external_weight > 0
            else 0.0
        )

        losses[bank.name] = bank.loss(
            inflation, output, float(rates[i]), external_gap=differential
        )
    return losses


def network_nash(
    banks: tuple[CentralBank, ...], spillovers: SpilloverMatrix
) -> NetworkEquilibrium:
    """Solve the simultaneous-move equilibrium for the whole network.

    Exact rather than searched. Enumerating a five bank game on a twenty
    five basis point grid would be thirty nine million profiles; the linear
    system is five equations.
    """
    if len(banks) < 2:
        raise EngineError("a network needs at least two banks")

    names = tuple(b.name for b in banks)
    if set(names) != set(spillovers.names):
        raise EngineError(
            f"banks {sorted(names)} do not match the spillover matrix "
            f"{sorted(spillovers.names)}"
        )

    # Reorder the banks to match the matrix.
    ordered = tuple(next(b for b in banks if b.name == n) for n in spillovers.names)

    matrix, rhs = _reaction_system(ordered, spillovers)
    condition = float(np.linalg.cond(matrix))

    if not np.isfinite(condition) or condition > 1e10:
        raise EngineError(
            f"the reaction system is singular (condition {condition:.2e}). The "
            f"banks' best responses are parallel, so no unique equilibrium "
            f"exists. Check the weights and the spillover magnitudes."
        )

    solution = np.linalg.solve(matrix, rhs)
    rates = {ordered[i].name: float(solution[i]) for i in range(len(ordered))}

    equilibrium = NetworkEquilibrium(
        rates=rates,
        losses=_losses_at(ordered, solution, spillovers),
        concept="network_nash",
        condition_number=condition,
        is_well_conditioned=condition < MAX_CONDITION_NUMBER,
        note=f"simultaneous move equilibrium across {len(ordered)} banks",
    )

    log.info(
        "network_nash_solved",
        n_banks=len(ordered),
        condition=round(condition, 1),
        rates={k: round(v, 5) for k, v in rates.items()},
    )
    return equilibrium


def network_stackelberg(
    banks: tuple[CentralBank, ...],
    spillovers: SpilloverMatrix,
    leader_name: str,
    *,
    grid: tuple[float, ...] | None = None,
) -> NetworkEquilibrium:
    """One bank moves first; the rest respond simultaneously.

    This is a specific game, not the only N player Stackelberg game. A full
    sequential ordering, where the Fed moves, then the ECB, then the RBI,
    is a different game with a different answer. The single-leader form is
    implemented because it matches the structure of the actual system:
    other central banks take Federal Reserve policy as given, and they do
    not queue up behind each other.

    Solved by searching the leader's action grid and computing the
    followers' simultaneous equilibrium at each point. The followers'
    problem is linear, so each evaluation is one linear solve.
    """
    names = tuple(b.name for b in banks)
    if leader_name not in names:
        raise EngineError(f"no bank named {leader_name!r}; have {sorted(names)}")
    if len(banks) < 2:
        raise EngineError("a network needs at least two banks")

    leader = next(b for b in banks if b.name == leader_name)
    followers = tuple(b for b in banks if b.name != leader_name)

    if grid is None:
        centre = leader.current_rate
        grid = tuple(np.round(np.linspace(max(centre - 0.03, 0.0), centre + 0.05, 33), 6))

    best_loss = np.inf
    best: NetworkEquilibrium | None = None

    for candidate in grid:
        # Pin the leader by giving it an overwhelming smoothing weight
        # anchored at the candidate rate: it then has no incentive to move.
        pinned = leader.model_copy(
            update={"current_rate": float(candidate), "smoothing_weight": 1e6}
        )
        trial = (pinned, *followers)

        try:
            outcome = network_nash(trial, spillovers)
        except EngineError:
            continue

        rates = np.array([outcome.rates[n] for n in spillovers.names])
        ordered = tuple(
            next(b for b in trial if b.name == n) for n in spillovers.names
        )
        # Evaluate the leader's loss with its true preferences, not the
        # pinned ones, since the pin is a solving device rather than a
        # change in what the leader wants.
        true_ordered = tuple(
            leader if b.name == leader_name else b for b in ordered
        )
        losses = _losses_at(true_ordered, rates, spillovers)

        if losses[leader_name] < best_loss:
            best_loss = losses[leader_name]
            best = NetworkEquilibrium(
                rates=dict(outcome.rates),
                losses=losses,
                concept="network_stackelberg",
                condition_number=outcome.condition_number,
                is_well_conditioned=outcome.is_well_conditioned,
                leader=leader_name,
                note=(
                    f"{leader_name} moves first; the remaining "
                    f"{len(followers)} banks respond simultaneously. A full "
                    f"sequential ordering would be a different game."
                ),
            )

    if best is None:
        raise EngineError("no leader action produced a solvable follower equilibrium")

    log.info(
        "network_stackelberg_solved",
        leader=leader_name,
        n_followers=len(followers),
        rates={k: round(v, 5) for k, v in best.rates.items()},
    )
    return best


def transmission_ranking(
    banks: tuple[CentralBank, ...], spillovers: SpilloverMatrix, shocked: str
) -> dict[str, Any]:
    """Who moves most when one bank faces an inflation shock.

    The question the network exists to answer: the Fed tightens, and how
    far does that force everyone else to follow?
    """
    baseline = network_nash(banks, spillovers)

    shocked_banks = tuple(
        b.model_copy(update={"current_inflation": b.current_inflation + 0.02})
        if b.name == shocked
        else b
        for b in banks
    )
    after = network_nash(shocked_banks, spillovers)

    responses = {
        name: (after.rates[name] - baseline.rates[name]) * 10_000
        for name in baseline.rates
    }
    origin = responses[shocked]

    return {
        "shocked": shocked,
        "shock": "inflation up two percentage points",
        "responses_bp": {k: round(v, 1) for k, v in responses.items()},
        "pass_through": {
            k: round(v / origin, 4) if origin != 0 else 0.0
            for k, v in responses.items()
            if k != shocked
        },
        "baseline_rates": {k: round(v, 6) for k, v in baseline.rates.items()},
    }