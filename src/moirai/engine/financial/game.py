"""
Strategic games between policy makers.

A VAR describes transmission: what happened, historically, when a rate
moved. It cannot describe what a central bank *would* do under a policy it
has never followed, because there is no history of that policy to estimate
from. A game model can, because it derives behaviour from stated
objectives rather than from observed correlation.

That is also its weakness. The payoffs here are assumed loss functions,
and how much a central bank dislikes inflation relative to output
volatility is a preference, not an estimable quantity. Every parameter
therefore carries the same confidence marking used for calibration
targets: an assumed number must not be able to pass for a measured one.

Three solution concepts, each answering a different question:

    Nash          both move at once, neither wants to deviate alone
    Stackelberg   one moves first, anticipating the other's reply
    Repeated      cooperation sustained by the threat of future punishment

The choice between them is a claim about the world, not a technical
preference. Nash says the players are symmetric and move simultaneously.
Stackelberg says one is large enough that the other takes its move as
given, which is a reasonable description of the Federal Reserve and most
other central banks.

A note on the Lucas critique. Spillover magnitudes taken from an estimated
VAR are estimated under the historical policy regime. If a game predicts a
genuinely novel strategy, those coefficients no longer describe the
economy, because private expectations would adapt to the new rule. This
layer is defensible for marginal strategic questions within an existing
framework and not for regime change. See the module-level warning emitted
when a solution lies far outside the historical range.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from itertools import product
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger

log = get_logger(__name__)

#: A payoff function maps every player's action to one player's payoff.
PayoffFunction = Callable[[tuple[float, ...]], float]


class Confidence(StrEnum):
    """Where a game parameter came from.

    Duplicated from the calibration module deliberately rather than
    imported: the economy and financial layers should not depend on each
    other, and the alternative is a shared module that exists only to hold
    one enum.
    """

    SOURCED = "sourced"
    DERIVED = "derived"
    ASSUMED = "assumed"


class SolutionConcept(StrEnum):
    """How the players are assumed to interact."""

    NASH = "nash"
    STACKELBERG = "stackelberg"
    COOPERATIVE = "cooperative"
    REPEATED = "repeated"


class Player(BaseModel):
    """One decision maker, its action space and its objective.

    The action space is discrete. Continuous optimisation would be more
    elegant, but policy rates move in increments of twenty five basis
    points and a grid makes the payoff surface directly inspectable, which
    matters more here than elegance: a Nash equilibrium nobody can see the
    shape of is hard to trust.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    actions: tuple[float, ...] = Field(min_length=1)
    payoff: Any = Field(description="Callable mapping an action profile to a payoff.")
    description: str = ""

    @model_validator(mode="after")
    def _actions_are_ordered_and_distinct(self) -> Player:
        if len(set(self.actions)) != len(self.actions):
            raise ValueError(f"{self.name}: actions must be distinct")
        if list(self.actions) != sorted(self.actions):
            raise ValueError(f"{self.name}: actions must be in ascending order")
        return self

    @property
    def n_actions(self) -> int:
        return len(self.actions)


class Outcome(BaseModel):
    """One action profile and the payoff each player receives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actions: dict[str, float]
    payoffs: dict[str, float]
    concept: SolutionConcept
    is_unique: bool = True
    note: str = ""

    @property
    def total_payoff(self) -> float:
        return sum(self.payoffs.values())

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept.value,
            "actions": {k: round(v, 6) for k, v in self.actions.items()},
            "payoffs": {k: round(v, 6) for k, v in self.payoffs.items()},
            "total_payoff": round(self.total_payoff, 6),
            "is_unique": self.is_unique,
            "note": self.note,
        }


class Game(BaseModel):
    """A normal-form game over discrete action spaces."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    players: tuple[Player, ...] = Field(min_length=2)
    title: str = ""

    @model_validator(mode="after")
    def _names_are_unique(self) -> Game:
        names = [p.name for p in self.players]
        if len(set(names)) != len(names):
            raise ValueError(f"player names must be unique, got {names}")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.players)

    @property
    def n_profiles(self) -> int:
        result = 1
        for player in self.players:
            result *= player.n_actions
        return result

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise EngineError(
                f"no player named {name!r}; have {list(self.names)}"
            ) from None

    def profiles(self):
        """Every combination of actions, in player order."""
        return product(*(p.actions for p in self.players))

    def payoffs_at(self, profile: tuple[float, ...]) -> dict[str, float]:
        if len(profile) != len(self.players):
            raise EngineError(
                f"profile has {len(profile)} actions for {len(self.players)} players"
            )
        return {p.name: float(p.payoff(profile)) for p in self.players}

    def payoff_matrix(self) -> np.ndarray:
        """Payoffs for every profile, shaped (a1, a2, ..., n_players).

        Materialising the whole surface is affordable for the small action
        spaces used here and makes every solution concept a lookup rather
        than a separate search. It also lets a caller plot the surface,
        which is the fastest way to see whether an equilibrium is a sharp
        peak or a plateau where the model is barely choosing at all.
        """
        shape = tuple(p.n_actions for p in self.players) + (len(self.players),)
        matrix = np.empty(shape, dtype=float)

        for indices in product(*(range(p.n_actions) for p in self.players)):
            profile = tuple(
                self.players[k].actions[i] for k, i in enumerate(indices)
            )
            payoffs = self.payoffs_at(profile)
            for k, player in enumerate(self.players):
                matrix[indices + (k,)] = payoffs[player.name]

        return matrix


# ---- solution concepts ----------------------------------------------------


def best_responses(
    game: Game, player_name: str, others: dict[str, float]
) -> tuple[float, ...]:
    """Every action maximising one player's payoff, given the others.

    Returns all maximisers rather than one. Ties matter: a game where a
    player is indifferent across several actions has no sharp prediction,
    and silently returning the first would hide that.
    """
    index = game.index_of(player_name)
    player = game.players[index]

    missing = set(game.names) - {player_name} - set(others)
    if missing:
        raise EngineError(f"actions not given for {sorted(missing)}")

    payoffs = []
    for action in player.actions:
        profile = tuple(
            action if k == index else others[name]
            for k, name in enumerate(game.names)
        )
        payoffs.append(player.payoff(profile))

    best = max(payoffs)
    return tuple(
        action
        for action, value in zip(player.actions, payoffs, strict=True)
        if np.isclose(value, best)
    )


def nash_equilibria(game: Game, *, tolerance: float = 1e-9) -> tuple[Outcome, ...]:
    """Every pure-strategy Nash equilibrium.

    A profile is an equilibrium when no player can raise their own payoff
    by changing their action alone. Found by exhaustive search, which is
    tractable for these action spaces and, unlike an iterative method,
    cannot miss an equilibrium or converge to something that is not one.

    Pure strategies only. A game with no pure equilibrium has a mixed one
    by Nash's theorem, and this returns an empty tuple rather than
    pretending otherwise.
    """
    matrix = game.payoff_matrix()
    found: list[Outcome] = []

    for indices in product(*(range(p.n_actions) for p in game.players)):
        stable = True
        for k, player in enumerate(game.players):
            own = matrix[indices + (k,)]
            alternatives = [
                matrix[indices[:k] + (a,) + indices[k + 1 :] + (k,)]
                for a in range(player.n_actions)
            ]
            if max(alternatives) > own + tolerance:
                stable = False
                break

        if stable:
            profile = tuple(
                game.players[k].actions[i] for k, i in enumerate(indices)
            )
            found.append(
                Outcome(
                    actions=dict(zip(game.names, profile, strict=True)),
                    payoffs=game.payoffs_at(profile),
                    concept=SolutionConcept.NASH,
                    is_unique=False,  # corrected below
                )
            )

    if not found:
        log.warning(
            "no_pure_nash_equilibrium",
            game=game.title,
            note="a mixed strategy equilibrium exists but is not computed here",
        )
        return ()

    unique = len(found) == 1
    outcomes = tuple(o.model_copy(update={"is_unique": unique}) for o in found)

    log.info(
        "nash_equilibria_found",
        game=game.title,
        count=len(outcomes),
        unique=unique,
    )
    return outcomes


def stackelberg_equilibrium(
    game: Game, leader_name: str, *, pessimistic: bool = True
) -> Outcome:
    """Solve by backward induction with one player moving first.

    The leader chooses knowing the follower will best-respond, so it
    optimises over the follower's reaction function rather than over the
    follower's actions. This is the right structure when one player is
    large enough that the other takes its move as given, which describes
    the Federal Reserve and most other central banks reasonably well.

    When the follower is indifferent between several replies, the leader
    cannot rely on which it picks. `pessimistic` assumes the worst for the
    leader, which is the conservative reading and avoids a prediction that
    depends on a coin flip going the leader's way.
    """
    if len(game.players) != 2:
        raise EngineError(
            f"Stackelberg solving is implemented for two players, "
            f"got {len(game.players)}"
        )

    leader_index = game.index_of(leader_name)
    follower_index = 1 - leader_index
    leader = game.players[leader_index]
    follower = game.players[follower_index]

    best_value = -np.inf
    best_profile: tuple[float, ...] | None = None
    ties = False

    for leader_action in leader.actions:
        replies = best_responses(
            game, follower.name, {leader.name: leader_action}
        )
        if len(replies) > 1:
            ties = True

        candidates = []
        for reply in replies:
            profile = (
                (leader_action, reply)
                if leader_index == 0
                else (reply, leader_action)
            )
            candidates.append((leader.payoff(profile), profile))

        value, profile = min(candidates) if pessimistic else max(candidates)

        if value > best_value:
            best_value, best_profile = value, profile

    if best_profile is None:
        raise EngineError("no leader action produced a defined payoff")

    note = "leader moves first; follower best-responds"
    if ties:
        note += (
            "; the follower was indifferent at some leader actions, resolved "
            f"{'pessimistically' if pessimistic else 'optimistically'} for the leader"
        )

    outcome = Outcome(
        actions=dict(zip(game.names, best_profile, strict=True)),
        payoffs=game.payoffs_at(best_profile),
        concept=SolutionConcept.STACKELBERG,
        note=note,
    )

    log.info(
        "stackelberg_solved",
        game=game.title,
        leader=leader_name,
        actions=outcome.actions,
        follower_ties=ties,
    )
    return outcome


def cooperative_optimum(game: Game) -> Outcome:
    """The profile maximising the sum of payoffs.

    Not an equilibrium: no player is choosing it, and in general at least
    one would deviate if it could. It exists as the benchmark that makes
    the cost of non-cooperation visible. The gap between this and the Nash
    outcome is the value of a coordination agreement that neither party
    can credibly commit to alone.
    """
    best_total = -np.inf
    best_profile: tuple[float, ...] | None = None

    for profile in game.profiles():
        total = sum(game.payoffs_at(profile).values())
        if total > best_total:
            best_total, best_profile = total, profile

    if best_profile is None:
        raise EngineError("game has no profiles")

    return Outcome(
        actions=dict(zip(game.names, best_profile, strict=True)),
        payoffs=game.payoffs_at(best_profile),
        concept=SolutionConcept.COOPERATIVE,
        note="maximises joint payoff; not individually rational in general",
    )


def cooperation_gain(game: Game) -> dict[str, Any]:
    """How much the players lose by not coordinating.

    Reports the joint payoff under cooperation against the worst Nash
    equilibrium, which is the honest comparison when several equilibria
    exist and nothing selects between them.
    """
    equilibria = nash_equilibria(game)
    if not equilibria:
        raise EngineError("no pure Nash equilibrium to compare against")

    cooperative = cooperative_optimum(game)
    worst = min(equilibria, key=lambda o: o.total_payoff)

    return {
        "cooperative_total": cooperative.total_payoff,
        "nash_total": worst.total_payoff,
        "gain": cooperative.total_payoff - worst.total_payoff,
        "n_equilibria": len(equilibria),
        "cooperative_actions": cooperative.actions,
        "nash_actions": worst.actions,
    }


def sustainable_by_repetition(
    game: Game, target: Outcome, discount_factor: float
) -> dict[str, Any]:
    """Can a non-equilibrium outcome be sustained by repeated play?

    Under grim trigger, a player cooperates while everyone else does and
    reverts permanently to the Nash outcome after any defection. Deviating
    pays off once and costs the difference forever after, so cooperation
    holds when

        one-off gain <= discount / (1 - discount) * per-period loss

    This is the folk theorem in its simplest form. It is why central banks
    coordinate at all without an enforcement mechanism: the relationship
    repeats, and the punishment is the loss of future coordination.

    The result is a statement about incentives, not a prediction. Real
    policy makers face domestic constraints that this ignores entirely.
    """
    if not 0.0 <= discount_factor < 1.0:
        raise EngineError(f"discount factor must lie in [0, 1), got {discount_factor}")

    equilibria = nash_equilibria(game)
    if not equilibria:
        raise EngineError("no pure Nash equilibrium to serve as the punishment")

    punishment = min(equilibria, key=lambda o: o.total_payoff)
    target_profile = tuple(target.actions[name] for name in game.names)

    results: dict[str, Any] = {"sustainable": True, "by_player": {}}

    for player in game.players:
        others = {n: target.actions[n] for n in game.names if n != player.name}
        best = best_responses(game, player.name, others)

        index = game.index_of(player.name)
        deviation_profile = tuple(
            best[0] if k == index else target_profile[k]
            for k in range(len(game.players))
        )

        cooperating = target.payoffs[player.name]
        deviating = player.payoff(deviation_profile)
        punished = punishment.payoffs[player.name]

        one_off_gain = deviating - cooperating
        per_period_loss = cooperating - punished

        if discount_factor == 0.0:
            sustainable = one_off_gain <= 1e-12
            threshold = float("inf") if one_off_gain > 0 else 0.0
        else:
            future_loss = discount_factor / (1 - discount_factor) * per_period_loss
            sustainable = one_off_gain <= future_loss + 1e-12
            threshold = (
                one_off_gain / (one_off_gain + per_period_loss)
                if one_off_gain + per_period_loss > 0
                else 0.0
            )

        results["by_player"][player.name] = {
            "one_off_gain": round(one_off_gain, 6),
            "per_period_loss": round(per_period_loss, 6),
            "minimum_discount_factor": round(min(max(threshold, 0.0), 1.0), 4),
            "sustainable": sustainable,
        }
        results["sustainable"] &= sustainable

    results["discount_factor"] = discount_factor
    results["punishment"] = punishment.actions
    return results


def payoff_surface_is_flat(
    game: Game, player_name: str, *, relative_tolerance: float = 0.01
) -> bool:
    """Is one player nearly indifferent across its whole action space?

    A flat surface means the model is barely choosing, so its equilibrium
    is an artefact of tie-breaking rather than a prediction. Worth knowing
    before quoting an equilibrium action as though the model had selected
    it for a reason.
    """
    index = game.index_of(player_name)
    values = game.payoff_matrix()[..., index]
    spread = float(values.max() - values.min())
    scale = float(np.abs(values).max())
    return scale == 0.0 or spread / scale < relative_tolerance