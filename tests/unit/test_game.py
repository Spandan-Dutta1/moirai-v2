"""Tests for the game solving framework.

The prisoner's dilemma is used throughout because every solution concept
has a known analytic answer there. Nash is mutual defection, the
cooperative optimum is mutual cooperation, and the minimum discount factor
sustaining cooperation follows from the payoff table rather than from a
previous run of this code.

That distinction matters. A test comparing against saved output only
proves the code still does what it did yesterday. A test comparing against
an answer derived independently proves it does the right thing.
"""

import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.game import (
    Confidence,
    Game,
    Outcome,
    Player,
    SolutionConcept,
    best_responses,
    cooperation_gain,
    cooperative_optimum,
    nash_equilibria,
    payoff_surface_is_flat,
    stackelberg_equilibrium,
    sustainable_by_repetition,
)

# Prisoner's dilemma. 0 cooperates, 1 defects.
#   CC 3,3   CD 0,5   DC 5,0   DD 1,1
PD = {
    (0.0, 0.0): (3.0, 3.0),
    (0.0, 1.0): (0.0, 5.0),
    (1.0, 0.0): (5.0, 0.0),
    (1.0, 1.0): (1.0, 1.0),
}

# Coordination game with two equilibria, one better for both.
#   AA 4,4   AB 0,0   BA 0,0   BB 2,2
STAG = {
    (0.0, 0.0): (4.0, 4.0),
    (0.0, 1.0): (0.0, 0.0),
    (1.0, 0.0): (0.0, 0.0),
    (1.0, 1.0): (2.0, 2.0),
}

# Matching pennies: no pure strategy equilibrium exists.
PENNIES = {
    (0.0, 0.0): (1.0, -1.0),
    (0.0, 1.0): (-1.0, 1.0),
    (1.0, 0.0): (-1.0, 1.0),
    (1.0, 1.0): (1.0, -1.0),
}


def two_player_game(table: dict, title: str = "test") -> Game:
    return Game(
        title=title,
        players=(
            Player(name="A", actions=(0.0, 1.0), payoff=lambda p: table[p][0]),
            Player(name="B", actions=(0.0, 1.0), payoff=lambda p: table[p][1]),
        ),
    )


@pytest.fixture
def dilemma() -> Game:
    return two_player_game(PD, "prisoner's dilemma")


@pytest.fixture
def stag() -> Game:
    return two_player_game(STAG, "stag hunt")


# --- Player ----------------------------------------------------------------

def test_player_reports_its_action_count():
    player = Player(name="A", actions=(0.0, 1.0, 2.0), payoff=lambda p: 0.0)
    assert player.n_actions == 3


def test_duplicate_actions_are_rejected():
    with pytest.raises(Exception, match="distinct"):
        Player(name="A", actions=(1.0, 1.0), payoff=lambda p: 0.0)


def test_unordered_actions_are_rejected():
    """Ordering makes the payoff surface readable and indexing predictable."""
    with pytest.raises(Exception, match="ascending"):
        Player(name="A", actions=(2.0, 1.0), payoff=lambda p: 0.0)


def test_player_is_frozen():
    player = Player(name="A", actions=(0.0,), payoff=lambda p: 0.0)
    with pytest.raises(Exception):
        player.name = "B"  # type: ignore[misc]


# --- Game ------------------------------------------------------------------

def test_game_lists_its_players(dilemma):
    assert dilemma.names == ("A", "B")


def test_profile_count_is_the_product_of_action_counts(dilemma):
    assert dilemma.n_profiles == 4


def test_profiles_enumerate_every_combination(dilemma):
    assert set(dilemma.profiles()) == set(PD)


def test_payoffs_at_a_profile(dilemma):
    assert dilemma.payoffs_at((0.0, 1.0)) == {"A": 0.0, "B": 5.0}


def test_a_malformed_profile_is_rejected(dilemma):
    with pytest.raises(EngineError, match="for 2 players"):
        dilemma.payoffs_at((0.0,))


def test_index_of_finds_a_player(dilemma):
    assert dilemma.index_of("B") == 1


def test_index_of_rejects_an_unknown_player(dilemma):
    with pytest.raises(EngineError, match="no player named"):
        dilemma.index_of("C")


def test_duplicate_player_names_are_rejected():
    with pytest.raises(Exception, match="unique"):
        Game(
            players=(
                Player(name="A", actions=(0.0,), payoff=lambda p: 0.0),
                Player(name="A", actions=(0.0,), payoff=lambda p: 0.0),
            )
        )


def test_a_single_player_is_not_a_game():
    with pytest.raises(Exception):
        Game(players=(Player(name="A", actions=(0.0,), payoff=lambda p: 0.0),))


def test_payoff_matrix_has_one_layer_per_player(dilemma):
    assert dilemma.payoff_matrix().shape == (2, 2, 2)


def test_payoff_matrix_matches_the_table(dilemma):
    matrix = dilemma.payoff_matrix()
    assert matrix[0, 1, 0] == 0.0  # A cooperates, B defects
    assert matrix[0, 1, 1] == 5.0


# --- best responses --------------------------------------------------------

def test_best_response_to_cooperation_is_defection(dilemma):
    """Defection is dominant, which is the whole point of the dilemma."""
    assert best_responses(dilemma, "A", {"B": 0.0}) == (1.0,)


def test_best_response_to_defection_is_also_defection(dilemma):
    assert best_responses(dilemma, "A", {"B": 1.0}) == (1.0,)


def test_ties_return_every_maximiser():
    """A player indifferent across actions has no sharp prediction, and
    returning only the first would hide that."""
    flat = {key: (1.0, 1.0) for key in PD}
    game = two_player_game(flat)
    assert best_responses(game, "A", {"B": 0.0}) == (0.0, 1.0)


def test_best_response_requires_the_other_actions(dilemma):
    with pytest.raises(EngineError, match="not given for"):
        best_responses(dilemma, "A", {})


# --- Nash ------------------------------------------------------------------

def test_dilemma_has_a_unique_equilibrium_at_mutual_defection(dilemma):
    equilibria = nash_equilibria(dilemma)
    assert len(equilibria) == 1
    assert equilibria[0].actions == {"A": 1.0, "B": 1.0}
    assert equilibria[0].payoffs == {"A": 1.0, "B": 1.0}


def test_the_equilibrium_is_marked_unique(dilemma):
    assert nash_equilibria(dilemma)[0].is_unique is True


def test_a_coordination_game_has_two_equilibria(stag):
    equilibria = nash_equilibria(stag)
    assert len(equilibria) == 2
    assert {tuple(o.actions.values()) for o in equilibria} == {(0.0, 0.0), (1.0, 1.0)}


def test_multiple_equilibria_are_not_marked_unique(stag):
    assert all(o.is_unique is False for o in nash_equilibria(stag))


def test_no_pure_equilibrium_returns_empty():
    """Matching pennies has only a mixed equilibrium, and this says so
    rather than returning something that is not an equilibrium."""
    assert nash_equilibria(two_player_game(PENNIES)) == ()


def test_equilibria_are_labelled_with_their_concept(dilemma):
    assert nash_equilibria(dilemma)[0].concept is SolutionConcept.NASH


def test_no_player_gains_by_deviating_at_an_equilibrium(dilemma):
    """The defining property, checked directly rather than assumed."""
    equilibrium = nash_equilibria(dilemma)[0]
    for name in dilemma.names:
        others = {n: v for n, v in equilibrium.actions.items() if n != name}
        assert equilibrium.actions[name] in best_responses(dilemma, name, others)


# --- Stackelberg -----------------------------------------------------------

def test_moving_first_does_not_help_against_a_dominant_strategy(dilemma):
    """Commitment has no value when the follower defects regardless."""
    outcome = stackelberg_equilibrium(dilemma, "A")
    assert outcome.actions == {"A": 1.0, "B": 1.0}


def test_leadership_selects_the_better_equilibrium_in_a_coordination_game(stag):
    """Here first-mover advantage is real: the leader picks the good one."""
    outcome = stackelberg_equilibrium(stag, "A")
    assert outcome.actions == {"A": 0.0, "B": 0.0}
    assert outcome.payoffs == {"A": 4.0, "B": 4.0}


def test_either_player_can_lead(stag):
    assert stackelberg_equilibrium(stag, "B").actions == {"A": 0.0, "B": 0.0}


def test_the_outcome_records_its_concept(dilemma):
    assert stackelberg_equilibrium(dilemma, "A").concept is SolutionConcept.STACKELBERG


def test_follower_indifference_is_reported():
    """The leader cannot rely on a coin flip, so ties are noted."""
    flat_follower = {key: (PD[key][0], 1.0) for key in PD}
    game = two_player_game(flat_follower)
    assert "indifferent" in stackelberg_equilibrium(game, "A").note


def test_pessimism_resolves_ties_against_the_leader():
    """When the follower is indifferent, assuming the worst is the
    conservative reading and avoids predicting a favourable coin flip."""
    table = {
        (0.0, 0.0): (5.0, 1.0),
        (0.0, 1.0): (0.0, 1.0),  # follower indifferent when leader plays 0
        (1.0, 0.0): (2.0, 0.0),
        (1.0, 1.0): (2.0, 3.0),
    }
    game = two_player_game(table)
    pessimistic = stackelberg_equilibrium(game, "A", pessimistic=True)
    optimistic = stackelberg_equilibrium(game, "A", pessimistic=False)
    assert optimistic.payoffs["A"] >= pessimistic.payoffs["A"]


def test_three_players_are_rejected():
    game = Game(
        players=tuple(
            Player(name=n, actions=(0.0, 1.0), payoff=lambda p: 0.0)
            for n in ("A", "B", "C")
        )
    )
    with pytest.raises(EngineError, match="two players"):
        stackelberg_equilibrium(game, "A")


def test_an_unknown_leader_is_rejected(dilemma):
    with pytest.raises(EngineError, match="no player named"):
        stackelberg_equilibrium(dilemma, "Z")


# --- cooperative -----------------------------------------------------------

def test_cooperative_optimum_maximises_the_joint_payoff(dilemma):
    outcome = cooperative_optimum(dilemma)
    assert outcome.actions == {"A": 0.0, "B": 0.0}
    assert outcome.total_payoff == 6.0


def test_the_cooperative_outcome_is_not_an_equilibrium(dilemma):
    """Both would deviate, which is why it needs enforcement."""
    cooperative = cooperative_optimum(dilemma)
    equilibrium_actions = {tuple(o.actions.values()) for o in nash_equilibria(dilemma)}
    assert tuple(cooperative.actions.values()) not in equilibrium_actions


def test_cooperation_never_scores_below_nash(dilemma):
    """Optimising jointly cannot do worse than optimising separately."""
    cooperative = cooperative_optimum(dilemma)
    for equilibrium in nash_equilibria(dilemma):
        assert cooperative.total_payoff >= equilibrium.total_payoff


def test_the_cost_of_non_cooperation(dilemma):
    """Six under cooperation against two under mutual defection."""
    assert cooperation_gain(dilemma)["gain"] == pytest.approx(4.0)


def test_a_coordination_game_has_no_cooperation_gap(stag):
    """The good equilibrium is already jointly optimal, so nothing is lost
    when both players happen to reach it."""
    gain = cooperation_gain(stag)
    assert gain["cooperative_total"] == 8.0
    assert gain["gain"] == pytest.approx(4.0)  # against the worse equilibrium


def test_cooperation_gain_needs_an_equilibrium():
    with pytest.raises(EngineError, match="no pure Nash"):
        cooperation_gain(two_player_game(PENNIES))


# --- repetition ------------------------------------------------------------

def test_cooperation_is_unsustainable_without_a_future(dilemma):
    """With no weight on tomorrow, the one-off gain always wins."""
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.0)
    assert result["sustainable"] is False


def test_cooperation_is_sustainable_with_a_patient_player(dilemma):
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.9)
    assert result["sustainable"] is True


def test_the_threshold_matches_the_folk_theorem(dilemma):
    """Defecting gains 5 - 3 = 2 once, and costs 3 - 1 = 2 every period
    thereafter, so cooperation holds from a discount factor of 2/(2+2)."""
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.9)
    assert result["by_player"]["A"]["minimum_discount_factor"] == pytest.approx(0.5)


def test_just_below_the_threshold_fails(dilemma):
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.49)
    assert result["sustainable"] is False


def test_at_the_threshold_it_holds(dilemma):
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.5)
    assert result["sustainable"] is True


def test_the_gain_and_loss_are_reported(dilemma):
    detail = sustainable_by_repetition(
        dilemma, cooperative_optimum(dilemma), 0.9
    )["by_player"]["A"]
    assert detail["one_off_gain"] == pytest.approx(2.0)
    assert detail["per_period_loss"] == pytest.approx(2.0)


def test_the_punishment_is_the_equilibrium(dilemma):
    result = sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), 0.9)
    assert result["punishment"] == {"A": 1.0, "B": 1.0}


def test_an_equilibrium_is_trivially_sustainable(dilemma):
    """Nobody wants to deviate from it in the first place."""
    equilibrium = nash_equilibria(dilemma)[0]
    assert sustainable_by_repetition(dilemma, equilibrium, 0.0)["sustainable"] is True


@pytest.mark.parametrize("discount", [-0.1, 1.0, 1.5])
def test_an_invalid_discount_factor_is_rejected(dilemma, discount):
    with pytest.raises(EngineError, match="discount factor"):
        sustainable_by_repetition(dilemma, cooperative_optimum(dilemma), discount)


# --- flat surfaces ---------------------------------------------------------

def test_a_flat_payoff_surface_is_detected():
    """A player indifferent everywhere is not choosing, so its equilibrium
    action is an artefact of tie-breaking rather than a prediction."""
    flat = {key: (1.0, PD[key][1]) for key in PD}
    assert payoff_surface_is_flat(two_player_game(flat), "A") is True


def test_a_varying_surface_is_not_flat(dilemma):
    assert payoff_surface_is_flat(dilemma, "A") is False


# --- serialisation ---------------------------------------------------------

def test_an_outcome_serialises_for_the_ledger(dilemma):
    payload = nash_equilibria(dilemma)[0].to_ledger_dict()
    assert payload["concept"] == "nash"
    assert payload["actions"] == {"A": 1.0, "B": 1.0}
    assert payload["total_payoff"] == 2.0


def test_an_outcome_is_frozen(dilemma):
    outcome = nash_equilibria(dilemma)[0]
    with pytest.raises(Exception):
        outcome.note = "changed"  # type: ignore[misc]


def test_confidence_levels_are_available():
    """Game parameters carry provenance for the same reason calibration
    targets do: an assumed number must not pass for a measured one."""
    assert {c.value for c in Confidence} == {"sourced", "derived", "assumed"}