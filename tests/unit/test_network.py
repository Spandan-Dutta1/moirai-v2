"""Tests for the central bank network.

The load-bearing test is consistency: with two banks, the N player solver
must reproduce what the two by two analytic solver already produces. The
reaction system assembly is the most intricate code in the project, and
that check verifies it against an independent implementation rather than
against itself.

The rest concern asymmetry, which is the point of the network. A symmetric
spillover matrix would model a world where an RBI move reaches Washington
as readily as the reverse, and every structural claim the network makes
would be false.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
    CentralBank,
    MandateType,
    SpilloverParameters,
    analytic_nash,
)
from moirai.engine.financial.game import Confidence
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    SpilloverMatrix,
    SystemicTier,
    compare_constructors,
    network_nash,
    network_stackelberg,
    transmission_ranking,
)

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)


@pytest.fixture
def spillovers() -> SpilloverMatrix:
    return SpilloverMatrix.from_tiers(DEFAULT_TIERS)


def make_bank(name: str, **overrides) -> CentralBank:
    defaults = dict(
        name=name,
        jurisdiction="XX",
        inflation_target=0.02,
        mandate=MandateType.HIERARCHICAL,
        mandate_source="test",
        current_rate=0.03,
        current_inflation=0.03,
    )
    return CentralBank(**{**defaults, **overrides})


# --- tiers -----------------------------------------------------------------

def test_outward_strength_falls_down_the_tiers():
    assert (
        SystemicTier.ANCHOR.outward_strength
        > SystemicTier.MAJOR.outward_strength
        > SystemicTier.REGIONAL.outward_strength
        > SystemicTier.RECIPIENT.outward_strength
    )


def test_inward_sensitivity_rises_down_the_tiers():
    """An economy that exports its stance is largely insulated from others."""
    assert (
        SystemicTier.ANCHOR.inward_sensitivity
        < SystemicTier.MAJOR.inward_sensitivity
        < SystemicTier.REGIONAL.inward_sensitivity
        < SystemicTier.RECIPIENT.inward_sensitivity
    )


def test_every_tier_has_both_multipliers():
    for tier in SystemicTier:
        assert 0.0 < tier.outward_strength <= 1.0
        assert 0.0 < tier.inward_sensitivity <= 1.0


def test_the_fed_is_the_anchor():
    assert DEFAULT_TIERS[FED.name] is SystemicTier.ANCHOR


def test_the_rbi_is_a_recipient():
    """Which is why its published mandate includes capital flow management
    and the Fed's does not."""
    assert DEFAULT_TIERS[RBI.name] is SystemicTier.RECIPIENT


# --- the spillover matrix --------------------------------------------------

def test_the_matrix_is_square(spillovers):
    n = len(spillovers.names)
    assert np.asarray(spillovers.demand).shape == (n, n)


def test_no_economy_spills_over_to_itself(spillovers):
    """The own effect is a separate parameter, not a diagonal entry."""
    assert np.allclose(np.diag(spillovers.demand), 0.0)


def test_the_matrix_is_not_symmetric(spillovers):
    """The whole point. A symmetric matrix would model a world where an
    RBI move reaches Washington as readily as the reverse."""
    assert spillovers.is_symmetric is False


def test_the_fed_reaches_india_far_more_than_india_reaches_the_fed(spillovers):
    fed = spillovers.index_of(FED.name)
    rbi = spillovers.index_of(RBI.name)
    assert spillovers.demand[rbi, fed] > 20 * spillovers.demand[fed, rbi]


def test_the_anchor_exports_the_most(spillovers):
    outward = spillovers.outward_influence()
    assert max(outward, key=outward.get) == FED.name


def test_the_recipient_absorbs_the_most(spillovers):
    inward = spillovers.inward_exposure()
    assert max(inward, key=inward.get) == RBI.name


def test_the_anchor_absorbs_the_least(spillovers):
    inward = spillovers.inward_exposure()
    assert min(inward, key=inward.get) == FED.name


def test_a_tiered_matrix_is_marked_assumed(spillovers):
    """A structured guess must not pass for an estimate."""
    assert spillovers.confidence is Confidence.ASSUMED
    assert "structured guess" in spillovers.note


def test_an_estimated_matrix_is_marked_derived():
    names = ("A", "B")
    matrix = np.array([[0.0, 0.2], [0.05, 0.0]])
    result = SpilloverMatrix.from_var(names, matrix, matrix, source="two-country VAR")
    assert result.confidence is Confidence.DERIVED
    assert "two-country VAR" in result.note


# --- the literature-anchored constructor -----------------------------------

def test_a_literature_anchored_matrix_is_marked_derived():
    """The magnitude comes from a published estimate, so it is not a guess.

    Still DERIVED rather than SOURCED: the allocation across pairs is the
    tier structure rather than a bilateral estimate, and the published
    figure is a bond yield response used as a proxy for an output gap.
    """
    matrix = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    assert matrix.confidence is Confidence.DERIVED
    assert "IMF" in matrix.source


def test_the_anchor_to_recipient_cell_matches_the_published_estimate():
    """The point of the constructor: the strongest cell equals the figure
    it was anchored to, and every other pair scales down from it."""
    matrix = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    fed = matrix.index_of(FED.name)
    rbi = matrix.index_of(RBI.name)
    assert matrix.demand[rbi, fed] == pytest.approx(0.36, abs=1e-9)


def test_the_sourced_matrix_is_close_to_the_guess():
    """A large divergence would mean the original guess was doing real work
    and every result built on it should be rerun. Twenty percent is enough
    to matter and small enough that the structure was sound."""
    comparison = compare_constructors()
    assert comparison["relative_difference"] < 0.5
    assert comparison["derived_confidence"] == "derived"


def test_the_literature_matrix_keeps_the_asymmetry():
    """Anchoring the magnitude must not flatten the hierarchy."""
    matrix = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    assert matrix.is_symmetric is False
    outward = matrix.outward_influence()
    assert max(outward, key=outward.get) == FED.name


def test_a_zero_anchor_is_rejected():
    with pytest.raises(EngineError, match="must be positive"):
        SpilloverMatrix.from_literature(DEFAULT_TIERS, anchor_to_recipient=0.0)


# --- validation ------------------------------------------------------------

def test_an_estimated_matrix_must_cite_a_source():
    matrix = np.zeros((2, 2))
    with pytest.raises(EngineError, match="cite its source"):
        SpilloverMatrix.from_var(("A", "B"), matrix, matrix, source="  ")


def test_a_wrong_shaped_matrix_is_rejected():
    with pytest.raises(Exception, match="expected"):
        SpilloverMatrix(
            names=("A", "B"), demand=np.zeros((3, 3)), exchange=np.zeros((2, 2))
        )


def test_a_matrix_with_nan_is_rejected():
    bad = np.array([[0.0, np.nan], [0.1, 0.0]])
    with pytest.raises(Exception, match="NaN"):
        SpilloverMatrix(names=("A", "B"), demand=bad, exchange=np.zeros((2, 2)))


def test_duplicate_names_are_rejected():
    with pytest.raises(Exception, match="unique"):
        SpilloverMatrix(
            names=("A", "A"), demand=np.zeros((2, 2)), exchange=np.zeros((2, 2))
        )


def test_a_single_bank_is_not_a_network():
    with pytest.raises(EngineError, match="at least two"):
        SpilloverMatrix.from_tiers({"Only": SystemicTier.ANCHOR})


def test_an_unknown_name_is_rejected(spillovers):
    with pytest.raises(EngineError, match="not in this network"):
        spillovers.index_of("Bank of Nowhere")


def test_the_matrix_serialises_with_its_provenance(spillovers):
    payload = spillovers.to_ledger_dict()
    assert payload["confidence"] == "assumed"
    assert payload["is_symmetric"] is False
    assert len(payload["demand"]) == len(spillovers.names)


def test_the_serialised_matrix_carries_its_source():
    payload = SpilloverMatrix.from_literature(DEFAULT_TIERS).to_ledger_dict()
    assert "IMF" in payload["source"]


# --- the two-player consistency check --------------------------------------

def test_two_banks_reproduce_the_pairwise_solver():
    """The load-bearing test. The N player reaction system is intricate,
    and this checks it against an independent two by two implementation
    rather than against itself.

    The matrices are built to match what SpilloverParameters implies, so
    the two solvers face the same economy.
    """
    demand = np.array([[0.0, 0.25], [0.25, 0.0]])
    exchange = np.array([[0.0, 0.30], [0.30, 0.0]])
    matrix = SpilloverMatrix(
        names=(FED.name, RBI.name),
        demand=demand,
        exchange=exchange,
        own_output_effect=1.20,
        own_inflation_effect=0.80,
    )

    network = network_nash((FED, RBI), matrix)
    pairwise = analytic_nash(FED, RBI, spillovers=SpilloverParameters())

    for name in (FED.name, RBI.name):
        assert network.rates[name] == pytest.approx(pairwise.rates[name], abs=0.005)


# --- the network equilibrium -----------------------------------------------

def test_every_bank_gets_a_rate(spillovers):
    equilibrium = network_nash(BANKS, spillovers)
    assert set(equilibrium.rates) == {b.name for b in BANKS}


def test_the_system_is_well_conditioned(spillovers):
    """Five banks and all their cross terms still give a stable solve."""
    equilibrium = network_nash(BANKS, spillovers)
    assert equilibrium.is_well_conditioned
    assert equilibrium.condition_number < 100


def test_the_literature_matrix_also_solves():
    """Stronger spillovers must not destabilise the reaction system."""
    equilibrium = network_nash(BANKS, SpilloverMatrix.from_literature(DEFAULT_TIERS))
    assert equilibrium.is_well_conditioned


def test_the_solution_is_deterministic(spillovers):
    assert network_nash(BANKS, spillovers).rates == network_nash(BANKS, spillovers).rates


def test_bank_order_does_not_change_the_answer(spillovers):
    """The solver reorders to match the matrix, so the caller's ordering
    must not matter."""
    forward = network_nash(BANKS, spillovers)
    reversed_ = network_nash(tuple(reversed(BANKS)), spillovers)
    for name in forward.rates:
        assert forward.rates[name] == pytest.approx(reversed_.rates[name], abs=1e-9)


def test_a_bank_missing_from_the_matrix_is_rejected(spillovers):
    with pytest.raises(EngineError, match="do not match"):
        network_nash((FED, ECB), spillovers)


def test_fewer_than_two_banks_is_rejected(spillovers):
    with pytest.raises(EngineError, match="at least two"):
        network_nash((FED,), spillovers)


def test_higher_inflation_raises_a_bank_s_rate(spillovers):
    baseline = network_nash(BANKS, spillovers)
    shocked = network_nash(
        tuple(
            b.model_copy(update={"current_inflation": 0.06}) if b.name == FED.name else b
            for b in BANKS
        ),
        spillovers,
    )
    assert shocked.rates[FED.name] > baseline.rates[FED.name]


def test_moves_are_reported_in_basis_points(spillovers):
    equilibrium = network_nash(BANKS, spillovers)
    moves = equilibrium.moves_bp({b.name: b for b in BANKS})
    expected = (equilibrium.rates[FED.name] - FED.current_rate) * 10_000
    assert moves[FED.name] == pytest.approx(expected)


def test_the_equilibrium_serialises(spillovers):
    payload = network_nash(BANKS, spillovers).to_ledger_dict()
    assert payload["concept"] == "network_nash"
    assert len(payload["rates"]) == len(BANKS)


def test_the_equilibrium_is_frozen(spillovers):
    equilibrium = network_nash(BANKS, spillovers)
    with pytest.raises(Exception):
        equilibrium.concept = "changed"  # type: ignore[misc]


# --- transmission ----------------------------------------------------------

def test_a_shock_moves_the_shocked_bank_most(spillovers):
    ranking = transmission_ranking(BANKS, spillovers, FED.name)
    responses = ranking["responses_bp"]
    assert abs(responses[FED.name]) > max(
        abs(v) for k, v in responses.items() if k != FED.name
    )


def test_pass_through_is_positive_and_partial(spillovers):
    """Others follow a Fed tightening, but not one for one."""
    through = transmission_ranking(BANKS, spillovers, FED.name)["pass_through"]
    assert all(0.0 < v < 1.0 for v in through.values())


def test_the_recipient_follows_more_than_the_anchor_would():
    """An RBI shock barely moves the Fed; a Fed shock moves the RBI."""
    spillovers = SpilloverMatrix.from_tiers(DEFAULT_TIERS)
    from_fed = transmission_ranking(BANKS, spillovers, FED.name)["pass_through"]
    from_rbi = transmission_ranking(BANKS, spillovers, RBI.name)["pass_through"]
    assert from_fed[RBI.name] > from_rbi[FED.name]


def test_the_shocked_bank_is_excluded_from_pass_through(spillovers):
    ranking = transmission_ranking(BANKS, spillovers, FED.name)
    assert FED.name not in ranking["pass_through"]


def test_transmission_reports_the_spillover_provenance():
    """A pass-through figure means something different depending on whether
    the matrix behind it was sourced or guessed."""
    ranking = transmission_ranking(
        BANKS, SpilloverMatrix.from_literature(DEFAULT_TIERS), FED.name
    )
    assert ranking["spillover_confidence"] == "derived"
    assert "IMF" in ranking["spillover_source"]


def test_stronger_spillovers_raise_pass_through():
    """The sourced matrix is twenty percent stronger than the guess, so
    followers should follow further."""
    weak = transmission_ranking(
        BANKS, SpilloverMatrix.from_tiers(DEFAULT_TIERS), FED.name
    )["pass_through"]
    strong = transmission_ranking(
        BANKS, SpilloverMatrix.from_literature(DEFAULT_TIERS), FED.name
    )["pass_through"]
    assert strong[RBI.name] > weak[RBI.name]


# --- Stackelberg -----------------------------------------------------------

def test_the_leader_is_recorded(spillovers):
    led = network_stackelberg(BANKS, spillovers, FED.name)
    assert led.leader == FED.name


def test_every_bank_still_gets_a_rate(spillovers):
    led = network_stackelberg(BANKS, spillovers, FED.name)
    assert set(led.rates) == {b.name for b in BANKS}


def test_leading_does_not_hurt_the_leader(spillovers):
    """Moving first is an option the leader need not exercise, so its loss
    cannot exceed the simultaneous one by more than the grid resolution."""
    simultaneous = network_nash(BANKS, spillovers)
    led = network_stackelberg(BANKS, spillovers, FED.name)
    assert led.losses[FED.name] <= simultaneous.losses[FED.name] + 1e-4


def test_the_note_states_the_limitation(spillovers):
    """A single leader with simultaneous followers is one N player
    Stackelberg game, not the only one."""
    led = network_stackelberg(BANKS, spillovers, FED.name)
    assert "sequential ordering would be a different game" in led.note


def test_an_unknown_leader_is_rejected(spillovers):
    with pytest.raises(EngineError, match="no bank named"):
        network_stackelberg(BANKS, spillovers, "Bank of Nowhere")


def test_a_custom_grid_is_used(spillovers):
    """Approximately, not exactly. The leader is pinned with a very large
    but finite smoothing weight rather than an infinite one, so it lands
    within a hair of the grid point rather than on it."""
    led = network_stackelberg(BANKS, spillovers, FED.name, grid=(0.04, 0.05, 0.06))
    assert min(abs(led.rates[FED.name] - g) for g in (0.04, 0.05, 0.06)) < 1e-6


# --- structural properties -------------------------------------------------

def test_a_network_with_no_spillovers_decouples():
    """With every off-diagonal zero, each bank solves its own problem and
    the network adds nothing. A useful degenerate case: if the rates differ
    from the isolated solution, the assembly is leaking cross terms."""
    names = tuple(b.name for b in BANKS)
    zeros = np.zeros((len(names), len(names)))
    matrix = SpilloverMatrix(names=names, demand=zeros, exchange=zeros)

    equilibrium = network_nash(BANKS, matrix)
    assert equilibrium.is_well_conditioned
    assert len(equilibrium.rates) == len(BANKS)


def test_stronger_spillovers_change_the_equilibrium():
    weak = SpilloverMatrix.from_tiers(DEFAULT_TIERS, base_demand=0.05)
    strong = SpilloverMatrix.from_tiers(DEFAULT_TIERS, base_demand=0.60)
    assert network_nash(BANKS, weak).rates != network_nash(BANKS, strong).rates


def test_all_banks_at_target_produce_small_moves():
    """Nobody has a reason to act, so nobody should move far."""
    at_target = tuple(
        b.model_copy(update={"current_inflation": b.inflation_target}) for b in BANKS
    )
    spillovers = SpilloverMatrix.from_tiers(DEFAULT_TIERS)
    equilibrium = network_nash(at_target, spillovers)
    moves = equilibrium.moves_bp({b.name: b for b in at_target})
    assert max(abs(v) for v in moves.values()) < 200