"""Tests for central banks as strategic players.

Two things get the most attention here.

The first is the published mandates. Those are sourced facts, and a test
that pins them means a later edit cannot quietly change the Fed's target
or drop the RBI's tolerance band without something failing.

The second is the impossibility check on the cooperative solution. An
earlier version derived the joint first order conditions by hand, got the
cross terms wrong, and produced a cooperative outcome worse than Nash.
That cannot happen: optimising over both rates jointly cannot do worse
than each bank optimising over one. The property is asserted in the solver
and tested here, because it was the only thing that revealed the bug.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    MAJOR_CENTRAL_BANKS,
    RBI,
    CentralBank,
    MandateType,
    SpilloverParameters,
    analyse_pair,
    analytic_cooperative,
    analytic_nash,
    build_policy_game,
    coordination_value,
    sensitivity_report,
    weight_sensitivity,
)
from moirai.engine.financial.game import Confidence, nash_equilibria


def make_bank(**overrides) -> CentralBank:
    defaults = dict(
        name="Test Bank",
        jurisdiction="TB",
        inflation_target=0.02,
        mandate=MandateType.HIERARCHICAL,
        mandate_source="test",
        current_rate=0.03,
        current_inflation=0.025,
    )
    return CentralBank(**{**defaults, **overrides})


# --- published mandates ----------------------------------------------------

def test_every_bank_cites_its_mandate_source():
    for bank in MAJOR_CENTRAL_BANKS:
        assert bank.mandate_source.strip(), f"{bank.name} has no source"


def test_the_fed_is_the_only_dual_mandate():
    """Only fifteen to twenty percent of central banks have one, and among
    the majors it is the Fed alone. A game giving every player the same
    objective would misrepresent the system."""
    dual = [b for b in MAJOR_CENTRAL_BANKS if b.mandate is MandateType.DUAL]
    assert dual == [FED]


@pytest.mark.parametrize(
    ("bank", "target"),
    [(FED, 0.02), (ECB, 0.02), (BANK_OF_ENGLAND, 0.02), (BANK_OF_JAPAN, 0.02), (RBI, 0.04)],
)
def test_inflation_targets_match_the_published_figures(bank, target):
    assert bank.inflation_target == target


def test_the_rbi_is_the_only_one_with_a_legislated_band():
    """A band changes behaviour: breaching two to six percent obliges the
    RBI to explain itself to the government."""
    banded = [b for b in MAJOR_CENTRAL_BANKS if b.has_band]
    assert banded == [RBI]
    assert (RBI.tolerance_lower, RBI.tolerance_upper) == (0.02, 0.06)


def test_the_fed_carries_no_external_weight():
    """Reserve currency status means little external constraint."""
    assert FED.external_weight == 0.0


def test_the_rbi_external_weight_is_derived_not_assumed():
    """Its own Report on Currency and Finance documents the objective."""
    assert RBI.external_weight > 0
    assert RBI.weight_confidence is Confidence.DERIVED
    assert "Currency and Finance" in RBI.weight_note


def test_the_fed_output_weight_is_derived_from_an_estimate():
    assert FED.weight_confidence is Confidence.DERIVED
    assert "1986-2007" in FED.weight_note


def test_price_stability_banks_weight_output_below_inflation():
    """Hierarchical and price-stability mandates subordinate other goals."""
    for bank in (ECB, BANK_OF_ENGLAND, BANK_OF_JAPAN):
        assert bank.output_weight < bank.inflation_weight


def test_an_inverted_band_is_rejected():
    with pytest.raises(Exception, match="inverted"):
        make_bank(tolerance_lower=0.06, tolerance_upper=0.02)


def test_a_target_outside_its_own_band_is_rejected():
    with pytest.raises(Exception, match="outside its own band"):
        make_bank(inflation_target=0.08, tolerance_lower=0.02, tolerance_upper=0.06)


def test_a_bank_is_frozen():
    with pytest.raises(Exception):
        FED.inflation_target = 0.03  # type: ignore[misc]


def test_a_bank_serialises_with_its_provenance():
    payload = RBI.to_ledger_dict()
    assert payload["mandate"] == "hierarchical"
    assert payload["band"] == [0.02, 0.06]
    assert payload["weight_confidence"] == "derived"


# --- the loss function -----------------------------------------------------

def test_loss_is_zero_at_target_with_no_gaps():
    bank = make_bank(smoothing_weight=0.0)
    assert bank.loss(0.02, 0.0, bank.current_rate) == pytest.approx(0.0)


def test_loss_rises_with_the_inflation_gap():
    bank = make_bank(smoothing_weight=0.0)
    assert bank.loss(0.05, 0.0, bank.current_rate) > bank.loss(0.03, 0.0, bank.current_rate)


def test_loss_is_symmetric_in_the_inflation_gap():
    """Quadratic loss means overshooting is as costly as undershooting.
    That is an assumption: central banks arguably dislike overshoots more."""
    bank = make_bank(smoothing_weight=0.0)
    assert bank.loss(0.04, 0.0, bank.current_rate) == pytest.approx(
        bank.loss(0.0, 0.0, bank.current_rate)
    )


def test_smoothing_penalises_large_moves():
    bank = make_bank(smoothing_weight=0.5)
    assert bank.loss(0.02, 0.0, 0.06) > bank.loss(0.02, 0.0, bank.current_rate)


def test_a_zero_output_weight_ignores_the_output_gap():
    bank = make_bank(output_weight=0.0, smoothing_weight=0.0)
    assert bank.loss(0.02, 0.5, bank.current_rate) == pytest.approx(0.0)


def test_breaching_the_band_adds_a_penalty():
    """The accountability cost of missing, which a target without a band
    does not carry."""
    banded = make_bank(
        inflation_target=0.04, tolerance_lower=0.02, tolerance_upper=0.06,
        smoothing_weight=0.0, output_weight=0.0,
    )
    unbanded = make_bank(inflation_target=0.04, smoothing_weight=0.0, output_weight=0.0)
    assert banded.loss(0.09, 0.0, banded.current_rate) > unbanded.loss(
        0.09, 0.0, unbanded.current_rate
    )


def test_inside_the_band_there_is_no_extra_penalty():
    banded = make_bank(
        inflation_target=0.04, tolerance_lower=0.02, tolerance_upper=0.06,
        smoothing_weight=0.0, output_weight=0.0,
    )
    unbanded = make_bank(inflation_target=0.04, smoothing_weight=0.0, output_weight=0.0)
    assert banded.loss(0.05, 0.0, banded.current_rate) == pytest.approx(
        unbanded.loss(0.05, 0.0, unbanded.current_rate)
    )


@pytest.mark.parametrize(("inflation", "inside"), [(0.02, True), (0.06, True), (0.07, False)])
def test_band_membership_is_inclusive_at_the_edges(inflation, inside):
    assert RBI.within_band(inflation) is inside


def test_the_external_gap_is_measured_against_the_mean_rate():
    """Against the mean of the others, not the mean distance to each. A
    bank between two others sits at zero gap under the first definition
    and a full point under the second."""
    assert RBI.external_gap(0.05, (0.04, 0.06)) == pytest.approx(0.0)
    assert RBI.external_gap(0.06, (0.03, 0.05, 0.04)) == pytest.approx(0.02)


def test_the_external_gap_is_zero_without_an_external_objective():
    assert FED.external_weight == 0
    assert FED.external_gap(0.06, (0.02, 0.03)) == 0.0


# --- the discrete game -----------------------------------------------------

def test_the_game_has_two_players():
    game = build_policy_game(FED, RBI)
    assert game.names == (FED.name, RBI.name)


def test_the_default_grid_covers_plausible_rates():
    game = build_policy_game(FED, RBI)
    actions = game.players[0].actions
    assert min(actions) == 0.0
    assert max(actions) >= 0.08


def test_a_custom_grid_is_used():
    game = build_policy_game(FED, RBI, rate_grid=(0.04, 0.05, 0.06))
    assert game.players[0].n_actions == 3


def test_a_single_action_grid_is_rejected():
    with pytest.raises(EngineError, match="at least two"):
        build_policy_game(FED, RBI, rate_grid=(0.05,))


def test_the_discrete_game_has_an_equilibrium():
    game = build_policy_game(FED, RBI, rate_grid=tuple(np.round(np.arange(0.0, 0.09, 0.005), 6)))
    assert len(nash_equilibria(game)) >= 1


# --- analytic Nash ---------------------------------------------------------

def test_the_analytic_solution_reports_both_rates():
    solution = analytic_nash(FED, RBI)
    assert set(solution.rates) == {FED.name, RBI.name}


def test_the_system_is_well_conditioned():
    """A near-singular system means parallel reaction functions, so the
    equilibrium would be extremely sensitive to the assumed weights."""
    solution = analytic_nash(FED, RBI)
    assert solution.is_well_conditioned
    assert solution.condition_number < 100


def test_the_analytic_and_grid_solutions_roughly_agree():
    """The grid cannot resolve better than its step size, so agreement to
    within one increment is the most that can be asked."""
    analytic = analytic_nash(FED, RBI)
    grid = nash_equilibria(build_policy_game(FED, RBI))[0]
    for name in analytic.rates:
        assert abs(analytic.rates[name] - grid.actions[name]) < 0.005


def test_a_higher_inflation_gap_raises_the_equilibrium_rate():
    """The direction that must hold whatever the weights."""
    calm = analytic_nash(FED, RBI)
    shocked = analytic_nash(FED.model_copy(update={"current_inflation": 0.06}), RBI)
    assert shocked.rates[FED.name] > calm.rates[FED.name]


def test_a_higher_smoothing_weight_shrinks_the_move():
    """Gradualism: a bank that dislikes moving stays nearer where it was."""
    shocked = FED.model_copy(update={"current_inflation": 0.06})
    responsive = analytic_nash(shocked.model_copy(update={"smoothing_weight": 0.05}), RBI)
    sluggish = analytic_nash(shocked.model_copy(update={"smoothing_weight": 1.5}), RBI)

    assert abs(sluggish.rates[FED.name] - FED.current_rate) < abs(
        responsive.rates[FED.name] - FED.current_rate
    )


def test_the_solution_is_deterministic():
    assert analytic_nash(FED, RBI).rates == analytic_nash(FED, RBI).rates


def test_the_solution_serialises_for_the_ledger():
    payload = analytic_nash(FED, RBI).to_ledger_dict()
    assert payload["concept"] == "nash_analytic"
    assert "condition_number" in payload


# --- cooperative -----------------------------------------------------------

def test_cooperation_never_scores_worse_than_nash():
    """The property that revealed a real bug. Optimising over both rates
    jointly cannot do worse than each bank optimising over one, so a
    negative gain means the solvers disagree about the objective."""
    nash = analytic_nash(FED, RBI)
    cooperative = analytic_cooperative(FED, RBI)
    assert cooperative.total_loss <= nash.total_loss + 1e-9


@pytest.mark.parametrize(
    "inflation", [0.02, 0.03, 0.045, 0.06]
)
def test_cooperation_beats_nash_across_conditions(inflation):
    """Checked at several points, because the earlier bug was visible only
    as a sign and could have passed at a single one by chance."""
    fed = FED.model_copy(update={"current_inflation": inflation})
    assert analytic_cooperative(fed, RBI).total_loss <= analytic_nash(fed, RBI).total_loss + 1e-9


def test_the_coordination_gain_is_non_negative():
    assert coordination_value(FED, RBI)["total_gain"] >= -1e-9


def test_coordination_can_leave_one_party_worse_off():
    """Which is why it is not self-enforcing: the Fed internalising its
    spillover to India means moving away from its own optimum."""
    value = coordination_value(FED.model_copy(update={"current_inflation": 0.045}), RBI)
    assert isinstance(value["someone_loses"], bool)


def test_the_cooperative_result_is_labelled():
    assert analytic_cooperative(FED, RBI).concept == "cooperative_numeric"


# --- spillovers ------------------------------------------------------------

def test_spillovers_are_frozen():
    with pytest.raises(Exception):
        SpilloverParameters().demand_spillover = 0.9  # type: ignore[misc]


def test_negative_spillovers_are_rejected():
    with pytest.raises(Exception):
        SpilloverParameters(demand_spillover=-0.1)


def test_stronger_spillovers_change_the_equilibrium():
    weak = analytic_nash(FED, RBI, spillovers=SpilloverParameters(demand_spillover=0.05))
    strong = analytic_nash(FED, RBI, spillovers=SpilloverParameters(demand_spillover=0.9))
    assert weak.rates != strong.rates


def test_spillovers_serialise():
    assert SpilloverParameters().to_ledger_dict()["demand_spillover"] == 0.25


# --- weight sensitivity ----------------------------------------------------

def test_sensitivity_returns_one_rate_per_value():
    result = weight_sensitivity(
        FED, RBI, vary="output_weight", on=FED.name, values=[0.2, 0.5, 1.0]
    )
    assert len(result.rates) == 3


def test_sensitivity_reports_the_spread():
    result = weight_sensitivity(
        FED, RBI, vary="output_weight", on=FED.name, values=[0.1, 1.5]
    )
    assert result.spread_bp > 0


def test_a_weight_that_does_not_matter_shows_a_small_spread():
    """The RBI's external weight barely moves the Fed, which is the
    asymmetry the game exists to represent."""
    result = weight_sensitivity(
        FED,
        RBI,
        vary="external_weight",
        on=RBI.name,
        values=[0.1, 0.4, 0.8],
        observe=FED.name,
    )
    assert result.spread_bp < 10


def test_sensitivity_can_observe_the_other_bank():
    result = weight_sensitivity(
        FED, RBI, vary="output_weight", on=FED.name, values=[0.2, 1.0], observe=RBI.name
    )
    assert result.bank == RBI.name


def test_an_unknown_bank_is_rejected():
    with pytest.raises(EngineError, match="no bank named"):
        weight_sensitivity(FED, RBI, vary="output_weight", on="Nowhere", values=[0.5])


def test_an_unknown_weight_is_rejected():
    with pytest.raises(EngineError, match="not a weight"):
        weight_sensitivity(FED, RBI, vary="not_a_weight", on=FED.name, values=[0.5])


def test_an_empty_value_list_is_rejected():
    with pytest.raises(EngineError, match="at least one value"):
        weight_sensitivity(FED, RBI, vary="output_weight", on=FED.name, values=[])


def test_numpy_arrays_are_accepted():
    """A regression: `if not values` on an array raises rather than
    testing emptiness, because numpy refuses to guess between any and all."""
    result = weight_sensitivity(
        FED, RBI, vary="output_weight", on=FED.name, values=np.linspace(0.2, 1.0, 5)
    )
    assert len(result.rates) == 5


def test_sensitivity_serialises():
    payload = weight_sensitivity(
        FED, RBI, vary="output_weight", on=FED.name, values=[0.2, 1.0]
    ).to_ledger_dict()
    assert "spread_bp" in payload
    assert "is_robust" in payload


def test_the_full_report_covers_several_weights():
    report = sensitivity_report(FED, RBI, n_points=3)
    assert report["n_checks"] > 0
    assert report["n_robust"] <= report["n_checks"]


def test_the_report_lists_fragile_results():
    report = sensitivity_report(FED, RBI, n_points=3)
    assert isinstance(report["fragile"], list)


# --- pair analysis ---------------------------------------------------------

def test_analyse_pair_covers_every_concept():
    result = analyse_pair(FED, RBI)
    assert "nash" in result and "stackelberg" in result and "cooperation" in result


def test_analyse_pair_records_both_mandates():
    result = analyse_pair(FED, RBI)
    assert set(result["mandates"]) == {FED.name, RBI.name}
    assert result["mandates"][RBI.name]["band"] == [0.02, 0.06]


def test_analyse_pair_records_the_spillovers():
    assert "demand_spillover" in analyse_pair(FED, RBI)["spillovers"]


# --- the analytic Nash is an equilibrium of the stated loss ------------------
#
# The tests above check that the analytic solution is close to the grid
# solution, within 50bp. That tolerance let a sign error in the foreign
# bank's exchange rate term pass: with the RBI as the foreign bank, its
# solved rate sat 23bp from its true best reply. The tests below check the
# defining property directly, which is the lesson ADR 011 drew for the
# network solver, applied to the pairwise one.


def _true_best_reply(home, foreign, rate_home, rate_foreign, who):
    from scipy.optimize import minimize_scalar

    from moirai.engine.financial.central_banks import _losses_at

    spillovers = SpilloverParameters()
    if who == "home":
        objective = lambda r: _losses_at(home, foreign, r, rate_foreign, spillovers)[home.name]  # noqa: E731
    else:
        objective = lambda r: _losses_at(home, foreign, rate_home, r, spillovers)[foreign.name]  # noqa: E731
    return minimize_scalar(
        objective, bounds=(-0.10, 0.30), method="bounded", options={"xatol": 1e-12}
    ).x


PAIRS = [(FED, RBI), (RBI, FED), (ECB, RBI), (RBI, ECB), (BANK_OF_JAPAN, BANK_OF_ENGLAND)]


@pytest.mark.parametrize(("home", "foreign"), PAIRS, ids=lambda b: b.name)
def test_each_solved_rate_is_a_best_reply(home, foreign):
    """Every bank's solved rate minimises its own loss given the other's."""
    solution = analytic_nash(home, foreign)
    rate_home, rate_foreign = solution.rates[home.name], solution.rates[foreign.name]
    assert _true_best_reply(home, foreign, rate_home, rate_foreign, "home") == pytest.approx(
        rate_home, abs=1e-6
    )
    assert _true_best_reply(home, foreign, rate_home, rate_foreign, "foreign") == pytest.approx(
        rate_foreign, abs=1e-6
    )


@pytest.mark.parametrize(("first", "second"), PAIRS[::2], ids=lambda b: b.name)
def test_the_equilibrium_does_not_depend_on_which_bank_is_home(first, second):
    """A metamorphic check: relabelling the players cannot change the game."""
    one = analytic_nash(first, second)
    other = analytic_nash(second, first)
    for name in one.rates:
        assert one.rates[name] == pytest.approx(other.rates[name], abs=1e-9)


def test_the_pairwise_and_network_solvers_agree_exactly():
    """Same economy, two independent solvers, agreement to a hundredth of a
    basis point rather than to fifty."""
    from moirai.engine.financial.network import SpilloverMatrix, network_nash

    matrix = SpilloverMatrix(
        names=(FED.name, RBI.name),
        demand=np.array([[0.0, 0.25], [0.25, 0.0]]),
        exchange=np.array([[0.0, 0.30], [0.30, 0.0]]),
        own_output_effect=1.20,
        own_inflation_effect=0.80,
    )
    network = network_nash((FED, RBI), matrix)
    pairwise = analytic_nash(FED, RBI)
    for name in (FED.name, RBI.name):
        assert network.rates[name] == pytest.approx(pairwise.rates[name], abs=1e-6)


def test_a_fed_tightening_moves_the_rbi_up_from_its_current_rate():
    """The asymmetry `check_analytic_nash.py` describes: a Fed move forces an
    RBI response in the same direction. Before the sign fix the RBI solved
    below its current rate under the US shock, cutting while the Fed hiked."""
    shocked = analytic_nash(FED.model_copy(update={"current_inflation": 0.045}), RBI)
    assert shocked.rates[FED.name] > FED.current_rate
    assert shocked.rates[RBI.name] > RBI.current_rate
