"""Tests for the dynamic game between central banks.

Four kinds of check: the transmission has the lags it claims; in the long
run it reproduces the static game's effects; the solver's rules are best
responses to each other; and on the declared banks it behaves as the
economics says it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.dynamic_game import (
    DynamicParameters,
    discounted_loss,
    dynamic_nash,
    transition,
)
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import IMPORTED_TIGHTENING

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)


@pytest.fixture(scope="module")
def matrix() -> SpilloverMatrix:
    return SpilloverMatrix.from_literature(DEFAULT_TIERS)


@pytest.fixture(scope="module")
def ordered(matrix):
    return tuple(next(b for b in BANKS if b.name == n) for n in matrix.names)


# --- transmission -------------------------------------------------------------


def test_a_rate_move_reaches_output_after_a_quarter_and_inflation_after_two(matrix, ordered):
    """The lags the module exists for. The own inflation channel runs
    through output, so a bank's own move cannot touch its inflation in the
    first quarter except through the exchange rate."""
    params = DynamicParameters()
    A, B = transition(ordered, matrix, params)
    n = len(ordered)
    i = matrix.index_of(FED.name)
    x1 = B[i][:, 0] * 0.01  # a 1pp move, state one quarter later
    assert x1[n + i] < 0  # output has fallen
    # the Fed's own inflation moves in quarter one only through exchange
    exchange_only = (
        -sum(matrix.exchange[i, j] for j in range(n) if j != i)
        * (1 - params.inflation_persistence)
        * 0.01
    )
    assert x1[i] == pytest.approx(exchange_only)
    x2 = A @ x1
    assert x2[i] < x1[i]  # inflation falls further once output has fallen


def test_a_permanent_move_has_the_static_long_run_effects(matrix, ordered):
    """The static game's own-rate and exchange effects are this game's long
    run: hold one bank's move at 1pp forever and read the steady state."""
    params = DynamicParameters()
    A, B = transition(ordered, matrix, params)
    n = len(ordered)
    i = matrix.index_of(RBI.name)
    steady = np.linalg.solve(np.eye(2 * n) - A, B[i][:, 0] * 0.01)
    assert steady[n + i] == pytest.approx(-matrix.own_output_effect * 0.01)
    others_exchange = sum(matrix.exchange[i, j] for j in range(n) if j != i)
    assert steady[i] == pytest.approx(-(matrix.own_inflation_effect + others_exchange) * 0.01)


def test_a_foreign_move_reaches_home_output_by_the_static_demand_effect(matrix, ordered):
    params = DynamicParameters()
    A, B = transition(ordered, matrix, params)
    n = len(ordered)
    fed, rbi = matrix.index_of(FED.name), matrix.index_of(RBI.name)
    steady = np.linalg.solve(np.eye(2 * n) - A, B[fed][:, 0] * 0.01)
    assert steady[n + rbi] == pytest.approx(-matrix.demand[rbi, fed] * 0.01)


# --- the solver ---------------------------------------------------------------


def test_the_equilibrium_is_stable_and_converges(matrix):
    result = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix)
    assert result.iterations < 200
    assert np.all(np.isfinite(np.asarray(result.moves)))


@pytest.mark.parametrize("bank", [FED, RBI, ECB])
def test_each_rule_is_a_best_response(matrix, bank):
    """Perturbing one bank's rule while the others keep theirs cannot lower
    that bank's discounted loss: the defining property of feedback Nash."""
    banks = IMPORTED_TIGHTENING.apply_to(BANKS)
    result = dynamic_nash(banks, matrix)
    rules = np.asarray(result.rules)
    i = matrix.index_of(bank.name)
    base = discounted_loss(i, banks, matrix, rules)
    rng = np.random.default_rng(0)
    for _ in range(20):
        perturbed = rules.copy()
        perturbed[i] += rng.normal(0, 0.05, rules.shape[1])
        assert discounted_loss(i, banks, matrix, perturbed) >= base - 1e-12


def test_the_answer_does_not_depend_on_the_order_the_banks_are_given(matrix):
    shocked = IMPORTED_TIGHTENING.apply_to(BANKS)
    one = dynamic_nash(shocked, matrix)
    two = dynamic_nash(tuple(reversed(shocked)), matrix)
    assert np.allclose(np.asarray(one.moves), np.asarray(two.moves), atol=1e-10)


def test_banks_that_do_not_match_the_matrix_are_refused(matrix):
    with pytest.raises(EngineError):
        dynamic_nash(BANKS[:4], matrix)


# --- behaviour on the declared banks ------------------------------------------


def test_with_no_gaps_nobody_moves(matrix):
    calm = tuple(
        b.model_copy(update={"current_inflation": b.inflation_target, "current_output_gap": 0.0})
        for b in BANKS
    )
    result = dynamic_nash(calm, matrix)
    assert np.allclose(np.asarray(result.moves), 0.0, atol=1e-14)


def test_a_us_inflation_shock_makes_the_fed_tighten_and_then_unwind(matrix):
    shocked = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix)
    reference = dynamic_nash(BANKS, matrix)
    caused = (np.asarray(shocked.moves) - np.asarray(reference.moves))[:, matrix.index_of(FED.name)]
    assert caused[0] > 0
    assert abs(caused[8]) < caused[0] / 4


def test_the_rbi_follows_the_fed_first(matrix):
    """The pattern ADR 022 measured: the RBI's policy rate rises in the
    months after a Fed tightening surprise."""
    shocked = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix)
    reference = dynamic_nash(BANKS, matrix)
    caused = (np.asarray(shocked.moves) - np.asarray(reference.moves))[:, matrix.index_of(RBI.name)]
    assert caused[0] > 0


def test_the_persistence_parameters_are_validated():
    with pytest.raises(ValueError):
        DynamicParameters(output_persistence=1.0)
    with pytest.raises(ValueError):
        DynamicParameters(inflation_persistence=-0.1)


# --- policy inertia (ADR 026) -------------------------------------------------


def test_without_inertia_the_state_and_rules_are_unchanged(matrix):
    """Zero inertia keeps ADR 023's two-block state, so its results
    reproduce exactly rather than approximately."""
    result = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix)
    assert np.asarray(result.rules).shape == (5, 10)


def test_with_inertia_last_quarters_moves_join_the_state(matrix):
    params = DynamicParameters(inertia_weight=1.0)
    result = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix, parameters=params)
    assert np.asarray(result.rules).shape == (5, 15)


@pytest.mark.parametrize("bank", [FED, RBI])
def test_rules_are_best_responses_with_inertia(matrix, bank):
    params = DynamicParameters(inertia_weight=2.0)
    banks = IMPORTED_TIGHTENING.apply_to(BANKS)
    result = dynamic_nash(banks, matrix, parameters=params)
    rules = np.asarray(result.rules)
    i = matrix.index_of(bank.name)
    base = discounted_loss(i, banks, matrix, rules, parameters=params)
    rng = np.random.default_rng(1)
    for _ in range(20):
        perturbed = rules.copy()
        perturbed[i] += rng.normal(0, 0.05, rules.shape[1])
        assert discounted_loss(i, banks, matrix, perturbed, parameters=params) >= base - 1e-12


def test_the_calibrated_weight_gives_the_fed_its_estimated_smoothing(matrix):
    from moirai.engine.financial.dynamic_game import (
        CALIBRATED_INERTIA_WEIGHT,
        FED_SMOOTHING_TARGET,
        smoothing_coefficient,
    )

    params = DynamicParameters(inertia_weight=CALIBRATED_INERTIA_WEIGHT)
    result = dynamic_nash(BANKS, matrix, parameters=params)
    assert smoothing_coefficient(result, FED.name) == pytest.approx(FED_SMOOTHING_TARGET, abs=0.005)


def test_inertia_makes_the_rbi_build_up_rather_than_jump(matrix):
    """The shape ADR 022 measured: the RBI's caused move rises for the
    first quarters instead of being largest at once."""
    from moirai.engine.financial.dynamic_game import CALIBRATED_INERTIA_WEIGHT

    params = DynamicParameters(inertia_weight=CALIBRATED_INERTIA_WEIGHT)
    shocked = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix, parameters=params)
    reference = dynamic_nash(BANKS, matrix, parameters=params)
    caused = (np.asarray(shocked.moves) - np.asarray(reference.moves))[:, matrix.index_of(RBI.name)]
    assert caused[0] > 0
    assert caused[2] > caused[0]
    assert caused[4] > 0


def _roughness(result) -> float:
    moves = np.vstack([np.zeros(5), np.asarray(result.moves)])
    return float(np.sum(np.diff(moves, axis=0) ** 2))


def test_inertia_smooths_every_banks_path(matrix):
    plain = dynamic_nash(IMPORTED_TIGHTENING.apply_to(BANKS), matrix)
    smooth = dynamic_nash(
        IMPORTED_TIGHTENING.apply_to(BANKS),
        matrix,
        parameters=DynamicParameters(inertia_weight=2.0),
    )
    assert _roughness(smooth) < _roughness(plain)


# --- the calibrated RBI (ADR 027) ---------------------------------------------


def _calibrated():
    from moirai.engine.dynamic_chain import calibrated_inputs

    return calibrated_inputs(BANKS, SpilloverMatrix.from_literature(DEFAULT_TIERS))


@pytest.mark.parametrize("bank", [FED, RBI])
def test_rules_are_best_responses_in_the_calibrated_game(bank):
    banks, matrix, params = _calibrated()
    shocked = IMPORTED_TIGHTENING.apply_to(banks)
    result = dynamic_nash(shocked, matrix, parameters=params)
    rules = np.asarray(result.rules)
    i = matrix.index_of(bank.name)
    base = discounted_loss(i, shocked, matrix, rules, parameters=params)
    rng = np.random.default_rng(2)
    for _ in range(20):
        perturbed = rules.copy()
        perturbed[i] += rng.normal(0, 0.05, rules.shape[1])
        assert discounted_loss(i, shocked, matrix, perturbed, parameters=params) >= base - 1e-12


def test_the_calibrated_rbi_follows_the_fed_by_the_measured_floor():
    from moirai.engine.financial.dynamic_game import RBI_FOLLOWING_FLOOR

    banks, matrix, params = _calibrated()
    shocked = dynamic_nash(IMPORTED_TIGHTENING.apply_to(banks), matrix, parameters=params)
    reference = dynamic_nash(banks, matrix, parameters=params)
    caused = np.asarray(shocked.moves) - np.asarray(reference.moves)
    fed = caused[:, matrix.index_of(FED.name)].max()
    rbi = caused[:, matrix.index_of(RBI.name)].max()
    assert rbi / fed == pytest.approx(RBI_FOLLOWING_FLOOR, abs=0.01)


def test_the_calibrated_rbi_keeps_the_measured_shape():
    """Calibrating the size must not undo ADR 026's timing: the RBI still
    builds up over the first quarters and stays up."""
    banks, matrix, params = _calibrated()
    shocked = dynamic_nash(IMPORTED_TIGHTENING.apply_to(banks), matrix, parameters=params)
    reference = dynamic_nash(banks, matrix, parameters=params)
    path = (np.asarray(shocked.moves) - np.asarray(reference.moves))[:, matrix.index_of(RBI.name)]
    measured = np.array([1.33, 3.06, 4.99, 4.97, 5.06])
    model = path[:5] / np.abs(path[:5]).max()
    assert ((model - measured / measured.max()) ** 2).sum() < 0.05


def test_calibration_leaves_the_validated_rbi_default_alone():
    from moirai.engine.financial.dynamic_game import CALIBRATED_RBI_EXTERNAL_WEIGHT

    banks, _, _ = _calibrated()
    calibrated_rbi = next(b for b in banks if b.name == RBI.name)
    assert calibrated_rbi.external_weight == CALIBRATED_RBI_EXTERNAL_WEIGHT
    assert RBI.external_weight == 0.40
