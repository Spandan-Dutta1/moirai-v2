"""Tests for the chain on the dynamic game (ADRs 023, 026, 027).

Households face the origin bank's own rate path from the dynamic game,
quarter by quarter, instead of the US impulse response rescaled to the
static game's move. These check the conversion to the monthly household
clock and that a scenario runs through to comparable results.
"""

from __future__ import annotations

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.dynamic_chain import (
    calibrated_inputs,
    household_path,
    run_dynamic_scenario,
)
from moirai.engine.economy.households import (
    LENDER_DEBT_TO_INCOME_CAP,
    PopulationParameters,
    generate_population,
)
from moirai.engine.economy.shock_path import MacroVariable
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.dynamic_game import dynamic_nash
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import HEADLINE_SCENARIO

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)


@pytest.fixture(scope="module")
def setup():
    return calibrated_inputs(BANKS, SpilloverMatrix.from_literature(DEFAULT_TIERS))


@pytest.fixture(scope="module")
def population():
    return generate_population(
        PopulationParameters(
            n_households=20_000, seed=1, max_debt_to_income=LENDER_DEBT_TO_INCOME_CAP
        )
    )


def _paths(setup, months=36):
    banks, matrix, params = setup
    conditioned = HEADLINE_SCENARIO.apply_to(banks)
    origin = next(b for b in conditioned if b.name == RBI.name)
    quarters = months // 3 + 2
    shocked = dynamic_nash(conditioned, matrix, parameters=params, horizon=quarters)
    reference = dynamic_nash(banks, matrix, parameters=params, horizon=quarters)
    return origin, shocked, reference


def test_rates_are_held_for_each_quarters_three_months(setup):
    origin, shocked, reference = _paths(setup)
    path, caused = household_path(origin, shocked, reference, months=36)
    rate = path.get(MacroVariable.POLICY_RATE) - origin.current_rate
    for q in range(12):
        block = rate[3 * q : 3 * q + 3]
        assert np.allclose(block, caused[q] / 10_000)


def test_the_path_starts_from_the_origin_economys_baselines(setup):
    origin, shocked, reference = _paths(setup)
    path, _ = household_path(origin, shocked, reference, months=36)
    assert path.baselines[MacroVariable.POLICY_RATE] == origin.current_rate
    assert path.baselines[MacroVariable.INCOME_GROWTH] == origin.current_income_growth


def test_a_short_solution_is_refused(setup):
    origin, shocked, reference = _paths(setup, months=12)
    with pytest.raises(EngineError):
        household_path(origin, shocked, reference, months=36)


def test_the_headline_runs_through_to_households(setup, population):
    banks, matrix, params = setup
    result = run_dynamic_scenario(
        HEADLINE_SCENARIO,
        banks,
        matrix,
        population=population,
        banking_system=INDIAN_BANKING_SYSTEM,
        parameters=params,
    )
    assert result.caused_moves_bp[0] > 0
    assert result.borrower_change < result.saver_change
    assert result.households_never_consuming == 0
    assert result.path.horizon <= HEADLINE_SCENARIO.horizon_cap.periods
