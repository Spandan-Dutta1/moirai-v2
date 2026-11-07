"""From a loan tape to a Population (Stage 2.4).

Written before the module existed. Observed fields come from the tape; the four
a tape cannot observe are imputed from the calibration and named as imputed.
"""

from typing import Any

import numpy as np
import pytest

from moirai.engine.credit.adapter import IMPUTED_FIELDS, to_population
from moirai.engine.credit.loan_tape import LoanTape, annuity_payment
from moirai.engine.economy.households import EmploymentStatus, PopulationParameters


def loan(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "loan_id": "L1",
        "borrower_id": "B1",
        "product": "home",
        "benchmark": "eblr",
        "rate": 0.09,
        "spread": 0.0375,
        "outstanding": 1_000_000.0,
        "remaining_months": 240,
        "emi": float(annuity_payment(1_000_000.0, 0.09, 240)),
        "reset_months": 3,
        "months_to_next_reset": 1,
        "borrower_income": 600_000.0,
        "borrower_age": 35,
        "days_past_due": 0,
    }
    return {**base, **overrides}


def fixed_vehicle(**overrides: Any) -> dict[str, Any]:
    return loan(
        loan_id="L2", product="vehicle", benchmark="fixed", reset_months=0,
        months_to_next_reset=0, outstanding=400_000.0, rate=0.10, remaining_months=60,
        emi=float(annuity_payment(400_000.0, 0.10, 60)), **overrides,
    )


def test_one_household_per_borrower_with_its_debt_summed() -> None:
    result = to_population(LoanTape.from_records([loan(), fixed_vehicle()]))
    assert len(result.population) == 1
    assert result.borrower_id == ("B1",)
    assert result.population.debt[0] == pytest.approx(1_400_000.0)


def test_income_and_age_are_the_tapes() -> None:
    population = to_population(LoanTape.from_records([loan()])).population
    assert population.income[0] == pytest.approx(600_000.0)
    assert population.age[0] == 35


def test_floating_when_most_of_the_debt_by_amount_is_floating() -> None:
    result = to_population(LoanTape.from_records([loan(), fixed_vehicle()]))
    assert result.floating_share[0] == pytest.approx(1_000_000.0 / 1_400_000.0)
    assert bool(result.population.debt_is_floating[0])


def test_mostly_fixed_debt_is_not_floating() -> None:
    small_floating = loan(outstanding=100_000.0, emi=float(annuity_payment(100_000.0, 0.09, 240)))
    result = to_population(LoanTape.from_records([small_floating, fixed_vehicle()]))
    assert not bool(result.population.debt_is_floating[0])


def test_employment_follows_the_retirement_age() -> None:
    records = [loan(), loan(loan_id="L9", borrower_id="B9", borrower_age=62)]
    population = to_population(LoanTape.from_records(records)).population
    assert population.employment[0] == EmploymentStatus.EMPLOYED
    assert population.employment[1] == EmploymentStatus.RETIRED


def test_the_imputed_fields_are_named() -> None:
    result = to_population(LoanTape.from_records([loan()]))
    assert result.imputed == IMPUTED_FIELDS
    assert set(result.imputed) == {"employment", "wealth", "expected_inflation", "risk_aversion"}


def many(n: int = 2_000) -> LoanTape:
    rng = np.random.default_rng(3)
    incomes = np.exp(rng.normal(12.8, 0.6, n))
    return LoanTape.from_records(
        [
            loan(loan_id=f"L{i}", borrower_id=f"B{i}", borrower_income=float(incomes[i]))
            for i in range(n)
        ]
    )


def test_imputed_wealth_rises_with_observed_income() -> None:
    population = to_population(many()).population
    correlation = np.corrcoef(np.log(population.income), np.log(population.wealth))[0, 1]
    assert correlation > 0.3  # the calibrated income-wealth correlation is 0.5


def test_the_same_seed_gives_the_same_population() -> None:
    tape = many(200)
    a = to_population(tape, PopulationParameters(seed=11)).population
    b = to_population(tape, PopulationParameters(seed=11)).population
    c = to_population(tape, PopulationParameters(seed=12)).population
    assert np.array_equal(a.wealth, b.wealth)
    assert not np.array_equal(a.wealth, c.wealth)


def test_the_book_reports_its_outstanding_weighted_remaining_term() -> None:
    result = to_population(LoanTape.from_records([loan(), fixed_vehicle()]))
    expected = (1_000_000.0 * 240 + 400_000.0 * 60) / 1_400_000.0 / 12.0
    assert result.remaining_years == pytest.approx(expected)
