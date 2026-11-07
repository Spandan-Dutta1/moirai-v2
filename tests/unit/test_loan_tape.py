"""The loan tape: a lender's book, one row per loan, validated on entry (Stage 2).

Written before the module existed. A loan tape is the first input Moirai takes
from a lender rather than from a statistics agency, so it is refused loudly
when it contradicts itself, and every refusal names the loans involved.
"""

from typing import Any

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import Benchmark, LoanTape, Product, annuity_payment


def loan(**overrides: Any) -> dict[str, Any]:
    """A home loan of Rs 10 lakh at 9% over 20 years, repo-linked, quarterly reset."""
    base: dict[str, Any] = {
        "loan_id": "L1",
        "borrower_id": "B1",
        "product": "home",
        "benchmark": "eblr",
        "rate": 0.09,
        "spread": 0.0375,
        "outstanding": 1_000_000.0,
        "remaining_months": 240,
        "emi": 8_997.26,
        "reset_months": 3,
        "months_to_next_reset": 1,
        "borrower_income": 600_000.0,
        "borrower_age": 35,
        "days_past_due": 0,
    }
    return {**base, **overrides}


# ---- the annuity ---------------------------------------------------------------


def test_annuity_matches_the_standard_emi() -> None:
    """Rs 10 lakh at 9% over 240 months is Rs 8,997.26 a month."""
    assert annuity_payment(1_000_000.0, 0.09, 240) == pytest.approx(8_997.26, abs=0.01)


def test_annuity_at_a_zero_rate_is_straight_line() -> None:
    assert annuity_payment(120_000.0, 0.0, 12) == pytest.approx(10_000.0)


# ---- a valid tape --------------------------------------------------------------


def test_a_valid_tape_is_read() -> None:
    tape = LoanTape.from_records(
        [
            loan(),
            loan(loan_id="L2", borrower_id="B2", benchmark="fixed", reset_months=0,
                 months_to_next_reset=0, product="vehicle", outstanding=500_000.0,
                 rate=0.10, remaining_months=60, emi=10_623.52),
        ]
    )
    assert len(tape) == 2
    assert list(tape.is_floating) == [True, False]
    assert tape.product[1] == Product.VEHICLE
    assert tape.benchmark[0] == Benchmark.EBLR


def test_borrower_dsti_adds_every_loan_of_the_borrower() -> None:
    tape = LoanTape.from_records(
        [
            loan(),
            loan(loan_id="L2", product="personal", benchmark="mclr", reset_months=12,
                 months_to_next_reset=5, outstanding=200_000.0, rate=0.12,
                 remaining_months=48, emi=5_266.77),
        ]
    )
    expected = (8_997.26 + 5_266.77) * 12 / 600_000.0
    assert tape.borrower_dsti == pytest.approx(np.array([expected, expected]), rel=1e-9)


def test_a_loan_whose_emi_no_longer_covers_interest_is_accepted_and_flagged() -> None:
    """Negative amortisation: the 2022-23 tenure-extension problem itself."""
    stretched = loan(loan_id="L2", borrower_id="B2", emi=7_000.0, remaining_months=360)
    tape = LoanTape.from_records([loan(), stretched])
    assert list(tape.covers_interest) == [True, False]


# ---- refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"benchmark": "libor"}, "benchmark"),
        ({"product": "credit_card"}, "product"),
        ({"reset_months": 6}, "reset"),  # EBLR must reset at least every 3 months
        ({"benchmark": "fixed", "reset_months": 3}, "reset"),
        ({"months_to_next_reset": 3}, "reset"),  # must be inside the reset period
        ({"emi": 20_000.0}, "EMI"),  # inconsistent with the annuity
        ({"outstanding": -5.0}, "outstanding"),
        ({"rate": 0.75}, "rate"),
        ({"borrower_age": 15}, "age"),
        ({"borrower_income": 0.0}, "income"),
        ({"remaining_months": 0}, "remaining"),
        ({"days_past_due": -1}, "days past due"),
    ],
)
def test_a_contradictory_loan_is_refused_by_name(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(EngineError) as error:
        LoanTape.from_records([loan(), loan(loan_id="BAD", borrower_id="B9", **override)])
    message = str(error.value)
    assert fragment in message
    assert "BAD" in message


def test_duplicate_loan_ids_are_refused() -> None:
    with pytest.raises(EngineError, match="duplicate"):
        LoanTape.from_records([loan(), loan()])


def test_one_borrower_with_two_incomes_is_refused() -> None:
    with pytest.raises(EngineError, match="income"):
        LoanTape.from_records([loan(), loan(loan_id="L2", borrower_income=900_000.0)])


def test_an_empty_tape_is_refused() -> None:
    with pytest.raises(EngineError):
        LoanTape.from_records([])
