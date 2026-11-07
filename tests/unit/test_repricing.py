"""The EMI reset engine (Stage 2.3).

Written before the module existed, and rewritten before the module changed when
the defaults were sourced. Each test pins one mechanism with numbers that can be
checked by hand: when a loan reprices, by how much, what the reset does to the
EMI or the tenure, and when the flags are raised.
"""

from typing import Any

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import LoanTape, annuity_payment
from moirai.engine.credit.repricing import RepricingParameters, ResetPolicy, reprice


def loan(**overrides: Any) -> dict[str, Any]:
    """Rs 10 lakh at 9% over 20 years, EBLR, quarterly reset, next reset in month 2.

    The borrower is 30. +100bp stretches the tenure to about 311 months, which
    ends near 56 and fits under the age-70 limit. At 45 the same extension
    would end past 70, and the reset raises the EMI instead (see
    test_at_45_the_same_extension_would_end_past_70_so_the_emi_rises).
    """
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
        "borrower_age": 30,
        "days_past_due": 0,
    }
    return {**base, **overrides}


def tape(*records: dict[str, Any]) -> LoanTape:
    return LoanTape.from_records(list(records))


FLAT = np.full(13, 0.055)
STEP = np.r_[0.055, np.full(12, 0.065)]  # +100bp from month 1


# ---- the defaults are the sourced values -----------------------------------------


def test_the_defaults_are_the_sourced_values() -> None:
    """MCLR 155bp for 250bp of repo; SBI's 70 and 30 years; the IMF's 50 percent."""
    params = RepricingParameters()
    assert params.internal_pass_through == pytest.approx(0.62)
    assert params.max_age_at_maturity == pytest.approx(70.0)
    assert params.max_tenure_months == 360
    assert params.dsti_threshold == pytest.approx(0.5)
    assert params.reset_policy is ResetPolicy.TENURE_FIRST


# ---- nothing moves when policy does not ----------------------------------------


def test_with_no_policy_change_the_emi_and_rate_stay_put() -> None:
    result = reprice(tape(loan()), FLAT)
    assert np.allclose(result.rate, 0.09)
    assert np.allclose(result.emi, result.emi[0, 0])
    assert not result.tenure_capped.any()


def test_a_loan_repays_to_zero_over_its_remaining_term() -> None:
    short = loan(
        outstanding=120_000.0, remaining_months=12, emi=float(annuity_payment(120_000.0, 0.09, 12))
    )
    result = reprice(tape(short), FLAT)
    assert result.outstanding[0, -1] == pytest.approx(0.0, abs=1e-6)
    assert result.emi[0, -1] == 0.0


# ---- when and how much -----------------------------------------------------------


def test_a_fixed_loan_does_not_reprice() -> None:
    fixed = loan(benchmark="fixed", reset_months=0, months_to_next_reset=0)
    assert np.allclose(reprice(tape(fixed), STEP).rate, 0.09)


def test_an_eblr_loan_reprices_on_its_schedule_by_the_full_move() -> None:
    """Next reset in month 2 (months_to_next_reset 1), then months 5, 8, 11."""
    rate = reprice(tape(loan()), STEP).rate[0]
    assert rate[1] == pytest.approx(0.09)  # month 1: not yet
    assert rate[2] == pytest.approx(0.10)  # month 2: the full 100bp
    assert np.allclose(rate[2:], 0.10)


def test_an_mclr_loan_passes_on_part_of_the_move() -> None:
    mclr = loan(benchmark="mclr", reset_months=12, months_to_next_reset=0)
    rate = reprice(tape(mclr), STEP).rate[0]
    assert rate[1] == pytest.approx(0.09 + 0.62 * 0.01)  # 155bp of MCLR per 250bp of repo


# ---- what the reset does ---------------------------------------------------------


def test_tenure_first_keeps_the_emi_and_extends_the_tenure() -> None:
    """+100bp on 9% over 20 years: the same EMI needs about 311 months, not 239."""
    result = reprice(tape(loan()), STEP)
    assert result.emi[0, 2] == pytest.approx(result.emi[0, 0])
    assert 300 < result.remaining_months[0, 2] < 330
    assert not result.tenure_capped[0]


def test_at_35_the_extension_fits_because_lenders_lend_to_70() -> None:
    """Under the old retirement age of 60 this borrower was capped; at 70 it fits."""
    result = reprice(tape(loan(borrower_age=35)), STEP)
    assert not result.tenure_capped[0]
    assert result.emi[0, 2] == pytest.approx(result.emi[0, 0])


def test_at_45_the_same_extension_would_end_past_70_so_the_emi_rises() -> None:
    result = reprice(tape(loan(borrower_age=45)), STEP)
    assert result.tenure_capped[0]
    assert result.emi[0, 2] > result.emi[0, 0]


def test_a_lender_with_a_lower_age_limit_caps_sooner() -> None:
    """At 40 the extension ends near 66: inside SBI's 70, outside a 65 limit."""
    assert not reprice(tape(loan(borrower_age=40)), STEP).tenure_capped[0]
    params = RepricingParameters(max_age_at_maturity=65.0)
    assert reprice(tape(loan(borrower_age=40)), STEP, params).tenure_capped[0]


def test_emi_first_keeps_the_tenure_and_raises_the_emi() -> None:
    params = RepricingParameters(reset_policy=ResetPolicy.EMI_FIRST)
    result = reprice(tape(loan()), STEP, params)
    expected = annuity_payment(result.outstanding[0, 1], 0.10, result.remaining_months[0, 1])
    assert result.emi[0, 2] == pytest.approx(expected)
    assert result.emi[0, 2] > result.emi[0, 0]


def test_an_extension_past_the_maturity_age_raises_the_emi_and_is_flagged() -> None:
    """Age 65 with 60 months left: no room to extend before 70, so the EMI rises."""
    late = loan(
        borrower_age=65,
        outstanding=500_000.0,
        remaining_months=60,
        emi=float(annuity_payment(500_000.0, 0.09, 60)),
    )
    result = reprice(tape(late), STEP)
    assert result.tenure_capped[0]
    assert result.emi[0, 2] > result.emi[0, 0]


# ---- flags -----------------------------------------------------------------------


def test_a_borrower_pushed_above_the_dsti_threshold_is_flagged() -> None:
    """EMI first at 49% DSTI: a 100bp rise takes the borrower above 50%."""
    income = annuity_payment(1_000_000.0, 0.09, 240) * 12 / 0.49
    params = RepricingParameters(reset_policy=ResetPolicy.EMI_FIRST)
    result = reprice(tape(loan(borrower_income=float(income))), STEP, params)
    assert result.borrower_dsti[0, 0] == pytest.approx(0.49)
    assert result.dsti_pushed_across[0]
    assert not result.dsti_above_at_start[0]


def test_a_borrower_already_above_the_threshold_is_not_counted_as_pushed() -> None:
    """At 59% DSTI, the Bajaj Housing Finance average FOIR at origination."""
    income = annuity_payment(1_000_000.0, 0.09, 240) * 12 / 0.59
    result = reprice(tape(loan(borrower_income=float(income))), STEP)
    assert result.dsti_above_at_start[0]
    assert not result.dsti_pushed_across[0]


def test_a_loan_already_below_interest_is_flagged_negative_amortisation() -> None:
    stretched = loan(emi=7_000.0, remaining_months=360)
    assert reprice(tape(stretched), FLAT).negative_amortisation[0]


def test_the_summary_reports_the_book() -> None:
    late = loan(
        loan_id="L2",
        borrower_id="B2",
        borrower_age=65,
        outstanding=500_000.0,
        remaining_months=60,
        emi=float(annuity_payment(500_000.0, 0.09, 60)),
    )
    summary = reprice(tape(loan(), late), STEP).summary()
    assert summary["loans"] == 2
    assert summary["share_tenure_capped"] == pytest.approx(0.5)
    assert summary["share_emi_up"] == pytest.approx(0.5)
    assert summary["share_dsti_above_at_start"] == 0.0
    assert summary["share_dsti_pushed_across"] == 0.0


# ---- refusals --------------------------------------------------------------------


@pytest.mark.parametrize("path", [[0.055], [[0.055, 0.06]], [0.055, float("nan")]])
def test_a_malformed_policy_path_is_refused(path: Any) -> None:
    with pytest.raises(EngineError):
        reprice(tape(loan()), path)
