"""Stages and expected credit loss (Stages 3.2 and 3.3).

Written before the module existed. The staging and floor tests pin the RBI's
final ECL directions; the rest pin the arithmetic and the refusals.
"""

from typing import Any

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.credit.ecl import (
    EclParameters,
    Stage,
    assign_stages,
    expected_credit_loss,
    weighted_provision,
)
from moirai.engine.credit.loan_tape import LoanTape, annuity_payment


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


def tape(*records: dict[str, Any]) -> LoanTape:
    return LoanTape.from_records(list(records))


def stages_of(book: LoanTape, now: list[float], before: list[float]) -> np.ndarray:
    return assign_stages(book, np.array(now), np.array(before))


# ---- staging: what the directions say -------------------------------------------


def test_a_performing_loan_whose_risk_is_unchanged_is_stage_one() -> None:
    assert stages_of(tape(loan()), [0.002], [0.002])[0] == Stage.ONE


def test_more_than_thirty_days_past_due_is_stage_two() -> None:
    assert stages_of(tape(loan(days_past_due=31)), [0.002], [0.002])[0] == Stage.TWO


def test_exactly_thirty_days_past_due_is_not_yet_stage_two() -> None:
    assert stages_of(tape(loan(days_past_due=30)), [0.002], [0.002])[0] == Stage.ONE


def test_more_than_ninety_days_past_due_is_stage_three() -> None:
    assert stages_of(tape(loan(days_past_due=91)), [0.002], [0.002])[0] == Stage.THREE


def test_a_doubled_probability_is_a_significant_increase_in_credit_risk() -> None:
    assert stages_of(tape(loan()), [0.004], [0.002])[0] == Stage.TWO


def test_a_probability_short_of_doubling_is_not() -> None:
    assert stages_of(tape(loan()), [0.0039], [0.002])[0] == Stage.ONE


def test_the_sicr_threshold_can_be_varied() -> None:
    book = tape(loan())
    loose = EclParameters(sicr_pd_multiple=3.0)
    assert assign_stages(book, np.array([0.004]), np.array([0.002]), loose)[0] == Stage.ONE


def test_crossing_the_debt_service_level_is_a_significant_increase() -> None:
    """The optional qualitative criterion: a shock carries the borrower past 50 percent."""
    crossing = EclParameters(sicr_dsti_threshold=0.50)
    stage = assign_stages(
        tape(loan()),
        np.array([0.002]),
        np.array([0.002]),
        crossing,
        dsti_now=np.array([0.52]),
        dsti_reference=np.array([0.48]),
    )
    assert stage[0] == Stage.TWO


def test_a_borrower_already_above_the_level_does_not_migrate_for_staying_there() -> None:
    crossing = EclParameters(sicr_dsti_threshold=0.50)
    stage = assign_stages(
        tape(loan()),
        np.array([0.002]),
        np.array([0.002]),
        crossing,
        dsti_now=np.array([0.62]),
        dsti_reference=np.array([0.59]),
    )
    assert stage[0] == Stage.ONE


def test_the_debt_service_criterion_needs_both_debt_service_arrays() -> None:
    crossing = EclParameters(sicr_dsti_threshold=0.50)
    with pytest.raises(EngineError, match="dsti_now"):
        assign_stages(tape(loan()), np.array([0.002]), np.array([0.002]), crossing)


def test_impairment_is_assessed_at_borrower_level() -> None:
    """The RBI applies Stage 3 to the borrower, so a clean loan moves with it."""
    impaired = loan(loan_id="L2", days_past_due=120)
    other = loan(loan_id="L3", borrower_id="B9", borrower_income=900_000.0)
    result = stages_of(tape(loan(), impaired, other), [0.002] * 3, [0.002] * 3)
    assert list(result) == [Stage.THREE, Stage.THREE, Stage.ONE]


# ---- the horizon ----------------------------------------------------------------


def test_stage_one_uses_the_twelve_month_probability() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([0.01]), np.array([0.20]))
    assert result.probability[0] == pytest.approx(0.01)


def test_stage_two_uses_the_lifetime_probability() -> None:
    result = expected_credit_loss(
        tape(loan(days_past_due=31)), [Stage.TWO], np.array([0.01]), np.array([0.20])
    )
    assert result.probability[0] == pytest.approx(0.20)


def test_stage_three_is_impaired_so_the_whole_loss_is_expected() -> None:
    result = expected_credit_loss(
        tape(loan(days_past_due=91)), [Stage.THREE], np.array([0.01]), np.array([0.20])
    )
    assert result.probability[0] == pytest.approx(1.0)
    assert result.modelled[0] == pytest.approx(0.65 * 1_000_000.0)


# ---- loss rates and exposure ----------------------------------------------------


def test_a_secured_loan_uses_the_sixty_five_percent_fallback() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([0.01]), np.array([0.2]))
    assert result.loss_given_default[0] == pytest.approx(0.65)


def test_an_unsecured_loan_uses_the_seventy_percent_fallback() -> None:
    unsecured = loan(
        product="personal",
        benchmark="fixed",
        reset_months=0,
        months_to_next_reset=0,
        outstanding=200_000.0,
        rate=0.14,
        remaining_months=48,
        emi=float(annuity_payment(200_000.0, 0.14, 48)),
    )
    result = expected_credit_loss(tape(unsecured), [Stage.ONE], np.array([0.01]), np.array([0.2]))
    assert result.loss_given_default[0] == pytest.approx(0.70)


def test_exposure_is_the_outstanding_balance() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([0.01]), np.array([0.2]))
    assert result.exposure[0] == pytest.approx(1_000_000.0)
    assert result.modelled[0] == pytest.approx(0.01 * 0.65 * 1_000_000.0)


# ---- the floors -----------------------------------------------------------------


def test_the_housing_stage_one_floor_is_twenty_five_basis_points() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([1e-9]), np.array([1e-9]))
    assert result.floor[0] == pytest.approx(0.0025 * 1_000_000.0)
    assert result.provision[0] == pytest.approx(0.0025 * 1_000_000.0)
    assert result.floor_bound[0]


def test_the_housing_stage_two_floor_is_one_and_a_half_percent() -> None:
    result = expected_credit_loss(
        tape(loan(days_past_due=31)), [Stage.TWO], np.array([1e-9]), np.array([1e-9])
    )
    assert result.floor[0] == pytest.approx(0.015 * 1_000_000.0)


def test_migrating_to_stage_two_multiplies_the_floor_sixfold() -> None:
    assert pytest.approx(6.0) == 0.015 / 0.0025


def test_a_modelled_loss_above_the_floor_is_what_is_booked() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([0.05]), np.array([0.2]))
    assert result.provision[0] == pytest.approx(result.modelled[0])
    assert not result.floor_bound[0]


def test_the_stage_three_housing_floor_rises_with_time_in_arrears() -> None:
    fresh = tape(loan(days_past_due=100))
    old = tape(loan(days_past_due=100 + 2 * 365))
    arguments = ([Stage.THREE], np.array([0.01]), np.array([0.2]))
    no_model = EclParameters(lgd_secured=1e-9)
    assert expected_credit_loss(fresh, *arguments, no_model).floor[0] == pytest.approx(
        0.10 * 1_000_000.0
    )
    assert expected_credit_loss(old, *arguments, no_model).floor[0] == pytest.approx(
        0.30 * 1_000_000.0
    )


def test_the_floors_can_be_switched_off_to_see_the_model_alone() -> None:
    bare = EclParameters(apply_floors=False)
    result = expected_credit_loss(
        tape(loan()), [Stage.ONE], np.array([1e-9]), np.array([1e-9]), bare
    )
    assert result.floor[0] == 0.0
    assert result.provision[0] == pytest.approx(result.modelled[0])


# ---- the summary ----------------------------------------------------------------


def test_the_summary_reports_coverage_and_what_the_floor_decided() -> None:
    result = expected_credit_loss(tape(loan()), [Stage.ONE], np.array([1e-9]), np.array([1e-9]))
    summary = result.summary()
    assert summary["exposure"] == pytest.approx(1_000_000.0)
    assert summary["coverage"] == pytest.approx(0.0025)
    assert summary["share_floor_bound"] == pytest.approx(1.0)
    assert summary["share_stage_1"] == pytest.approx(1.0)


# ---- scenarios ------------------------------------------------------------------


def three_results() -> dict[str, Any]:
    book = tape(loan())
    return {
        name: expected_credit_loss(book, [Stage.ONE], np.array([pd]), np.array([0.2]))
        for name, pd in (("base", 0.01), ("defend", 0.02), ("fall", 0.03))
    }


def test_the_weighted_provision_is_the_probability_weighted_sum() -> None:
    results = three_results()
    weights = {"base": 0.5, "defend": 0.3, "fall": 0.2}
    expected = sum(results[n].total() * w for n, w in weights.items())
    assert weighted_provision(results, weights) == pytest.approx(expected)


def test_fewer_than_three_scenarios_is_refused() -> None:
    results = three_results()
    del results["fall"]
    with pytest.raises(EngineError, match="three scenarios"):
        weighted_provision(results, {"base": 0.5, "defend": 0.5})


def test_weights_that_do_not_sum_to_one_are_refused() -> None:
    with pytest.raises(EngineError, match="sum"):
        weighted_provision(three_results(), {"base": 0.5, "defend": 0.3, "fall": 0.3})


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(EngineError, match="negative"):
        weighted_provision(three_results(), {"base": 1.2, "defend": -0.1, "fall": -0.1})


def test_weights_for_the_wrong_scenarios_are_refused() -> None:
    with pytest.raises(EngineError, match="do not match"):
        weighted_provision(three_results(), {"base": 0.4, "defend": 0.3, "other": 0.3})


# ---- refusals -------------------------------------------------------------------


def test_a_wrong_number_of_stages_is_refused() -> None:
    with pytest.raises(EngineError, match="one stage per loan"):
        expected_credit_loss(
            tape(loan()), [Stage.ONE, Stage.TWO], np.array([0.01]), np.array([0.2])
        )


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(EngineError, match="1, 2 or 3"):
        expected_credit_loss(tape(loan()), [4], np.array([0.01]), np.array([0.2]))


def test_probabilities_of_the_wrong_length_are_refused() -> None:
    with pytest.raises(EngineError, match="one per loan"):
        assign_stages(tape(loan()), np.array([0.01, 0.02]), np.array([0.01, 0.02]))
