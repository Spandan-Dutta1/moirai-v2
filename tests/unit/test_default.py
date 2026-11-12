"""Probability of default from debt service (Stage 3.1).

Written before the module existed. Each test pins one published number from
Nier, Popa, Shamloo and Voinea (IMF WP/19/182) or one property of the shape
they estimate: flat below the threshold, compounding above it, continuous at
every band edge.
"""

from typing import Any

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.credit.default import (
    CONSUMER_BANDS,
    MORTGAGE_BANDS,
    DefaultParameters,
    calibrate_flat_pd,
    dsti_multiplier,
    lifetime_default,
    probability_of_default,
    tape_default,
)
from moirai.engine.credit.loan_tape import LoanTape, Product, annuity_payment

FLAT = DefaultParameters().flat_pd_12m


def home(dsti: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([dsti]), np.array([Product.HOME])


def personal(dsti: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([dsti]), np.array([Product.PERSONAL])


# ---- the published bands -------------------------------------------------------


def test_the_bands_are_the_published_ones() -> None:
    """Mortgages: flat to 50, then 16, 14 and 8 percent per 10pp. Table 4, col 7."""
    assert ((0.50, 0.90, 0.16), (0.90, 1.20, 0.14), (1.20, np.inf, 0.08)) == MORTGAGE_BANDS
    assert ((0.30, 0.50, 0.08), (0.50, 0.70, 0.06), (0.70, np.inf, 0.03)) == CONSUMER_BANDS


@pytest.mark.parametrize("dsti", [0.0, 0.20, 0.40, 0.4999, 0.50])
def test_mortgage_risk_is_flat_below_fifty_percent(dsti: float) -> None:
    assert probability_of_default(*home(dsti))[0] == pytest.approx(FLAT)


@pytest.mark.parametrize("dsti", [0.0, 0.15, 0.2999, 0.30])
def test_consumer_risk_is_flat_below_thirty_percent(dsti: float) -> None:
    assert probability_of_default(*personal(dsti))[0] == pytest.approx(FLAT)


def test_ten_points_above_the_mortgage_threshold_raises_pd_by_sixteen_percent() -> None:
    assert probability_of_default(*home(0.60))[0] == pytest.approx(FLAT * 1.16)


def test_ten_points_above_the_consumer_threshold_raises_pd_by_eight_percent() -> None:
    assert probability_of_default(*personal(0.40))[0] == pytest.approx(FLAT * 1.08)


def test_the_bands_compound_through_each_other() -> None:
    """At 100 percent: four steps of 16 percent to 90, then one of 14."""
    expected = FLAT * 1.16**4 * 1.14
    assert probability_of_default(*home(1.00))[0] == pytest.approx(expected)


def test_the_multiplier_is_continuous_at_every_band_edge() -> None:
    for edge in (0.50, 0.90, 1.20):
        below = dsti_multiplier(np.array([edge - 1e-9]), MORTGAGE_BANDS)[0]
        above = dsti_multiplier(np.array([edge + 1e-9]), MORTGAGE_BANDS)[0]
        assert above == pytest.approx(below, rel=1e-6)


def test_the_slope_flattens_as_dsti_rises() -> None:
    """8 percent per 10pp above 120 against 16 below 90: the published shape."""
    steep = (
        dsti_multiplier(np.array([0.70]), MORTGAGE_BANDS)[0]
        / dsti_multiplier(np.array([0.60]), MORTGAGE_BANDS)[0]
    )
    shallow = (
        dsti_multiplier(np.array([1.40]), MORTGAGE_BANDS)[0]
        / dsti_multiplier(np.array([1.30]), MORTGAGE_BANDS)[0]
    )
    assert steep == pytest.approx(1.16)
    assert shallow == pytest.approx(1.08)
    assert shallow < steep


def test_a_loan_against_property_follows_the_mortgage_bands() -> None:
    lap = probability_of_default(np.array([0.60]), np.array([Product.LOAN_AGAINST_PROPERTY]))
    assert lap[0] == pytest.approx(FLAT * 1.16)


# ---- the floor ------------------------------------------------------------------


def test_the_rbi_pd_floor_binds_below_three_basis_points() -> None:
    tiny = DefaultParameters(flat_pd_12m=1e-6)
    assert probability_of_default(*home(0.10), tiny)[0] == pytest.approx(0.0003)


# ---- lifetime -------------------------------------------------------------------


def test_lifetime_over_twelve_months_is_the_twelve_month_probability() -> None:
    assert lifetime_default(np.array([0.01]), np.array([12]))[0] == pytest.approx(0.01)


def test_lifetime_compounds_over_the_remaining_term() -> None:
    assert lifetime_default(np.array([0.01]), np.array([240]))[0] == pytest.approx(1.0 - 0.99**20)


def test_a_repaid_loan_has_no_lifetime_risk_left() -> None:
    assert lifetime_default(np.array([0.05]), np.array([0]))[0] == pytest.approx(0.0)


# ---- calibration ----------------------------------------------------------------


def test_calibration_reproduces_the_target_coverage() -> None:
    """Bajaj Housing Finance disclose 0.33 percent Stage 1 coverage."""
    rng = np.random.default_rng(0)
    dsti = rng.uniform(0.2, 1.1, 500)
    product = np.full(500, Product.HOME)
    outstanding = rng.uniform(5e5, 5e6, 500)
    flat = calibrate_flat_pd(dsti, product, outstanding, 0.0033, 0.65)

    annual = probability_of_default(dsti, product, DefaultParameters(flat_pd_12m=flat))
    coverage = float(np.average(annual * 0.65, weights=outstanding))
    assert coverage == pytest.approx(0.0033, rel=1e-6)


def test_the_default_flat_pd_is_the_calibrated_one() -> None:
    """The prime book's home loans against Bajaj Housing Finance's 0.33 percent."""
    assert pytest.approx(0.00434, abs=5e-5) == FLAT


# ---- a tape ---------------------------------------------------------------------


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


def test_a_tape_gives_both_horizons() -> None:
    tape = LoanTape.from_records([loan()])
    annual, lifetime = tape_default(tape)
    assert annual[0] == pytest.approx(probability_of_default(tape.borrower_dsti, tape.product)[0])
    assert lifetime[0] > annual[0]


# ---- refusals -------------------------------------------------------------------


@pytest.mark.parametrize("dsti", [-0.1, float("nan"), float("inf")])
def test_an_impossible_dsti_is_refused(dsti: float) -> None:
    with pytest.raises(EngineError, match="DSTI"):
        probability_of_default(*home(dsti))


def test_a_shape_mismatch_is_refused() -> None:
    with pytest.raises(EngineError, match="shape"):
        probability_of_default(np.array([0.5, 0.6]), np.array([Product.HOME]))


@pytest.mark.parametrize(("coverage", "lgd"), [(0.0, 0.65), (1.5, 0.65), (0.003, 0.0)])
def test_an_impossible_calibration_target_is_refused(coverage: float, lgd: float) -> None:
    with pytest.raises(EngineError):
        calibrate_flat_pd(np.array([0.5]), np.array([Product.HOME]), np.array([1.0]), coverage, lgd)
