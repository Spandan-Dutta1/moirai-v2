"""The synthetic loan book (Stage 2.5).

Written before the module existed. A synthetic book is only useful if it
reproduces what is published, so each sourced figure is a test with a
tolerance; what is assumed must be named on the book.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import Benchmark, Product
from moirai.engine.credit.repricing import reprice
from moirai.engine.credit.synthetic import (
    AFFORDABLE,
    AVERAGE_TICKET,
    PRIVATE,
    PSB,
    REPO_RATE,
    SyntheticBook,
    generate,
)


@pytest.fixture(scope="module")
def psb() -> SyntheticBook:
    return generate(PSB, n=100_000, seed=0)


@pytest.fixture(scope="module")
def private() -> SyntheticBook:
    return generate(PRIVATE, n=100_000, seed=0)


def value_share(book: SyntheticBook, mask: np.ndarray) -> float:
    out = book.tape.outstanding
    return float(out[mask].sum() / out.sum())


def test_the_product_mix_by_value_matches_crif(psb: SyntheticBook) -> None:
    """Home 43.0, auto 9.0, personal 15.9 lakh crore: 61.7, 12.9, 22.8 percent."""
    product = psb.tape.product
    assert value_share(psb, product == Product.HOME) == pytest.approx(43.0 / 67.9, abs=0.03)
    assert value_share(psb, product == Product.VEHICLE) == pytest.approx(9.0 / 67.9, abs=0.03)
    assert value_share(psb, product == Product.PERSONAL) == pytest.approx(15.9 / 67.9, abs=0.03)


@pytest.mark.parametrize("product", [Product.HOME, Product.VEHICLE, Product.PERSONAL])
def test_the_average_ticket_matches_crif(psb: SyntheticBook, product: Product) -> None:
    mask = psb.tape.product == product
    mean = float(np.mean(psb.original_principal[mask]))
    assert mean == pytest.approx(AVERAGE_TICKET[product], rel=0.05)


@pytest.mark.parametrize(("name", "target"), [("psb", 0.498), ("private", 0.886)])
def test_the_eblr_share_of_floating_loans_matches_the_rbi(
    psb: SyntheticBook, private: SyntheticBook, name: str, target: float
) -> None:
    book = psb if name == "psb" else private
    floating = book.tape.is_floating
    eblr = book.tape.benchmark == Benchmark.EBLR
    share = float(book.tape.outstanding[eblr].sum() / book.tape.outstanding[floating].sum())
    assert share == pytest.approx(target, abs=0.03)


def test_the_average_age_at_origination_is_38(psb: SyntheticBook) -> None:
    assert float(np.mean(psb.origination_age)) == pytest.approx(38.0, abs=0.5)


def test_no_loan_runs_past_age_70(psb: SyntheticBook) -> None:
    tape = psb.tape
    assert np.all(tape.borrower_age + tape.remaining_months / 12.0 <= 70.0 + 1.0)


@pytest.mark.parametrize(("profile", "target"), [(PSB, 0.59), (AFFORDABLE, 0.40)])
def test_the_average_dsti_matches_the_lenders(profile: object, target: float) -> None:
    book = generate(profile, n=20_000, seed=0)  # type: ignore[arg-type]
    assert float(np.mean(book.tape.borrower_dsti)) == pytest.approx(target, abs=0.01)


def test_home_loans_float_and_the_rest_are_fixed(psb: SyntheticBook) -> None:
    home = psb.tape.product == Product.HOME
    assert np.all(psb.tape.is_floating[home])
    assert not np.any(psb.tape.is_floating[~home])


def test_eblr_loans_are_priced_off_the_repo_rate(psb: SyntheticBook) -> None:
    eblr = psb.tape.benchmark == Benchmark.EBLR
    assert np.allclose(psb.tape.rate[eblr] - psb.tape.spread[eblr], REPO_RATE)


def test_the_assumptions_are_named_on_the_book(psb: SyntheticBook) -> None:
    assert "dsti_shape" in psb.assumed
    assert "rate_type" in psb.assumed


def test_the_same_seed_gives_the_same_book() -> None:
    a = generate(AFFORDABLE, n=1_000, seed=4).tape
    b = generate(AFFORDABLE, n=1_000, seed=4).tape
    c = generate(AFFORDABLE, n=1_000, seed=5).tape
    assert np.array_equal(a.outstanding, b.outstanding)
    assert not np.array_equal(a.outstanding, c.outstanding)


def test_the_book_runs_through_the_reset_engine() -> None:
    book = generate(PRIVATE, n=2_000, seed=1)
    path = np.r_[REPO_RATE, np.full(12, REPO_RATE + 0.0025)]
    result = reprice(book.tape, path)
    fixed = ~book.tape.is_floating
    assert np.allclose(result.rate[fixed, -1], book.tape.rate[fixed])
    assert np.all(result.rate[~fixed, -1] > book.tape.rate[~fixed])


def test_a_tiny_book_is_refused() -> None:
    with pytest.raises(EngineError):
        generate(PSB, n=10)
