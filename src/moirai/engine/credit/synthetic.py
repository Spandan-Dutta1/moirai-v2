"""
A synthetic loan book matched to published Indian aggregates (Stage 2.5).

Until a lender shares a tape, the engine runs on a book built to match what is
published. The published numbers are targets the book must reproduce, and the
tests check that it does. What no source publishes (the shape of each
distribution around those numbers, and how the fields vary together) is set
here, named in ASSUMED and reported on every book, so that a result on a
synthetic book cannot be presented as a result on a real one.

Sourced:

    repo rate          5.50 percent, after the 7 October 2026 hike (RBI MPC)
    median MCLR        8.61 percent, 1-year, September 2026 (RBI)
    product mix        outstanding by value, December 2025: home 43.0, auto
                       9.0, personal 15.9 lakh crore (CRIF High Mark)
    average ticket     at origination, Q3 FY26: home Rs 33 lakh, auto Rs 8.4
                       lakh, personal Rs 69,400 (CRIF High Mark)
    benchmark mix      EBLR share of floating loans, September 2025: public
                       sector banks 49.8, private banks 88.6 percent (RBI)
    borrower age       38 on average at origination (HDFC, March 2021)
    DSTI               average FOIR at origination: 59 percent for a prime
                       book (Bajaj Housing Finance), about 40 percent for an
                       affordable book (Home First)
    maximum age        no loan runs past age 70 (SBI)

Vehicle loans are cars; two-wheelers, with a ticket a tenth the size, are left
out. Housing finance companies price off their own reference rate; the
affordable book treats it as an internal benchmark that resets yearly, like
MCLR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import Benchmark, LoanTape, Product, annuity_payment

#: Repo rate on the book's date: 5.50% after the 7 October 2026 hike (RBI MPC).
REPO_RATE = 0.055
#: 1-year median MCLR, September 2026 (RBI, Lending and Deposit Rates).
MEDIAN_MCLR = 0.0861
#: Outstanding by value, Rs lakh crore, December 2025 (CRIF High Mark).
SYSTEM_PORTFOLIO = {Product.HOME: 43.0, Product.VEHICLE: 9.0, Product.PERSONAL: 15.9}
#: Average ticket at origination, Rs, Q3 FY26 (CRIF High Mark).
AVERAGE_TICKET = {
    Product.HOME: 3_300_000.0,
    Product.VEHICLE: 840_000.0,
    Product.PERSONAL: 69_400.0,
}
#: Average borrower age at origination (HDFC, March 2021).
AVERAGE_AGE = 38.0
#: No loan may run past this age (SBI: 30 years or age 70, whichever is earlier).
MAX_AGE_AT_MATURITY = 70.0

#: What no source publishes, and what the generator does instead.
ASSUMED = {
    "one_loan_per_borrower": "each borrower holds one loan",
    "rate_type": "home loans float; vehicle and personal loans are fixed",
    "ticket_shape": "lognormal around the sourced mean; sigma 0.6 home, 0.5 vehicle, 0.8 personal",
    "original_tenure": "home 180, 240 or 300 months; vehicle 60; personal 36, 48 or 60",
    "seasoning": "months since origination uniform over the tenure",
    "spreads": (
        "EBLR home: repo plus the profile's spread range; internal home: the profile's "
        "reference rate + 0 to 0.50pp; vehicle 8.5 to 11%; personal 10.5 to 16%"
    ),
    "reset_timing": "months to the next reset uniform within the reset period",
    "age_shape": "normal, sd 7, clipped to 21 to 60 at origination",
    "dsti_shape": (
        "beta around the sourced mean, concentration 20, clipped to 0.15 to 0.85; "
        "held at its origination value (no income growth since origination)"
    ),
    "performing": "no loan is past due",
}

SIGMA = {Product.HOME: 0.6, Product.VEHICLE: 0.5, Product.PERSONAL: 0.8}
TENURES = {Product.HOME: (180, 240, 300), Product.VEHICLE: (60,), Product.PERSONAL: (36, 48, 60)}
FIXED_RATES = {Product.VEHICLE: (0.085, 0.11), Product.PERSONAL: (0.105, 0.16)}
DSTI_CONCENTRATION = 20.0
PILOT = 4_000


class BookProfile(BaseModel):
    """The published figures a synthetic book is built to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    portfolio: dict[Product, float] = Field(description="Outstanding by value, any units.")
    eblr_share: float = Field(ge=0.0, le=1.0, description="EBLR share of floating loans.")
    mean_dsti: float = Field(gt=0.0, lt=1.0, description="Average DSTI (FOIR).")
    home_ticket: float = Field(default=AVERAGE_TICKET[Product.HOME], gt=0.0)
    repo_rate: float = Field(
        default=REPO_RATE, gt=0.0, le=0.25, description="The policy rate on the book's date."
    )
    internal_reference_rate: float = Field(
        default=MEDIAN_MCLR,
        gt=0.0,
        le=0.25,
        description="The internal benchmark MCLR loans are priced off.",
    )
    eblr_spread: tuple[float, float] = Field(
        default=(0.02, 0.0325), description="Range of spreads over the repo rate for EBLR loans."
    )
    source: str


PSB = BookProfile(
    name="psb",
    portfolio=SYSTEM_PORTFOLIO,
    eblr_share=0.498,
    mean_dsti=0.59,
    source="EBLR 49.8% (RBI, Sep 2025); FOIR 59% (Bajaj Housing Finance); mix CRIF",
)
PRIVATE = BookProfile(
    name="private",
    portfolio=SYSTEM_PORTFOLIO,
    eblr_share=0.886,
    mean_dsti=0.59,
    source="EBLR 88.6% (RBI, Sep 2025); FOIR 59% (Bajaj Housing Finance); mix CRIF",
)
AFFORDABLE = BookProfile(
    name="affordable",
    portfolio={Product.HOME: 1.0},
    eblr_share=0.0,
    mean_dsti=0.40,
    home_ticket=1_200_000.0,
    source="FOIR about 40% (Home First); home only; ticket Rs 12 lakh ASSUMED",
)
#: The book as it stood when the 2022-23 tightening began, for the backtest
#: (ADR 035). Home loans only: the housing benchmark mix is published by sector
#: for December 2021, and the FY22 product mix is not reachable. The ticket is
#: not period-sourced; it scales rupee totals and nothing the backtest tests,
#: which `test_the_crossing_share_does_not_depend_on_the_ticket` pins.
APRIL_2022 = BookProfile(
    name="april_2022",
    portfolio={Product.HOME: 1.0},
    eblr_share=0.637,
    mean_dsti=0.59,
    repo_rate=0.04,
    internal_reference_rate=0.0725,
    eblr_spread=(0.0325, 0.0375),
    source=(
        "repo 4.00% (RBI, 1 April 2022); EBLR 58.2% against MCLR 33.1% of housing loans, "
        "so 63.7% of floating (RBI Bulletin, April 2022, Table 3, Dec-21); housing WALR "
        "outstanding 7.49% (RBI, Feb-22) is what the spreads are set to reproduce; "
        "FOIR 59% (Bajaj Housing Finance, Q4 FY21); ticket not period-sourced"
    ),
)

PROFILES = (PSB, PRIVATE, AFFORDABLE)


class SyntheticBook(BaseModel):
    """A generated tape, with what it was built from and what was assumed."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    profile: BookProfile
    seed: int
    tape: LoanTape
    original_principal: Any = Field(description="(n,) amount lent at origination.")
    origination_age: Any = Field(description="(n,) borrower's age at origination.")
    assumed: tuple[str, ...] = tuple(ASSUMED)


def _draw(
    product: Product, k: int, rng: np.random.Generator, profile: BookProfile
) -> dict[str, np.ndarray]:
    """k loans of one product, with their state on the book's date."""
    ticket = profile.home_ticket if product is Product.HOME else AVERAGE_TICKET[product]
    sigma = SIGMA[product]
    principal = ticket * np.exp(sigma * rng.standard_normal(k) - sigma**2 / 2.0)

    age0 = np.clip(rng.normal(AVERAGE_AGE, 7.0, k), 21.0, 60.0)
    tenure = rng.choice(np.asarray(TENURES[product]), k)
    tenure = np.maximum(np.minimum(tenure, np.floor((MAX_AGE_AT_MATURITY - age0) * 12.0)), 12)
    tenure = tenure.astype(int)
    elapsed = rng.integers(0, tenure)

    if product is Product.HOME:
        eblr = rng.random(k) < profile.eblr_share
        benchmark = np.where(eblr, Benchmark.EBLR.value, Benchmark.MCLR.value)
        low_spread, high_spread = profile.eblr_spread
        rate = np.where(
            eblr,
            profile.repo_rate + rng.uniform(low_spread, high_spread, k),
            profile.internal_reference_rate + rng.uniform(0.0, 0.005, k),
        )
        spread = np.where(
            eblr, rate - profile.repo_rate, rate - profile.internal_reference_rate
        )
        reset_months = np.where(eblr, 3, 12)
        next_reset = np.floor(rng.random(k) * reset_months).astype(int)
    else:
        low, high = FIXED_RATES[product]
        benchmark = np.full(k, Benchmark.FIXED.value)
        rate = rng.uniform(low, high, k)
        spread = np.zeros(k)
        reset_months = np.zeros(k, dtype=int)
        next_reset = np.zeros(k, dtype=int)

    growth = 1.0 + rate / 12.0
    outstanding = principal * (growth**tenure - growth**elapsed) / (growth**tenure - 1.0)
    remaining = tenure - elapsed
    emi = annuity_payment(outstanding, rate, remaining)

    m = profile.mean_dsti
    dsti = np.clip(rng.beta(m * DSTI_CONCENTRATION, (1.0 - m) * DSTI_CONCENTRATION, k), 0.15, 0.85)
    return {
        "product": np.full(k, product.value),
        "benchmark": benchmark,
        "rate": rate,
        "spread": spread,
        "outstanding": outstanding,
        "remaining_months": remaining,
        "emi": emi,
        "reset_months": reset_months,
        "months_to_next_reset": next_reset,
        "borrower_income": emi * 12.0 / dsti,
        "borrower_age": np.floor(age0 + elapsed / 12.0).astype(int),
        "original_principal": principal,
        "origination_age": age0,
    }


def generate(profile: BookProfile, n: int = 100_000, seed: int = 0) -> SyntheticBook:
    """A book of about n loans whose aggregates match the profile's sources.

    Loan counts per product are set so that outstanding value, not the number of
    loans, matches the portfolio mix: a home loan is worth about forty personal
    loans, so a book weighted by value is mostly personal loans by count.
    """
    if n < 100:
        raise EngineError("a synthetic book needs at least 100 loans")
    rng = np.random.default_rng(seed)
    products = list(profile.portfolio)
    weights = np.array([profile.portfolio[p] for p in products], dtype=float)
    weights /= weights.sum()

    pilot = np.random.default_rng(seed + 1)
    mean_outstanding = np.array(
        [_draw(p, PILOT, pilot, profile)["outstanding"].mean() for p in products]
    )
    count_share = weights / mean_outstanding
    counts = np.maximum(np.round(n * count_share / count_share.sum()).astype(int), 1)

    parts = [_draw(p, int(k), rng, profile) for p, k in zip(products, counts, strict=True)]
    columns = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    total = int(counts.sum())
    loan_id = np.array([f"{profile.name}-{i:07d}" for i in range(total)])
    fields: dict[str, Any] = {
        k: v for k, v in columns.items() if k not in ("original_principal", "origination_age")
    }
    fields["loan_id"] = loan_id
    fields["borrower_id"] = np.char.replace(loan_id, profile.name, f"{profile.name}-b")
    fields["days_past_due"] = np.zeros(total, dtype=int)
    tape = LoanTape(**fields)
    return SyntheticBook(
        profile=profile,
        seed=seed,
        tape=tape,
        original_principal=columns["original_principal"],
        origination_age=columns["origination_age"],
    )
