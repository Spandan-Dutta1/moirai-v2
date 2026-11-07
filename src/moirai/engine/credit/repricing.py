"""
The EMI reset engine: what a policy rate path does to each loan, month by month.

A policy move reaches a borrower only when the loan reprices, by as much as its
benchmark passes on, and in the form the lender and borrower choose at the
reset. Those three things are the transmission the household layer does not
model, and they are what the August 2023 reset framework was about.

    when     EBLR loans on their reset schedule (at least every three months);
             MCLR and base-rate loans on their own cycle; fixed loans never.
             months_to_next_reset = k means the next reset is in month k + 1,
             then every reset_months after.
    how much the policy change since the tape's date, times the benchmark's
             pass-through: 1.0 for EBLR (the repo rate plus a fixed spread),
             a measured fraction for internal benchmarks.
    how      tenure first (keep the EMI, extend the tenure) or EMI first
             (keep the tenure, recompute the EMI). Under tenure first an
             extension may not run past the lender's maximum age at maturity
             or its longest tenure; where it would, the EMI rises instead.

Then each month interest accrues on the outstanding principal and the EMI is
paid. Income is held at the tape's value over the horizon.

Flags, per loan over the horizon:

    tenure_capped         an extension hit the age or tenure limit, so the EMI
                          was forced up
    negative_amortisation the EMI did not cover a month's interest
    dsti_above_at_start   the borrower's DSTI was already above the threshold
                          on the tape's date
    dsti_pushed_across    the borrower started at or below the threshold and
                          the path took them above it

The two DSTI flags are kept apart because Indian books start high: Bajaj
Housing Finance reports an average FOIR of 59 percent at origination, so a
single flag would mark most of a prime book before any shock.

Each default on RepricingParameters names its source. The pass-through for
internal benchmarks is one number, although the evidence shows it grows with
time; a lagged estimate on the RBI's monthly MCLR and repo series is the next
refinement.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import Benchmark, LoanTape, annuity_payment


class ResetPolicy(StrEnum):
    """What happens at a reset (RBI, August 2023: the borrower must be offered both)."""

    TENURE_FIRST = "tenure_first"
    EMI_FIRST = "emi_first"


class RepricingParameters(BaseModel):
    """How the book reprices. Each default names its source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    eblr_pass_through: float = Field(
        default=1.0,
        ge=0.0,
        le=1.5,
        description=(
            "Contractual. RBI, September 2019: the repo rate plus a spread fixed for "
            "the loan, reset at least every three months."
        ),
    )
    internal_pass_through: float = Field(
        default=0.62,
        ge=0.0,
        le=1.5,
        description=(
            "SOURCED. The 1-year median MCLR moved 155bp for 250bp of repo in the "
            "2019-21 easing (RBI Bulletin, July 2021) and in the 2022-24 tightening "
            "(NIBM, 2024). It grows with time: about 0.36 some 15 months into the "
            "2025 easing, 0.70 after the full tightening to November 2024."
        ),
    )
    reset_policy: ResetPolicy = Field(
        default=ResetPolicy.TENURE_FIRST,
        description=(
            "SOURCED for 2022-23: lenders extended tenure first (RBI statement of "
            "10 August 2023, 'unreasonable elongation of tenor'). Since the August "
            "2023 framework the borrower chooses, so forward runs report both."
        ),
    )
    max_age_at_maturity: float = Field(
        default=70.0,
        ge=40.0,
        le=85.0,
        description=(
            "SOURCED. The age by which the loan must be repaid: 70 at SBI (30 years "
            "or age 70, whichever is earlier), about 65 at HDFC. A lender's rule, "
            "set per book; not the household layer's retirement age."
        ),
    )
    max_tenure_months: int = Field(
        default=360,
        ge=12,
        le=600,
        description=(
            "SOURCED. 30 years at SBI and HDFC. Applied to the remaining tenure "
            "because the tape has no months on book, so it bounds an extension "
            "from above."
        ),
    )
    dsti_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=2.0,
        description=(
            "SOURCED. Mortgage default risk is flat below 50 percent DSTI and rises "
            "above it (Nier, Popa, Shamloo and Voinea, IMF WP/19/182, loan-level "
            "data). Measured on net income, as the tape's income is."
        ),
    )

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RepricingResult(BaseModel):
    """Each loan's path over the horizon; column 0 is the tape's own state."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    parameters: RepricingParameters
    rate: Any = Field(description="(n, H+1) annual rate.")
    emi: Any = Field(description="(n, H+1) monthly payment; 0 once repaid.")
    remaining_months: Any = Field(description="(n, H+1) months left.")
    outstanding: Any = Field(description="(n, H+1) principal outstanding.")
    borrower_dsti: Any = Field(description="(n, H+1) the borrower's DSTI, per loan row.")
    tenure_capped: Any = Field(description="(n,) an extension hit a limit.")
    negative_amortisation: Any = Field(description="(n,) EMI below a month's interest.")
    dsti_above_at_start: Any = Field(description="(n,) DSTI above the threshold in month 0.")
    dsti_pushed_across: Any = Field(
        description="(n,) at or below the threshold in month 0, above it later."
    )

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.emi).shape[1] - 1)

    def summary(self) -> dict[str, float]:
        emi = np.asarray(self.emi)
        start, peak = emi[:, 0], emi.max(axis=1)
        extended = (
            np.asarray(self.remaining_months)[:, -1]
            + self.horizon
            - np.asarray(self.remaining_months)[:, 0]
        )
        return {
            "loans": float(emi.shape[0]),
            "share_emi_up": float(np.mean(peak > start * (1 + 1e-9))),
            "mean_peak_emi_change": float(np.mean(peak / start - 1.0)),
            "mean_months_extended": float(np.mean(np.maximum(extended, 0))),
            "share_tenure_capped": float(np.mean(self.tenure_capped)),
            "share_negative_amortisation": float(np.mean(self.negative_amortisation)),
            "share_dsti_above_at_start": float(np.mean(self.dsti_above_at_start)),
            "share_dsti_pushed_across": float(np.mean(self.dsti_pushed_across)),
        }


def _months_to_repay(principal: np.ndarray, annual_rate: np.ndarray, emi: np.ndarray) -> np.ndarray:
    """Months an EMI takes to repay principal at a rate; infinite if it cannot."""
    r = annual_rate / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = principal * r / emi
        months = np.where(r > 0, -np.log(1.0 - ratio) / np.log(1.0 + r), principal / emi)
    months = np.where((r > 0) & (ratio >= 1.0), np.inf, months)
    return np.ceil(months - 1e-9)


def reprice(
    tape: LoanTape,
    policy_path: Any,
    parameters: RepricingParameters | None = None,
) -> RepricingResult:
    """Run a monthly policy rate path through every loan in the tape.

    `policy_path[0]` is the policy rate on the tape's date; `policy_path[t]` the
    rate in month t. Only the change from the first entry matters.
    """
    parameters = parameters or RepricingParameters()
    path = np.asarray(policy_path, dtype=float)
    if path.ndim != 1 or path.size < 2 or not np.all(np.isfinite(path)):
        raise EngineError("the policy path must be a finite 1-D array of at least two months")
    horizon = path.size - 1
    n = len(tape)

    pass_through = np.where(
        tape.benchmark == Benchmark.EBLR,
        parameters.eblr_pass_through,
        parameters.internal_pass_through,
    )
    floating = tape.is_floating
    first_reset = tape.months_to_next_reset + 1
    _, borrower = np.unique(tape.borrower_id, return_inverse=True)

    rate = np.empty((n, horizon + 1))
    emi = np.empty((n, horizon + 1))
    remaining = np.empty((n, horizon + 1))
    outstanding = np.empty((n, horizon + 1))
    dsti = np.empty((n, horizon + 1))

    p = tape.outstanding.astype(float).copy()
    r = tape.rate.astype(float).copy()
    e = tape.emi.astype(float).copy()
    m = tape.remaining_months.astype(float).copy()
    capped = np.zeros(n, dtype=bool)
    negative = ~tape.covers_interest

    def record(t: int) -> None:
        rate[:, t], emi[:, t], remaining[:, t], outstanding[:, t] = r, e, m, p
        total = np.bincount(borrower, weights=e)
        dsti[:, t] = total[borrower] * 12.0 / tape.borrower_income

    record(0)
    for t in range(1, horizon + 1):
        open_ = p > 0
        resets = (
            floating
            & open_
            & (t >= first_reset)
            & ((t - first_reset) % np.maximum(tape.reset_months, 1) == 0)
        )
        if resets.any():
            r = np.where(
                resets, np.maximum(tape.rate + pass_through * (path[t] - path[0]), 1e-6), r
            )
            age = tape.borrower_age + t / 12.0
            limit = np.minimum(
                np.floor((parameters.max_age_at_maturity - age) * 12.0),
                parameters.max_tenure_months,
            )
            limit = np.maximum(limit, m)  # a reset never shortens the tenure it may use
            if parameters.reset_policy is ResetPolicy.EMI_FIRST:
                e = np.where(resets, annuity_payment(p, r, np.maximum(m, 1)), e)
            else:
                needed = _months_to_repay(p, r, e)
                fits = needed <= limit
                newly_capped = resets & ~fits
                m = np.where(resets & fits, needed, m)
                m = np.where(newly_capped, limit, m)
                e = np.where(newly_capped, annuity_payment(p, r, np.maximum(limit, 1)), e)
                capped |= newly_capped

        interest = p * r / 12.0
        negative |= open_ & (e < interest - 1e-9)
        payment = np.where(open_, np.minimum(e, p + interest), 0.0)
        p = np.where(open_, p + interest - payment, 0.0)
        m = np.where(open_, np.maximum(m - 1.0, 0.0), 0.0)
        repaid = p <= 1e-6
        p = np.where(repaid, 0.0, p)
        e = np.where(repaid, 0.0, e)
        record(t)

    above_at_start = dsti[:, 0] > parameters.dsti_threshold
    pushed_across = ~above_at_start & (dsti[:, 1:] > parameters.dsti_threshold).any(axis=1)

    return RepricingResult(
        parameters=parameters,
        rate=rate,
        emi=emi,
        remaining_months=remaining,
        outstanding=outstanding,
        borrower_dsti=dsti,
        tenure_capped=capped,
        negative_amortisation=negative,
        dsti_above_at_start=above_at_start,
        dsti_pushed_across=pushed_across,
    )
