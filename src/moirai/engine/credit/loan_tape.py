"""
A lender's loan book, one row per loan (Stage 2).

The first input Moirai takes from a lender rather than from a statistics
agency. It is held as parallel arrays, like Population, so a book of a million
loans is processed by vectorised expressions. Everything a loan must satisfy is
checked on construction, and a refusal names the loans that broke the rule:
a tape that contradicts itself is a data problem the lender must see, not one
the model should quietly smooth over.

The rules come from how Indian retail loans work:

  * Benchmarks. Externally benchmarked loans (EBLR, usually the repo rate plus
    a spread) must reset at least once every three months (RBI, September
    2019). MCLR and base-rate loans reset on their own cycle, at most yearly
    here. Fixed-rate loans do not reset.
  * The EMI is the annuity on the outstanding principal at the current rate
    over the remaining months. A tape whose EMI disagrees is refused, with one
    exception: a loan whose EMI no longer covers its monthly interest is
    accepted and flagged. That is negative amortisation, the outcome of the
    tenure extensions of 2022-23 that led to the RBI's August 2023 reset
    framework, and it is a signal to report, not bad data.
  * DSTI (debt service to income) is a borrower's measure: every EMI the
    borrower pays, against one income.

The core household arrays are untouched. A tape becomes a Population through
an adapter, and its loan-level mechanics live in this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from moirai.core.exceptions import EngineError


class Product(StrEnum):
    """Amortising retail loan products. Revolving credit is out of scope."""

    HOME = "home"
    LOAN_AGAINST_PROPERTY = "lap"
    VEHICLE = "vehicle"
    PERSONAL = "personal"
    EDUCATION = "education"
    BUSINESS = "business"


class Benchmark(StrEnum):
    """What the loan's rate is linked to."""

    EBLR = "eblr"
    MCLR = "mclr"
    BASE_RATE = "base_rate"
    FIXED = "fixed"


#: RBI, September 2019: an externally benchmarked loan resets at least quarterly.
MAX_EBLR_RESET_MONTHS = 3
#: Internal benchmarks reset on their own cycle; at most yearly here.
MAX_INTERNAL_RESET_MONTHS = 12
#: How far a tape's EMI may sit from the annuity before the row is refused.
EMI_TOLERANCE = 0.05
#: How many offending loans an error message lists.
SHOWN = 5

FIELDS = (
    "loan_id",
    "borrower_id",
    "product",
    "benchmark",
    "rate",
    "spread",
    "outstanding",
    "remaining_months",
    "emi",
    "reset_months",
    "months_to_next_reset",
    "borrower_income",
    "borrower_age",
    "days_past_due",
)


def annuity_payment(principal: Any, annual_rate: Any, months: Any) -> Any:
    """The monthly payment that repays `principal` over `months` at `annual_rate`.

    EMI = P r / (1 - (1 + r)^-n), with r the monthly rate; straight-line at r = 0.
    Works on scalars and arrays alike.
    """
    p = np.asarray(principal, dtype=float)
    r = np.asarray(annual_rate, dtype=float) / 12.0
    n = np.asarray(months, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        payment = np.where(r > 0, p * r / (1.0 - (1.0 + r) ** -n), p / n)
    return payment.item() if payment.ndim == 0 else payment


def _ids(loan_id: np.ndarray, mask: np.ndarray) -> str:
    bad = [str(x) for x in loan_id[mask]]
    more = f" and {len(bad) - SHOWN} more" if len(bad) > SHOWN else ""
    return ", ".join(bad[:SHOWN]) + more


class LoanTape:
    """A validated loan book held as parallel arrays, one entry per loan."""

    __slots__ = FIELDS

    def __init__(self, *, emi_tolerance: float = EMI_TOLERANCE, **columns: Any) -> None:
        missing = [f for f in FIELDS if f not in columns]
        extra = [c for c in columns if c not in FIELDS]
        if missing or extra:
            raise EngineError(f"loan tape columns: missing {missing}, unexpected {extra}")
        lengths = {f: len(columns[f]) for f in FIELDS}
        if len(set(lengths.values())) != 1:
            raise EngineError(f"loan tape columns have mismatched lengths: {lengths}")
        if lengths["loan_id"] == 0:
            raise EngineError("the loan tape is empty")

        self.loan_id: np.ndarray = np.asarray(columns["loan_id"], dtype=str)
        self.borrower_id: np.ndarray = np.asarray(columns["borrower_id"], dtype=str)
        self.product: np.ndarray = self._enum(Product, columns["product"], "product")
        self.benchmark: np.ndarray = self._enum(Benchmark, columns["benchmark"], "benchmark")
        self.rate: np.ndarray = np.asarray(columns["rate"], dtype=float)
        self.spread: np.ndarray = np.asarray(columns["spread"], dtype=float)
        self.outstanding: np.ndarray = np.asarray(columns["outstanding"], dtype=float)
        self.emi: np.ndarray = np.asarray(columns["emi"], dtype=float)
        self.borrower_income: np.ndarray = np.asarray(columns["borrower_income"], dtype=float)
        self.remaining_months: np.ndarray = np.asarray(columns["remaining_months"], dtype=int)
        self.reset_months: np.ndarray = np.asarray(columns["reset_months"], dtype=int)
        self.months_to_next_reset: np.ndarray = np.asarray(
            columns["months_to_next_reset"], dtype=int
        )
        self.borrower_age: np.ndarray = np.asarray(columns["borrower_age"], dtype=int)
        self.days_past_due: np.ndarray = np.asarray(columns["days_past_due"], dtype=int)
        self._validate(emi_tolerance)

    def _enum(self, kind: type[StrEnum], values: Sequence[Any], name: str) -> np.ndarray:
        allowed = {member.value for member in kind}
        raw = np.asarray([str(v).strip().lower() for v in values], dtype=str)
        bad = ~np.isin(raw, list(allowed))
        if bad.any():
            loan_ids = np.asarray(self.loan_id)
            raise EngineError(
                f"unknown {name} {sorted(set(raw[bad]))} for loans {_ids(loan_ids, bad)}; "
                f"expected one of {sorted(allowed)}"
            )
        return np.asarray([kind(v) for v in raw], dtype=object)

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping[str, Any]], *, emi_tolerance: float = EMI_TOLERANCE
    ) -> LoanTape:
        """Build a tape from one mapping per loan, such as rows read from a file."""
        if not records:
            raise EngineError("the loan tape is empty")
        columns = {f: [record.get(f) for record in records] for f in FIELDS}
        absent = [f for f, values in columns.items() if any(v is None for v in values)]
        if absent:
            raise EngineError(f"loan tape records lack values for {absent}")
        return cls(emi_tolerance=emi_tolerance, **columns)

    # ---- validation ----

    def _validate(self, emi_tolerance: float) -> None:
        ids = self.loan_id
        unique, counts = np.unique(ids, return_counts=True)
        if (counts > 1).any():
            raise EngineError(f"duplicate loan ids: {sorted(unique[counts > 1])[:SHOWN]}")

        numeric = np.column_stack(
            [self.rate, self.spread, self.outstanding, self.emi, self.borrower_income]
        )
        if not np.all(np.isfinite(numeric)):
            raise EngineError("the loan tape contains NaN or infinite values")

        checks: list[tuple[np.ndarray, str]] = [
            (self.outstanding <= 0, "outstanding principal must be positive"),
            (self.emi <= 0, "EMI must be positive"),
            ((self.rate <= 0) | (self.rate >= 0.5), "rate must lie between 0 and 50%"),
            (self.borrower_income <= 0, "borrower income must be positive"),
            (
                (self.borrower_age < 18) | (self.borrower_age > 100),
                "borrower age must lie between 18 and 100",
            ),
            (self.remaining_months < 1, "remaining months must be at least 1"),
            (self.days_past_due < 0, "days past due must not be negative"),
        ]
        fixed = self.benchmark == Benchmark.FIXED
        eblr = self.benchmark == Benchmark.EBLR
        internal = ~fixed & ~eblr
        checks += [
            (fixed & (self.reset_months != 0), "a fixed-rate loan has no reset (reset_months 0)"),
            (
                eblr & ((self.reset_months < 1) | (self.reset_months > MAX_EBLR_RESET_MONTHS)),
                f"an EBLR loan must reset at least every {MAX_EBLR_RESET_MONTHS} months",
            ),
            (
                internal
                & ((self.reset_months < 1) | (self.reset_months > MAX_INTERNAL_RESET_MONTHS)),
                f"an MCLR or base-rate loan resets within {MAX_INTERNAL_RESET_MONTHS} months",
            ),
            (
                ~fixed
                & (
                    (self.months_to_next_reset < 0)
                    | (self.months_to_next_reset >= self.reset_months)
                ),
                "the next reset must fall inside the reset period",
            ),
            (fixed & (self.months_to_next_reset != 0), "a fixed-rate loan has no next reset"),
        ]
        for mask, message in checks:
            if mask.any():
                raise EngineError(f"{message}: loans {_ids(ids, mask)}")

        # Only now, with every loan possible, can its EMI be checked against the
        # annuity; a loan whose EMI no longer covers the interest is exempt.
        expected = annuity_payment(self.outstanding, self.rate, self.remaining_months)
        gap = np.abs(self.emi - expected) / expected
        inconsistent = self.covers_interest & (gap > emi_tolerance)
        if inconsistent.any():
            raise EngineError(
                f"EMI differs from the annuity on outstanding, rate and remaining months by "
                f"more than {emi_tolerance:.0%}: loans {_ids(ids, inconsistent)}"
            )

        # One income per borrower.
        borrowers, index = np.unique(self.borrower_id, return_inverse=True)
        low = np.full(borrowers.size, np.inf)
        high = np.full(borrowers.size, -np.inf)
        np.minimum.at(low, index, self.borrower_income)
        np.maximum.at(high, index, self.borrower_income)
        differing = high - low > 1e-6
        if differing.any():
            raise EngineError(
                f"one borrower, two incomes: borrowers {sorted(borrowers[differing])[:SHOWN]}"
            )

    # ---- derived quantities ----

    def __len__(self) -> int:
        return int(self.loan_id.shape[0])

    def __repr__(self) -> str:
        return f"LoanTape(n={len(self):,}, borrowers={np.unique(self.borrower_id).size:,})"

    @property
    def is_floating(self) -> np.ndarray:
        return np.asarray(self.benchmark != Benchmark.FIXED, dtype=bool)

    @property
    def monthly_interest(self) -> np.ndarray:
        return np.asarray(self.outstanding * self.rate / 12.0)

    @property
    def covers_interest(self) -> np.ndarray:
        """False where the EMI no longer pays the month's interest: negative amortisation."""
        return np.asarray(self.emi > self.monthly_interest, dtype=bool)

    @property
    def borrower_dsti(self) -> np.ndarray:
        """Each loan's borrower's total EMIs times twelve over income, per loan row."""
        _, index = np.unique(self.borrower_id, return_inverse=True)
        total = np.bincount(index, weights=self.emi)
        return np.asarray(total[index] * 12.0 / self.borrower_income)
