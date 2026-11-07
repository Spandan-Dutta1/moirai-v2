"""
From a loan tape to a Population (Stage 2.4).

The household layer runs on a Population: parallel arrays of age, income,
wealth, debt, the floating-rate flag, employment, expected inflation and risk
aversion. A loan tape observes some of these and not others, and the adapter
keeps the two apart:

    observed   income and age (the tape's); debt, the borrower's outstanding
               principal across every loan; floating, when most of that debt
               by amount is on a floating benchmark
    imputed    employment: employed below the retirement age, retired above it,
               since the borrower is servicing an EMI
               wealth: drawn from the calibrated distribution of
               generate_population, conditional on the borrower's actual
               income through the same income-wealth correlation, with the
               same age profile
               expected inflation and risk aversion: the calibrated draws

The imputed fields are named on the result, so nothing downstream can present
imputed wealth as observed. The draws are seeded by the PopulationParameters,
so the same tape and calibration give the same population.

Behaviour amortises debt over one maturity for everyone; the adapter reports
the book's outstanding-weighted remaining term, so a run on a real book can
use the book's own.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.engine.credit.loan_tape import LoanTape
from moirai.engine.economy.households import EmploymentStatus, Population, PopulationParameters

#: The Population fields a loan tape does not observe.
IMPUTED_FIELDS = ("employment", "wealth", "expected_inflation", "risk_aversion")


class TapePopulation(BaseModel):
    """A Population built from a loan tape, one household per borrower."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    population: Population
    borrower_id: tuple[str, ...] = Field(description="The borrower each household is.")
    floating_share: tuple[float, ...] = Field(
        description="Share of each borrower's debt, by amount, on a floating benchmark."
    )
    imputed: tuple[str, ...] = IMPUTED_FIELDS
    remaining_years: float = Field(
        description="Outstanding-weighted remaining term of the book, in years."
    )


def to_population(tape: LoanTape, parameters: PopulationParameters | None = None) -> TapePopulation:
    """One household per borrower: observed where the tape observes, imputed otherwise."""
    parameters = parameters or PopulationParameters()
    borrowers, index = np.unique(tape.borrower_id, return_inverse=True)
    n = borrowers.size

    debt = np.bincount(index, weights=tape.outstanding, minlength=n)
    floating_debt = np.bincount(
        index, weights=tape.outstanding * tape.is_floating, minlength=n
    )
    floating_share = floating_debt / debt
    first = np.zeros(n, dtype=int)
    first[index[::-1]] = np.arange(len(tape))[::-1]  # each borrower's first loan
    income = tape.borrower_income[first].astype(float)
    age = tape.borrower_age[first].astype(np.int16)

    employment = np.where(
        age >= parameters.retirement_age, EmploymentStatus.RETIRED, EmploymentStatus.EMPLOYED
    ).astype(np.int8)

    rng = np.random.default_rng(parameters.seed)
    rho = parameters.income_wealth_correlation
    income_z = (np.log(income) - parameters.log_income_mean) / parameters.log_income_sd
    wealth_z = rho * income_z + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    age_position = np.clip(
        (age - parameters.min_age) / (parameters.max_age - parameters.min_age), 0.0, 1.0
    )
    wealth = np.exp(parameters.log_wealth_mean + parameters.log_wealth_sd * wealth_z) * (
        0.35 + 0.9 * age_position
    )
    expected_inflation = rng.normal(
        parameters.initial_expected_inflation, parameters.expected_inflation_sd, size=n
    )
    risk_aversion = np.maximum(
        rng.normal(parameters.risk_aversion_mean, parameters.risk_aversion_sd, size=n), 0.1
    )

    population = Population(
        age=age,
        income=income,
        wealth=wealth,
        debt=debt,
        debt_is_floating=floating_share >= 0.5,
        employment=employment,
        expected_inflation=expected_inflation,
        risk_aversion=risk_aversion,
        parameters=parameters,
    )
    remaining_years = float(
        np.sum(tape.outstanding * tape.remaining_months) / np.sum(tape.outstanding) / 12.0
    )
    return TapePopulation(
        population=population,
        borrower_id=tuple(str(b) for b in borrowers),
        floating_share=tuple(float(s) for s in floating_share),
        remaining_years=remaining_years,
    )
