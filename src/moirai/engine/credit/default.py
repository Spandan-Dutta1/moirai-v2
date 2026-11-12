"""
Probability of default from debt service (Stage 3.1).

The loan layer says what a policy move does to each borrower's payment. This
turns that into a probability, which is what a provision is built from.

The shape comes from Nier, Popa, Shamloo and Voinea, "Debt Service and
Default: Calibrating Macroprudential Policy Using Micro Data" (IMF WP/19/182),
estimated on Romanian loan-level credit-register data. Their finding is a
threshold, not a slope: default risk is flat in DSTI up to a point and rises
above it, by less and less as DSTI goes further.

    mortgages    flat below 50 percent DSTI; a 10 percentage point rise then
                 raises PD by about 16 percent in [50, 90), 14 percent in
                 [90, 120) and 8 percent above 120
    consumer     flat below 30 percent; then about 8 percent in [30, 50),
                 6 percent in [50, 70) and 3 percent above 70

Two things follow from that shape. The effect is multiplicative, so the bands
scale a base level rather than adding to it; and because it is flat below the
threshold, the base level is the PD of a borrower who is comfortably inside
their means, not the average of the book.

The base level is the one number the IMF paper cannot give, because it is a
property of the lender and the country rather than of the DSTI relationship.
It is calibrated instead: `calibrate_flat_pd` solves for the base level at
which a book reproduces a lender's disclosed Stage 1 provision coverage. The
default is the value this gives on the synthetic prime book's home loans
against Bajaj Housing Finance's disclosed 0.33 percent, at the RBI's fallback
LGD of 65 percent. Across the three lenders whose stage tables are published
the implied range is 0.26 percent (LIC Housing) to 0.45 percent (PNB
Housing), which is the honest band to run sensitivity over.

One assumption inside that: the calibration uses housing disclosures, because
no Indian lender publishes a comparable Stage 1 coverage for vehicle or
personal lending. The same base level is applied to those products, so their
absolute level is inherited rather than measured. Their shape is still the
published consumer one, and the shape is what a shock moves.

Direction, for India: Bandyopadhyay and Saha (2009), on 13,487 Indian housing
loans, find default rising in the EMI-to-income ratio, so the sign carries
over even though the magnitudes are Romanian. The level comes from Indian
disclosures, not from Romania.

Lifetime PD assumes a constant monthly hazard over the remaining term. That
ignores seasoning (default risk peaks a few years after origination), which
the tape cannot support until it carries months on book.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import LoanTape, Product

#: (band start, band end, relative rise in PD per 10pp of DSTI inside the band).
#: Mortgages: IMF WP/19/182, Table 3 bands, Table 4 column 7.
MORTGAGE_BANDS: tuple[tuple[float, float, float], ...] = (
    (0.50, 0.90, 0.16),
    (0.90, 1.20, 0.14),
    (1.20, np.inf, 0.08),
)

#: Consumer loans: IMF WP/19/182, Table 9 bands, Table 10 column 7.
CONSUMER_BANDS: tuple[tuple[float, float, float], ...] = (
    (0.30, 0.50, 0.08),
    (0.50, 0.70, 0.06),
    (0.70, np.inf, 0.03),
)

#: The step the published effects are measured over.
BAND_STEP = 0.10

#: Products secured on property, which follow the mortgage bands.
MORTGAGE_PRODUCTS = (Product.HOME, Product.LOAN_AGAINST_PROPERTY)


class DefaultParameters(BaseModel):
    """The base level and the regulatory floor. The bands are not parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flat_pd_12m: float = Field(
        default=0.00434,
        gt=0.0,
        le=1.0,
        description=(
            "CALIBRATED. The 12-month PD of a borrower inside the flat region "
            "(DSTI below 50 percent for a mortgage). Set so the synthetic prime "
            "book's home loans reproduce Bajaj Housing Finance's disclosed "
            "Stage 1 coverage of 0.33 percent at a 65 percent LGD. The three "
            "published lenders imply 0.26 to 0.45 percent. See calibrate_flat_pd."
        ),
    )
    pd_floor: float = Field(
        default=0.0003,
        ge=0.0,
        le=1.0,
        description=(
            "SOURCED. The RBI's final ECL directions (2026) set a minimum "
            "12-month PD of 0.03 percent."
        ),
    )

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def dsti_multiplier(dsti: Any, bands: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    """How much the flat-region PD is multiplied by at this DSTI.

    One below the first threshold, then compounding through each band at that
    band's published rate. Piecewise in the exponent, so the multiplier is
    continuous: a borrower just above a band edge is barely different from one
    just below it.
    """
    ratio = np.asarray(dsti, dtype=float)
    multiplier = np.ones_like(ratio)
    for start, end, rise in bands:
        inside = np.clip(ratio - start, 0.0, end - start)
        multiplier *= (1.0 + rise) ** (inside / BAND_STEP)
    return multiplier


def probability_of_default(
    dsti: Any,
    product: Any,
    parameters: DefaultParameters | None = None,
) -> np.ndarray:
    """The 12-month probability of default of each loan's borrower."""
    parameters = parameters or DefaultParameters()
    ratio = np.asarray(dsti, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise EngineError("DSTI must be finite and non-negative")
    kinds = np.asarray(product)
    if kinds.shape != ratio.shape:
        raise EngineError(f"product has shape {kinds.shape}, DSTI has {ratio.shape}")

    mortgage = np.isin(kinds, np.asarray(MORTGAGE_PRODUCTS))
    multiplier = np.where(
        mortgage,
        dsti_multiplier(ratio, MORTGAGE_BANDS),
        dsti_multiplier(ratio, CONSUMER_BANDS),
    )
    return np.maximum(parameters.flat_pd_12m * multiplier, parameters.pd_floor)


def lifetime_default(probability_12m: Any, remaining_months: Any) -> np.ndarray:
    """Default probability over the loan's remaining life, at a constant hazard.

    No seasoning: the tape has no months on book, so every month of remaining
    life carries the same risk. Published term structures peak two to four
    years after origination, so this understates a young book and overstates
    an old one.
    """
    annual = np.clip(np.asarray(probability_12m, dtype=float), 0.0, 1.0)
    years = np.maximum(np.asarray(remaining_months, dtype=float), 0.0) / 12.0
    return 1.0 - (1.0 - annual) ** years


def calibrate_flat_pd(
    dsti: Any,
    product: Any,
    outstanding: Any,
    target_coverage: float,
    loss_given_default: float,
) -> float:
    """The flat-region PD at which a book's Stage 1 coverage matches a target.

    Coverage is PD times LGD, weighted by exposure, and PD is proportional to
    the flat-region level, so this is one division rather than a search. The
    target is a lender's disclosed Stage 1 provision as a share of its Stage 1
    exposure.
    """
    if not 0.0 < target_coverage < 1.0 or not 0.0 < loss_given_default <= 1.0:
        raise EngineError("target coverage and LGD must lie in (0, 1]")
    kinds = np.asarray(product)
    ratio = np.asarray(dsti, dtype=float)
    weight = np.asarray(outstanding, dtype=float)
    if weight.sum() <= 0.0:
        raise EngineError("the book has no exposure to weight by")

    mortgage = np.isin(kinds, np.asarray(MORTGAGE_PRODUCTS))
    multiplier = np.where(
        mortgage,
        dsti_multiplier(ratio, MORTGAGE_BANDS),
        dsti_multiplier(ratio, CONSUMER_BANDS),
    )
    mean_multiplier = float(np.average(multiplier, weights=weight))
    return target_coverage / (loss_given_default * mean_multiplier)


def tape_default(
    tape: LoanTape, parameters: DefaultParameters | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The 12-month and lifetime PD of every loan on a tape, as it stands."""
    annual = probability_of_default(tape.borrower_dsti, tape.product, parameters)
    return annual, lifetime_default(annual, tape.remaining_months)
