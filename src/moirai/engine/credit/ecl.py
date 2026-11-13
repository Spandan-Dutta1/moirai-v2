"""
Stages and expected credit loss (Stages 3.2 and 3.3).

What a lender reports is not a probability but a provision, and the rules for
turning one into the other are the RBI's final ECL directions of 27 April
2026, in force from 1 April 2027. They fix most of what this module does:

    staging    Stage 2 is presumed above 30 days past due; the presumption is
               rebuttable only on board-approved evidence. Stage 3 is
               credit-impaired, with the 90-day NPA rule as the backstop,
               applied at borrower level, so all of a borrower's loans move
               together.
    horizon    12-month expected loss in Stage 1, lifetime in Stages 2 and 3.
    loss rates where a lender cannot estimate its own, 65 percent secured and
               70 percent unsecured.
    floors     a provision may never fall below the floor for that stage and
               asset class. For individual housing loans: 0.25 percent in
               Stage 1 and 1.5 percent in Stage 2 - a six-fold step for every
               loan that migrates.
    scenarios  at least three, probability-weighted.

Stage 3 carries no modelled PD: default has happened, so the expected loss is
the whole loss given default.

Two things here are not in the directions and are marked as such. The
significant-increase-in-credit-risk test has no regulatory number anywhere -
the RBI, the IASB and the EBA all leave it to the lender - so the doubling
convention is used and exposed as a parameter to vary. And the Stage 3 floors
rise with time in Stage 3, which the tape cannot observe; it is approximated
from days past due beyond 90.

Where a provision is set by the floor rather than by the model, the result
says so. That matters more than it sounds: Indian housing lenders disclose
Stage 1 coverage of 0.17 to 0.34 percent against a 0.25 percent floor, so a
model of Stage 1 losses may be describing a number the lender never books.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.engine.credit.loan_tape import LoanTape, Product


class Stage(IntEnum):
    """The three stages of the RBI's directions, matching IFRS 9."""

    ONE = 1
    TWO = 2
    THREE = 3


#: Secured on an asset the lender can realise, for the fallback LGD.
SECURED_PRODUCTS = (Product.HOME, Product.LOAN_AGAINST_PROPERTY, Product.VEHICLE)

#: Individual housing loans, which have their own floors (para 82(4)(xi)(a)).
#: A loan against property is secured on a house but is not a housing loan, so
#: it sits with secured retail here even though it follows the mortgage default
#: bands in `default.py`: the regulatory class and the risk shape differ.
HOUSING_PRODUCTS = (Product.HOME,)

#: Floors as a share of exposure: (Stage 1, Stage 2).
HOUSING_FLOORS = (0.0025, 0.015)
SECURED_RETAIL_FLOORS = (0.0040, 0.050)
UNSECURED_RETAIL_FLOORS = (0.0100, 0.050)

#: Stage 3 floors by completed years in Stage 3, housing and everything else.
HOUSING_STAGE_THREE_FLOORS = (0.10, 0.20, 0.30, 0.40, 1.00)
OTHER_STAGE_THREE_FLOORS = (0.25, 1.00)


class EclParameters(BaseModel):
    """How a probability becomes a provision. Defaults name their source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lgd_secured: float = Field(
        default=0.65,
        gt=0.0,
        le=1.0,
        description=(
            "SOURCED. The RBI's fallback loss given default for secured "
            "exposures where the lender cannot estimate its own."
        ),
    )
    lgd_unsecured: float = Field(
        default=0.70, gt=0.0, le=1.0, description="SOURCED. The RBI's unsecured fallback."
    )
    sicr_pd_multiple: float = Field(
        default=2.0,
        gt=1.0,
        le=10.0,
        description=(
            "ASSUMED. No regulator sets a number for a significant increase in "
            "credit risk. Doubling is the common industry convention; vary it."
        ),
    )
    sicr_dsti_threshold: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description=(
            "ASSUMED, optional. A qualitative SICR criterion, which the "
            "directions permit alongside the quantitative one: crossing the "
            "debt-service level above which the IMF's loan-level work finds "
            "default risk starts rising (50 percent for a mortgage). None "
            "leaves SICR to the probability test alone."
        ),
    )
    dpd_stage_two: int = Field(
        default=30, ge=1, description="SOURCED. The RBI's rebuttable presumption."
    )
    dpd_stage_three: int = Field(
        default=90, ge=1, description="SOURCED. The NPA rule, as the backstop."
    )
    apply_floors: bool = Field(
        default=True,
        description=(
            "SWITCH, not a model value. Off to see the modelled loss without the "
            "regulatory floor, which is how the floor's bite is measured."
        ),
    )

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EclResult(BaseModel):
    """Per loan: its stage, the loss expected of it, and what was booked."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    parameters: EclParameters
    stage: Any = Field(description="(n,) Stage 1, 2 or 3.")
    probability: Any = Field(description="(n,) PD applied: 12-month, lifetime, or 1.")
    loss_given_default: Any = Field(description="(n,) LGD applied.")
    exposure: Any = Field(description="(n,) exposure at default: the outstanding balance.")
    modelled: Any = Field(description="(n,) PD x LGD x EAD.")
    floor: Any = Field(description="(n,) the regulatory minimum provision.")
    provision: Any = Field(description="(n,) what is booked: the greater of the two.")

    @property
    def floor_bound(self) -> np.ndarray:
        """Loans whose provision is set by the floor, not by the model."""
        return np.asarray(self.floor) > np.asarray(self.modelled)

    def total(self) -> float:
        return float(np.sum(self.provision))

    def summary(self) -> dict[str, float]:
        exposure = np.asarray(self.exposure)
        provision = np.asarray(self.provision)
        stage = np.asarray(self.stage)
        total_exposure = float(exposure.sum())
        out = {
            "exposure": total_exposure,
            "provision": float(provision.sum()),
            "coverage": float(provision.sum() / total_exposure) if total_exposure else 0.0,
            "modelled": float(np.sum(self.modelled)),
            "share_floor_bound": float(np.average(self.floor_bound, weights=exposure)),
        }
        for value in Stage:
            inside = stage == int(value)
            share = float(exposure[inside].sum() / total_exposure) if total_exposure else 0.0
            out[f"share_stage_{int(value)}"] = share
            out[f"coverage_stage_{int(value)}"] = (
                float(provision[inside].sum() / exposure[inside].sum())
                if exposure[inside].sum()
                else 0.0
            )
        return out


def assign_stages(
    tape: LoanTape,
    probability_now: Any,
    probability_reference: Any,
    parameters: EclParameters | None = None,
    *,
    dsti_now: Any = None,
    dsti_reference: Any = None,
) -> np.ndarray:
    """Stage every loan: arrears first, then a significant increase in risk.

    `probability_reference` is the PD the loan started the run at. The
    accounting test compares against origination, which a tape without months
    on book cannot give, so this is a scenario-conditional test: it answers
    "whose risk did this shock double", not "whose risk has doubled since the
    loan was written".

    With `sicr_dsti_threshold` set, a loan whose borrower is carried across
    that debt-service level by the shock is also Stage 2, even if its
    probability has not doubled. Both debt-service arrays are then required:
    crossing is what counts, so a borrower already above the level does not
    migrate on account of having stayed there.
    """
    parameters = parameters or EclParameters()
    now = np.asarray(probability_now, dtype=float)
    reference = np.asarray(probability_reference, dtype=float)
    if now.shape != reference.shape or now.shape != (len(tape),):
        raise EngineError("the probabilities must be one per loan on the tape")

    past_due = np.asarray(tape.days_past_due)
    impaired = past_due > parameters.dpd_stage_three

    # Borrower level: one impaired loan takes the borrower's whole exposure.
    _, index = np.unique(tape.borrower_id, return_inverse=True)
    impaired_borrower = np.bincount(index, weights=impaired.astype(float)) > 0
    impaired = impaired_borrower[index]

    sicr = now >= parameters.sicr_pd_multiple * reference
    if parameters.sicr_dsti_threshold is not None:
        if dsti_now is None or dsti_reference is None:
            raise EngineError("a debt-service SICR threshold needs dsti_now and dsti_reference")
        level = parameters.sicr_dsti_threshold
        crossed = (np.asarray(dsti_now, dtype=float) > level) & (
            np.asarray(dsti_reference, dtype=float) <= level
        )
        sicr = sicr | crossed
    deteriorated = (past_due > parameters.dpd_stage_two) | sicr

    return np.where(
        impaired, int(Stage.THREE), np.where(deteriorated, int(Stage.TWO), int(Stage.ONE))
    ).astype(np.int8)


def _floor_rate(tape: LoanTape, stage: np.ndarray) -> np.ndarray:
    """The regulatory minimum provision rate for each loan's stage and class."""
    housing = np.isin(np.asarray(tape.product), np.asarray(HOUSING_PRODUCTS))
    secured = np.isin(np.asarray(tape.product), np.asarray(SECURED_PRODUCTS))

    stage_one, stage_two = (
        np.select(
            [housing, secured],
            [HOUSING_FLOORS[column], SECURED_RETAIL_FLOORS[column]],
            default=UNSECURED_RETAIL_FLOORS[column],
        )
        for column in (0, 1)
    )

    # Years in Stage 3, approximated from arrears beyond the 90-day backstop.
    years = np.floor(np.maximum(np.asarray(tape.days_past_due) - 90, 0) / 365.0).astype(int)
    housing_three = np.take(
        np.asarray(HOUSING_STAGE_THREE_FLOORS),
        np.minimum(years, len(HOUSING_STAGE_THREE_FLOORS) - 1),
    )
    other_three = np.take(
        np.asarray(OTHER_STAGE_THREE_FLOORS),
        np.minimum(years, len(OTHER_STAGE_THREE_FLOORS) - 1),
    )
    stage_three = np.where(housing, housing_three, other_three)

    return np.select(
        [stage == int(Stage.ONE), stage == int(Stage.TWO)],
        [stage_one, stage_two],
        default=stage_three,
    )


def expected_credit_loss(
    tape: LoanTape,
    stage: Any,
    probability_12m: Any,
    probability_lifetime: Any,
    parameters: EclParameters | None = None,
) -> EclResult:
    """Provision every loan: the modelled loss, or the floor if it is higher."""
    parameters = parameters or EclParameters()
    stages = np.asarray(stage).astype(int)
    if stages.shape != (len(tape),):
        raise EngineError("one stage per loan is required")
    if not np.isin(stages, [int(s) for s in Stage]).all():
        raise EngineError("stages must be 1, 2 or 3")

    annual = np.asarray(probability_12m, dtype=float)
    lifetime = np.asarray(probability_lifetime, dtype=float)
    secured = np.isin(np.asarray(tape.product), np.asarray(SECURED_PRODUCTS))
    loss = np.where(secured, parameters.lgd_secured, parameters.lgd_unsecured)
    exposure = np.asarray(tape.outstanding, dtype=float)

    applied = np.select(
        [stages == int(Stage.ONE), stages == int(Stage.TWO)],
        [annual, lifetime],
        default=1.0,  # Stage 3 is credit-impaired: the loss is LGD on the exposure.
    )
    modelled = applied * loss * exposure
    floor = (
        _floor_rate(tape, stages) * exposure if parameters.apply_floors else np.zeros_like(exposure)
    )

    return EclResult(
        parameters=parameters,
        stage=stages.astype(np.int8),
        probability=applied,
        loss_given_default=loss,
        exposure=exposure,
        modelled=modelled,
        floor=floor,
        provision=np.maximum(modelled, floor),
    )


def weighted_provision(results: dict[str, EclResult], weights: dict[str, float]) -> float:
    """The probability-weighted provision across scenarios.

    The RBI requires at least three. It sets no weights, so they are an input
    to the run and belong in its ledger.
    """
    if set(results) != set(weights):
        raise EngineError(f"scenarios {sorted(results)} do not match weights {sorted(weights)}")
    if len(weights) < 3:
        raise EngineError("the RBI's directions require at least three scenarios")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise EngineError(f"the scenario weights sum to {total}, not 1")
    if any(w < 0.0 for w in weights.values()):
        raise EngineError("a scenario weight cannot be negative")
    return float(sum(results[name].total() * weight for name, weight in weights.items()))
