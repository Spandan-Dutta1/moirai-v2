"""
Moment matching for the household population.

A simulation parameter chosen because it produced the expected answer is
not a calibration, it is a conclusion written backwards. This module makes
that impossible to do quietly: targets are declared with their sources
before any fitting happens, every gap is reported including the ones that
were not closed, and each target carries a confidence flag so a number
somebody guessed cannot be mistaken for a number somebody measured.

The confidence flag is the important part. Some targets here come straight
from AIDIS 2019; others are derived arithmetic on published aggregates;
others are placeholders that nobody has sourced yet. All three appear in
the report, labelled. A calibration that reports only the targets it hit,
or that does not say where its targets came from, is a much weaker claim
than one that shows its failures.

Two limitations that apply to every wealth target here:

  * AIDIS measures total assets, which for Indian households are dominated
    by land and buildings. The `wealth` field in this model is the liquid
    buffer a household can actually spend from, which is a smaller and far
    more concentrated quantity. AIDIS asset totals therefore cannot be
    used as a wealth target directly.

  * Household surveys understate the top of the wealth distribution.
    Wealthy households respond less often and under-report financial
    assets, so a survey-derived Gini is a lower bound.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.economy.households import (
    Population,
    PopulationParameters,
    generate_population,
)

log = get_logger(__name__)

#: Liquid wealth below this many months of income counts as hand-to-mouth.
HAND_TO_MOUTH_MONTHS = 0.5


class Confidence(StrEnum):
    """Where a target value came from.

    This exists so that an unsourced placeholder cannot pass for a
    measured quantity simply by sitting next to one in a table.
    """

    #: Taken directly from a named published statistic.
    SOURCED = "sourced"
    #: Arithmetic on published statistics, with the derivation stated.
    DERIVED = "derived"
    #: Nobody has sourced this yet. Treated as provisional everywhere.
    UNSOURCED = "unsourced"


class Moment(BaseModel):
    """One observable statistic the population should reproduce."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    target: float
    tolerance: float = Field(gt=0, description="Absolute tolerance for a pass.")
    source: str = Field(min_length=1, description="Where the target came from.")
    confidence: Confidence
    note: str = ""

    def measure(self, population: Population) -> float:
        return MEASUREMENTS[self.name](population)

    def gap(self, population: Population) -> float:
        return self.measure(population) - self.target

    def passes(self, population: Population) -> bool:
        return abs(self.gap(population)) <= self.tolerance

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "tolerance": self.tolerance,
            "source": self.source,
            "confidence": self.confidence.value,
            "note": self.note,
        }


# ---- measurements ---------------------------------------------------------
#
# Each function maps a population to one scalar. Keeping them in a registry
# rather than as methods means a target can be declared by name, and a
# target naming a measurement that does not exist fails loudly at
# construction rather than silently at comparison time.


def _share_indebted(population: Population) -> float:
    return float(population.is_indebted.mean())


def _share_hand_to_mouth(population: Population) -> float:
    """Households with less than half a month of income in liquid wealth.

    These drive most of the aggregate consumption response, because they
    spend a windfall immediately rather than smoothing it. A population
    with too few of them will understate how much a shock moves spending,
    however well its other moments match.
    """
    monthly_income = np.maximum(population.income / 12.0, 1.0)
    buffer_months = np.maximum(population.wealth, 0.0) / monthly_income
    return float((buffer_months < HAND_TO_MOUTH_MONTHS).mean())


def _income_gini(population: Population) -> float:
    return population.gini(population.income)


def _wealth_gini(population: Population) -> float:
    return population.gini(np.maximum(population.wealth, 0.0))


def _wealth_top_10_share(population: Population) -> float:
    return population.top_share(np.maximum(population.wealth, 0.0), 0.1)


def _median_debt_to_income_of_borrowers(population: Population) -> float:
    borrowers = population.is_indebted
    if not borrowers.any():
        return 0.0
    return float(np.median(population.debt_to_income[borrowers]))


def _mean_debt_to_mean_income(population: Population) -> float:
    """Aggregate household debt as a multiple of aggregate income.

    Comparable to published aggregates, which report totals rather than
    household-level ratios.
    """
    total_income = population.income.sum()
    if total_income <= 0:
        return 0.0
    return float(population.debt.sum() / total_income)


def _employment_rate(population: Population) -> float:
    return float(population.is_employed.mean())


def _share_rate_exposed(population: Population) -> float:
    return float(population.is_rate_exposed.mean())


def _indebtedness_gradient(population: Population) -> float:
    """Ratio of indebtedness in the top income quintile to the bottom.

    Borrowing is hump-shaped in income: the poorest are credit
    constrained, the middle and upper-middle borrow most. A ratio near one
    means the model has made rate exposure uniform across the income
    distribution, which removes the mechanism by which monetary policy
    lands unevenly.
    """
    quintile = population.quantile_groups(population.income, 5)
    bottom = population.is_indebted[quintile == 0].mean()
    top = population.is_indebted[quintile == 4].mean()
    return float(top / bottom) if bottom > 0 else float("inf")


MEASUREMENTS: dict[str, Callable[[Population], float]] = {
    "share_indebted": _share_indebted,
    "share_hand_to_mouth": _share_hand_to_mouth,
    "income_gini": _income_gini,
    "wealth_gini": _wealth_gini,
    "wealth_top_10_share": _wealth_top_10_share,
    "median_debt_to_income_of_borrowers": _median_debt_to_income_of_borrowers,
    "mean_debt_to_mean_income": _mean_debt_to_mean_income,
    "employment_rate": _employment_rate,
    "share_rate_exposed": _share_rate_exposed,
    "indebtedness_gradient": _indebtedness_gradient,
}


# ---- declared target sets -------------------------------------------------

AIDIS_2019 = "AIDIS 2019 (NSS 77th round), reference date 30 June 2018"

URBAN_INDIA_MOMENTS: tuple[Moment, ...] = (
    Moment(
        name="share_indebted",
        target=0.224,
        tolerance=0.03,
        source=AIDIS_2019,
        confidence=Confidence.SOURCED,
        note=(
            "Incidence of indebtedness, urban India. Rural is 35 percent; "
            "this model is calibrated to urban households."
        ),
    ),
    Moment(
        name="mean_debt_to_mean_income",
        target=0.50,
        tolerance=0.25,
        source=f"derived from {AIDIS_2019} and NSS consumption rounds",
        confidence=Confidence.DERIVED,
        note=(
            "AIDIS reports mean urban household debt of Rs 1,20,336. Dividing "
            "by an assumed mean urban household income gives this ratio, so it "
            "inherits the uncertainty of that income figure and the tolerance "
            "is wide accordingly."
        ),
    ),
    Moment(
        name="employment_rate",
        target=0.494,
        tolerance=0.05,
        source="PLFS 2023-24 (July 2023 - June 2024), urban WPR, age 15 and above",
        confidence=Confidence.SOURCED,
        note=(
            "Worker population ratio for urban India, persons aged 15 and "
            "above, which rose from 43.9 percent in 2017-18. The all-ages "
            "urban figure is lower at 47.6 percent for calendar 2024; the "
            "15-plus measure is used because this model's population starts "
            "at 18."
        ),
    ),
    Moment(
        name="wealth_gini",
        target=0.75,
        tolerance=0.08,
        source="not verified against a published estimate",
        confidence=Confidence.UNSOURCED,
        note=(
            "AIDIS computes a Gini of asset distribution but the value was "
            "not obtained. Note also that AIDIS assets are dominated by land "
            "and buildings, whereas this field models liquid wealth, which is "
            "more concentrated. Survey data understates the top tail in any "
            "case, so any survey-derived figure is a lower bound."
        ),
    ),
    Moment(
        name="income_gini",
        target=0.50,
        tolerance=0.08,
        source="not verified against a published estimate",
        confidence=Confidence.UNSOURCED,
        note=(
            "Widely cited Indian Gini figures are usually consumption based "
            "and substantially lower than income based ones. The two are not "
            "interchangeable and this target has not been checked."
        ),
    ),
    Moment(
        name="share_hand_to_mouth",
        target=0.374,
        tolerance=0.12,
        source=(
            "Gupta, Pizzolon and Singh (2025), Identifying Hand-to-Mouth "
            "Households: Evidence from India, Table 5"
        ),
        confidence=Confidence.SOURCED,
        note=(
            "Gupta, Pizzolon and Singh (2025 working paper), Table 5, monthly "
            "pay period: total hand-to-mouth share 0.374 (poor 0.052, wealthy "
            "0.322) from AIDIS 2019 with income imputed from CPHS. The monthly "
            "row's threshold of half a month's income matches this model's "
            "definition. The total is used because the model has a single "
            "liquid wealth field, so poor and wealthy hand-to-mouth behave "
            "identically here. Limitations: the estimate is all-India, while "
            "this model is calibrated to urban households, and no urban split "
            "is reported; income is imputed; and the monthly figure is a "
            "robustness row for one imputation method. For comparison, KVW "
            "estimate about one-third for the US at a one-week threshold. "
            "Structural limitation: the model has no illiquid wealth, and "
            "liquid wealth is positively correlated with income, so its "
            "hand-to-mouth households are concentrated at low incomes, "
            "resembling the poor hand-to-mouth. In the data, 86 percent of "
            "the total (0.322 of 0.374) are wealthy hand-to-mouth, who hold "
            "illiquid assets and are not concentrated at low incomes. The "
            "model can match the share but probably misplaces these "
            "households in the income and debt distribution, which affects "
            "how hand-to-mouth status overlaps with rate exposure. The "
            "remaining gap in the share itself is a calibration gap, not a "
            "structural one: log_wealth_mean was fitted to the earlier 0.40 "
            "target and could close it by re-calibration. It is deliberately "
            "not re-fitted, because ADR 008 records it as a fitted-to-target "
            "parameter and fitting it to a second target would compound that "
            "circularity. The gap against a published figure is left visible."
        ),
    ),
    Moment(
        name="indebtedness_gradient",
        target=7.0,
        tolerance=4.0,
        source="qualitative: credit access rises steeply with income",
        confidence=Confidence.UNSOURCED,
        note=(
            "AIDIS reports indebtedness by asset decile rather than by income "
            "quintile, so this is a direction rather than a level. A ratio "
            "near one would mean the model has no credit constraint at the "
            "bottom, which is the failure this target exists to catch."
        ),
    ),
)


class MomentResult(BaseModel):
    """One target compared against one population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    moment: Moment
    measured: float
    gap: float
    passed: bool

    @property
    def relative_gap(self) -> float:
        scale = abs(self.moment.target)
        return float("inf") if scale == 0 else self.gap / scale


class CalibrationReport(BaseModel):
    """How well a population reproduces its declared targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[MomentResult, ...]
    parameters: PopulationParameters

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failures(self) -> tuple[MomentResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def sourced_failures(self) -> tuple[MomentResult, ...]:
        """Failures against targets that came from a real published figure.

        These matter more than failures against placeholders: missing an
        unsourced target may mean the target is wrong.
        """
        return tuple(
            r for r in self.failures if r.moment.confidence is Confidence.SOURCED
        )

    def loss(self) -> float:
        """Sum of squared gaps scaled by tolerance.

        Scaling by tolerance makes moments in different units comparable,
        so a Gini and a debt ratio contribute on the same footing.
        """
        return float(
            sum((r.gap / r.moment.tolerance) ** 2 for r in self.results)
        )

    def summary(self) -> str:
        return (
            f"{self.n_passed}/{len(self.results)} moments matched, "
            f"loss {self.loss():.3f}, "
            f"{len(self.sourced_failures)} sourced target(s) missed"
        )

    def table(self) -> str:
        """Readable comparison, with the confidence of each target shown."""
        lines = [
            f"{'moment':<36} {'target':>9} {'measured':>10} {'gap':>9}  "
            f"{'':<4} confidence",
            "-" * 92,
        ]
        for result in self.results:
            mark = "ok" if result.passed else "MISS"
            lines.append(
                f"{result.moment.name:<36} {result.moment.target:>9.3f} "
                f"{result.measured:>10.3f} {result.gap:>+9.3f}  "
                f"{mark:<4} {result.moment.confidence.value}"
            )
        return "\n".join(lines)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "loss": round(self.loss(), 6),
            "n_passed": self.n_passed,
            "n_moments": len(self.results),
            "parameters": self.parameters.to_ledger_dict(),
            "results": [
                {
                    "moment": r.moment.to_ledger_dict(),
                    "measured": round(r.measured, 6),
                    "gap": round(r.gap, 6),
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def evaluate(
    population: Population,
    moments: Sequence[Moment] = URBAN_INDIA_MOMENTS,
) -> CalibrationReport:
    """Compare a population against its declared targets."""
    if not moments:
        raise EngineError("at least one moment is required")
    if population.parameters is None:
        raise EngineError(
            "population has no parameters recorded, so a calibration report "
            "could not be attributed to a calibration"
        )

    results = tuple(
        MomentResult(
            moment=moment,
            measured=moment.measure(population),
            gap=moment.gap(population),
            passed=moment.passes(population),
        )
        for moment in moments
    )

    report = CalibrationReport(results=results, parameters=population.parameters)
    log.info(
        "calibration_evaluated",
        n_passed=report.n_passed,
        n_moments=len(results),
        loss=round(report.loss(), 4),
        sourced_failures=[r.moment.name for r in report.sourced_failures],
    )
    return report


def search(
    grid: dict[str, Sequence[float]],
    *,
    base: PopulationParameters | None = None,
    moments: Sequence[Moment] = URBAN_INDIA_MOMENTS,
    n_households: int = 50_000,
) -> tuple[PopulationParameters, CalibrationReport, list[tuple[dict, float]]]:
    """Grid search for the parameters that best reproduce the targets.

    Returns the best parameters, their report, and every evaluation so the
    shape of the loss surface can be inspected. A flat surface means a
    parameter is not identified by these moments, which is worth knowing
    before quoting its fitted value as though it meant something.

    Grid search rather than an optimiser: the parameter count is small, the
    surface is not smooth, and an exhaustive result is easier to defend
    than one that depends on a starting point.
    """
    if not grid:
        raise EngineError("search requires at least one parameter to vary")

    base = base or PopulationParameters()
    for name in grid:
        if not hasattr(base, name):
            raise EngineError(f"{name!r} is not a population parameter")

    names = list(grid)
    trials: list[tuple[dict, float]] = []
    best_loss = float("inf")
    best: tuple[PopulationParameters, CalibrationReport] | None = None

    def recurse(index: int, chosen: dict[str, float]) -> None:
        nonlocal best_loss, best
        if index == len(names):
            candidate = base.model_copy(
                update={**chosen, "n_households": n_households}
            )
            report = evaluate(generate_population(candidate), moments)
            loss = report.loss()
            trials.append((dict(chosen), loss))
            if loss < best_loss:
                best_loss, best = loss, (candidate, report)
            return

        name = names[index]
        for value in grid[name]:
            recurse(index + 1, {**chosen, name: value})

    recurse(0, {})
    assert best is not None

    log.info(
        "calibration_search_completed",
        n_trials=len(trials),
        best_loss=round(best_loss, 4),
        best_parameters={name: getattr(best[0], name) for name in names},
    )
    return best[0], best[1], trials