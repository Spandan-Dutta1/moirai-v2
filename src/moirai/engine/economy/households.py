"""
Heterogeneous household population.

Stored as a Structure of Arrays rather than as objects. A Python object
carries roughly fifty bytes of overhead and every attribute access is a
dictionary lookup, so a million households as objects is a slow simulation.
A million households as a handful of contiguous numpy arrays vectorises,
fits in cache, and runs three orders of magnitude faster. The architecture
document specifies this, and it is the single decision that makes a
population of this size feasible at all.

The reason to model households heterogeneously rather than as a
representative agent is that policy does not land evenly. A rate rise
transfers from floating-rate borrowers to savers. A representative
household nets those effects to nearly zero and reports that monetary
policy barely matters, which is exactly the wrong conclusion. The
distribution is the finding.

Calibration note: wealth is far more unequal than income. In India the top
decile holds roughly two thirds of wealth against a little over half of
income. A population where everyone has a similar buffer would badly
understate how differently a shock lands, so the two are drawn with
different dispersions and correlated rather than independently.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger

log = get_logger(__name__)

#: Below this, distributional statistics are too noisy to be meaningful.
MIN_HOUSEHOLDS = 100


class EmploymentStatus(IntEnum):
    """Stored as small integers so the array stays compact and vectorises."""

    UNEMPLOYED = 0
    EMPLOYED = 1
    RETIRED = 2
    OUT_OF_LABOUR_FORCE = 3


class PopulationParameters(BaseModel):
    """Calibration for generating a synthetic population.

    Defaults are loosely calibrated to urban India. They are starting
    points for a simulation, not estimates: any published result should
    state which parameters produced it, which is why this object is frozen
    and serialisable into the run ledger.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_households: int = Field(default=10_000, ge=MIN_HOUSEHOLDS, le=100_000_000)

    # ---- age ----
    min_age: int = Field(default=18, ge=15, le=40)
    max_age: int = Field(default=85, ge=50, le=120)
    retirement_age: int = Field(default=60, ge=45, le=80)

    # ---- income: log-normal ----
    log_income_mean: float = Field(default=12.5, description="Log of annual income.")
    log_income_sd: float = Field(default=0.85, gt=0, le=3.0)

    # ---- wealth: log-normal, wider, correlated with income ----
    log_wealth_mean: float = Field(default=12.0)
    log_wealth_sd: float = Field(default=1.6, gt=0, le=4.0)
    income_wealth_correlation: float = Field(
        default=0.85,
        ge=-1.0,
        le=1.0,
        description=(
            "High: the bottom of the income distribution holds very little "
            "liquid wealth. A weak correlation produces poor households with "
            "large savings buffers, which removes hand-to-mouth behaviour and "
            "makes rate rises look progressive."
        ),
    )

    # ---- debt ----
    share_with_debt: float = Field(default=0.45, ge=0.0, le=1.0)
    debt_to_income_mean: float = Field(default=1.8, gt=0)
    debt_to_income_sd: float = Field(default=1.0, gt=0)
    share_floating_rate: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Indian mortgages are predominantly floating, unlike the US.",
    )
    debt_peak_income_rank: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Income percentile where borrowing propensity peaks.",
    )
    debt_rank_spread: float = Field(
        default=0.28,
        gt=0.0,
        le=1.0,
        description="Width of the borrowing hump across the income distribution.",
    )

    # ---- labour ----
    unemployment_rate: float = Field(default=0.07, ge=0.0, le=0.5)
    out_of_labour_force_rate: float = Field(default=0.35, ge=0.0, le=0.9)

    # ---- beliefs and preferences ----
    initial_expected_inflation: float = Field(default=0.05, ge=-0.1, le=0.5)
    expected_inflation_sd: float = Field(default=0.02, ge=0.0, le=0.2)
    risk_aversion_mean: float = Field(default=2.0, gt=0)
    risk_aversion_sd: float = Field(default=0.5, ge=0.0)

    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.min_age >= self.max_age:
            raise ValueError(f"min_age {self.min_age} is not below max_age {self.max_age}")
        if not self.min_age < self.retirement_age < self.max_age:
            raise ValueError("retirement_age must lie between min_age and max_age")
        if self.unemployment_rate + self.out_of_labour_force_rate >= 1.0:
            raise ValueError(
                "unemployment_rate and out_of_labour_force_rate leave no one employed"
            )
        return self

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class Population:
    """A heterogeneous household population held as parallel arrays.

    Every array has length n and shares an index: household i is described
    by age[i], income[i], wealth[i] and so on. Behaviour rules operate on
    whole arrays, so a decision rule applied to ten million households is
    one vectorised expression rather than ten million iterations.

    Arrays are writable because a simulation must update state. What is
    protected instead is the *shape*: the invariant that every array has
    the same length is checked on construction and after any structural
    change, since a length mismatch silently misaligns households and
    corrupts every subsequent result.
    """

    __slots__ = (
        "age",
        "income",
        "wealth",
        "debt",
        "debt_is_floating",
        "employment",
        "expected_inflation",
        "risk_aversion",
        "parameters",
    )

    def __init__(
        self,
        *,
        age: np.ndarray,
        income: np.ndarray,
        wealth: np.ndarray,
        debt: np.ndarray,
        debt_is_floating: np.ndarray,
        employment: np.ndarray,
        expected_inflation: np.ndarray,
        risk_aversion: np.ndarray,
        parameters: PopulationParameters | None = None,
    ) -> None:
        self.age = age
        self.income = income
        self.wealth = wealth
        self.debt = debt
        self.debt_is_floating = debt_is_floating
        self.employment = employment
        self.expected_inflation = expected_inflation
        self.risk_aversion = risk_aversion
        self.parameters = parameters
        self._validate()

    # ---- invariants ----

    def _validate(self) -> None:
        arrays = {
            "age": self.age,
            "income": self.income,
            "wealth": self.wealth,
            "debt": self.debt,
            "debt_is_floating": self.debt_is_floating,
            "employment": self.employment,
            "expected_inflation": self.expected_inflation,
            "risk_aversion": self.risk_aversion,
        }

        lengths = {name: array.shape for name, array in arrays.items()}
        distinct = set(lengths.values())
        if len(distinct) != 1:
            raise EngineError(f"population arrays have mismatched shapes: {lengths}")

        for name, array in arrays.items():
            if array.ndim != 1:
                raise EngineError(f"{name} must be one-dimensional, got {array.ndim}")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise EngineError(f"{name} contains NaN or infinity")

        if np.any(self.income < 0):
            raise EngineError("income must not be negative")
        if np.any(self.debt < 0):
            raise EngineError("debt must not be negative")
        if np.any(self.risk_aversion <= 0):
            raise EngineError("risk aversion must be strictly positive")

    def __len__(self) -> int:
        return int(self.age.shape[0])

    def __repr__(self) -> str:
        return f"Population(n={len(self):,})"

    # ---- derived quantities ----

    @property
    def net_worth(self) -> np.ndarray:
        return self.wealth - self.debt

    @property
    def debt_to_income(self) -> np.ndarray:
        """Zero income means the ratio is undefined; reported as infinity."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.income > 0, self.debt / self.income, np.inf)

    @property
    def is_employed(self) -> np.ndarray:
        return self.employment == EmploymentStatus.EMPLOYED

    @property
    def is_retired(self) -> np.ndarray:
        return self.employment == EmploymentStatus.RETIRED

    @property
    def is_in_labour_force(self) -> np.ndarray:
        return np.isin(
            self.employment, [EmploymentStatus.EMPLOYED, EmploymentStatus.UNEMPLOYED]
        )

    @property
    def is_indebted(self) -> np.ndarray:
        return self.debt > 0

    @property
    def is_net_borrower(self) -> np.ndarray:
        """Who loses from a rate rise, holding everything else fixed."""
        return self.net_worth < 0

    @property
    def is_rate_exposed(self) -> np.ndarray:
        """Floating-rate borrowers: the transmission channel for policy.

        Fixed-rate borrowers are insulated until they refinance, which is
        why the fixed-floating split changes how much a rate rise bites.
        """
        return self.is_indebted & self.debt_is_floating

    # ---- distributional statistics ----

    def gini(self, values: np.ndarray) -> float:
        """Gini coefficient: 0 is perfect equality, 1 is total concentration.

        Computed by the sorted-rank formula, which is O(n log n) rather
        than the O(n^2) pairwise-difference definition.
        """
        if values.size == 0:
            raise EngineError("cannot compute a Gini coefficient of an empty array")
        if np.any(values < 0):
            raise EngineError("Gini is not defined for negative values")

        ordered = np.sort(values)
        n = ordered.size
        total = ordered.sum()
        if total == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * (index * ordered).sum()) / (n * total) - (n + 1) / n)

    def top_share(self, values: np.ndarray, percentile: float = 0.1) -> float:
        """Share of the total held by the top `percentile` of holders."""
        if not 0 < percentile < 1:
            raise EngineError(f"percentile must lie in (0, 1), got {percentile}")
        total = values.sum()
        if total <= 0:
            return 0.0
        cutoff = int(np.ceil(values.size * (1 - percentile)))
        return float(np.sort(values)[cutoff:].sum() / total)

    def quantile_groups(self, values: np.ndarray, n_groups: int = 5) -> np.ndarray:
        """Assign each household to a quantile group, 0 being the lowest.

        The workhorse for distributional analysis: apply a shock, then read
        the response by income quintile or wealth decile.
        """
        if n_groups < 2:
            raise EngineError("need at least two groups")
        ranks = np.argsort(np.argsort(values))
        return (ranks * n_groups // values.size).astype(np.int8)

    def summary(self) -> dict[str, Any]:
        """Headline statistics, suitable for the run ledger."""
        return {
            "n_households": len(self),
            "mean_age": float(self.age.mean()),
            "median_income": float(np.median(self.income)),
            "mean_income": float(self.income.mean()),
            "income_gini": round(self.gini(self.income), 4),
            "wealth_gini": round(self.gini(np.maximum(self.wealth, 0.0)), 4),
            "wealth_top_10_share": round(self.top_share(np.maximum(self.wealth, 0.0)), 4),
            "share_indebted": round(float(self.is_indebted.mean()), 4),
            "share_rate_exposed": round(float(self.is_rate_exposed.mean()), 4),
            "share_net_borrower": round(float(self.is_net_borrower.mean()), 4),
            "employment_rate": round(float(self.is_employed.mean()), 4),
            "mean_expected_inflation": round(float(self.expected_inflation.mean()), 5),
        }

    # ---- copying ----

    def copy(self) -> Population:
        """Deep copy. A simulation branch must not mutate its parent."""
        return Population(
            age=self.age.copy(),
            income=self.income.copy(),
            wealth=self.wealth.copy(),
            debt=self.debt.copy(),
            debt_is_floating=self.debt_is_floating.copy(),
            employment=self.employment.copy(),
            expected_inflation=self.expected_inflation.copy(),
            risk_aversion=self.risk_aversion.copy(),
            parameters=self.parameters,
        )

    def subset(self, mask: np.ndarray) -> Population:
        """A new population containing only the selected households."""
        if mask.shape != (len(self),):
            raise EngineError(f"mask has shape {mask.shape}, expected ({len(self)},)")
        if mask.dtype != bool:
            raise EngineError("mask must be a boolean array")
        if not mask.any():
            raise EngineError("mask selects no households")

        return Population(
            age=self.age[mask],
            income=self.income[mask],
            wealth=self.wealth[mask],
            debt=self.debt[mask],
            debt_is_floating=self.debt_is_floating[mask],
            employment=self.employment[mask],
            expected_inflation=self.expected_inflation[mask],
            risk_aversion=self.risk_aversion[mask],
            parameters=self.parameters,
        )


# ---- generation -----------------------------------------------------------


def generate_population(parameters: PopulationParameters | None = None) -> Population:
    """Draw a synthetic population from the calibration.

    Income and wealth are drawn from a correlated bivariate normal in logs
    rather than independently. Drawing them independently would produce
    wealthy households with no income and vice versa, and would understate
    concentration: in reality high earners also hold the assets.

    The seed lives in the parameters, so a population is a deterministic
    function of its calibration. Two runs with the same parameters produce
    identical households, which is what makes a simulation reproducible.
    """
    parameters = parameters or PopulationParameters()
    n = parameters.n_households
    rng = np.random.default_rng(parameters.seed)

    # ---- age ----
    # Real age distributions are pyramid-shaped, not uniform: India's median
    # age is around 28. A uniform draw would make a third of the population
    # retired and collapse the employment rate.
    span = parameters.max_age - parameters.min_age
    shape = rng.beta(1.6, 3.0, size=n)
    age = (parameters.min_age + shape * span).astype(np.int16)

    # ---- income and wealth, correlated in logs ----
    correlation = np.array(
        [[1.0, parameters.income_wealth_correlation],
         [parameters.income_wealth_correlation, 1.0]]
    )
    draws = rng.multivariate_normal(np.zeros(2), correlation, size=n)

    income = np.exp(parameters.log_income_mean + parameters.log_income_sd * draws[:, 0])
    wealth = np.exp(parameters.log_wealth_mean + parameters.log_wealth_sd * draws[:, 1])

    # Age profile: earnings rise then fall, wealth accumulates with age.
    peak_earning_age = 45.0
    age_factor = 1.0 - 0.00035 * (age - peak_earning_age) ** 2
    income = income * np.clip(age_factor, 0.35, 1.0)
    age_span = parameters.max_age - parameters.min_age
    age_position = (age - parameters.min_age) / age_span
    wealth = wealth * (0.35 + 0.9 * age_position)

    # ---- employment ----
    employment = np.full(n, EmploymentStatus.EMPLOYED, dtype=np.int8)
    retired = age >= parameters.retirement_age
    employment[retired] = EmploymentStatus.RETIRED

    working_age = ~retired
    draw = rng.random(n)
    employment[working_age & (draw < parameters.unemployment_rate)] = (
        EmploymentStatus.UNEMPLOYED
    )
    inactive_cutoff = parameters.unemployment_rate + parameters.out_of_labour_force_rate
    employment[
        working_age
        & (draw >= parameters.unemployment_rate)
        & (draw < inactive_cutoff)
    ] = EmploymentStatus.OUT_OF_LABOUR_FORCE

    # Retirement and unemployment reduce income.
    income[employment == EmploymentStatus.RETIRED] *= 0.45
    income[employment == EmploymentStatus.UNEMPLOYED] *= 0.25
    income[employment == EmploymentStatus.OUT_OF_LABOUR_FORCE] *= 0.30

    # ---- debt ----
    # Borrowing is hump-shaped in income. The poorest are credit
    # constrained and cannot borrow much regardless of want; the middle and
    # upper-middle borrow most, typically against housing; the wealthiest
    # need less leverage because they can buy outright. A flat borrowing
    # probability makes rate exposure uniform across the distribution,
    # which removes the mechanism that makes monetary policy regressive.
    income_rank = np.argsort(np.argsort(income)) / max(n - 1, 1)
    borrowing_propensity = np.exp(
        -((income_rank - parameters.debt_peak_income_rank) ** 2)
        / (2 * parameters.debt_rank_spread**2)
    )
    has_debt = rng.random(n) < parameters.share_with_debt * borrowing_propensity

    # Older households have paid more of it down.
    has_debt &= rng.random(n) > np.clip((age - 30) / 60.0, 0.0, 0.8)
    ratio = rng.lognormal(
        np.log(parameters.debt_to_income_mean), parameters.debt_to_income_sd, size=n
    )
    debt = np.where(has_debt, income * ratio, 0.0)
    debt_is_floating = has_debt & (rng.random(n) < parameters.share_floating_rate)

    # ---- beliefs and preferences ----
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
        debt_is_floating=debt_is_floating,
        employment=employment,
        expected_inflation=expected_inflation,
        risk_aversion=risk_aversion,
        parameters=parameters,
    )

    log.info(
        "population_generated",
        n_households=n,
        seed=parameters.seed,
        income_gini=round(population.gini(population.income), 4),
        wealth_gini=round(population.gini(np.maximum(population.wealth, 0.0)), 4),
        share_rate_exposed=round(float(population.is_rate_exposed.mean()), 4),
    )
    return population