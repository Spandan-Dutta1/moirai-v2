"""
How a household responds to the macro path.

A rate rise reaches a household through four channels that pull in
opposite directions for different people:

    debt service      hurts floating-rate borrowers, nobody else
    interest income   helps savers holding liquid wealth
    employment        hurts everyone, and hurts the low paid far more
    real income       inflation erodes wages, and also erodes debt

A representative household nets these to roughly zero and concludes that
monetary policy barely matters. That conclusion is an artefact of
averaging. The households holding floating-rate debt are hit hard; wealthy
savers gain. The aggregate hides a transfer, and the transfer is the
finding.

A household never faces the policy rate. It faces what its bank charges,
and the two differ by a spread that moves asymmetrically: of a hundred
basis point repo rise, Indian borrowers absorb roughly eighty while savers
receive around thirty, and the wedge accrues to the banking system. Those
rates come from the banking layer rather than from a constant, so the
wedge is produced by bank characteristics rather than assumed.

Consumption uses a marginal propensity to consume rather than a solved
dynamic programme. Two reasons: it is tractable across ten million
households, and MPCs are directly estimable from survey and administrative
data, which means the calibration layer can validate against something
real rather than against another model.

The limitation is stated plainly. These are behavioural rules with
parameters, not choices derived from optimisation, so they cannot claim
the internal consistency of a full heterogeneous-agent equilibrium model.
What they offer instead is transparency: every response decomposes into
named channels, and it is always possible to say why a household did what
it did.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from moirai.core.logging import get_logger
from moirai.engine.economy.households import EmploymentStatus, Population
from moirai.engine.economy.shock_path import MacroVariable, ShockPath
from moirai.engine.financial.commercial_banks import BankingSystem

log = get_logger(__name__)

#: Spreads used when no banking system is supplied. Deliberately crude:
#: the point of Layer 2 is that these should not be constants, and a
#: caller running without banks should get a visibly simpler answer.
FALLBACK_LENDING_SPREAD = 0.030
FALLBACK_DEPOSIT_SPREAD = 0.015

#: The source for the high-liquidity end of the MPC gradient (ADR 017).
FHN_2021 = (
    "Fagereng, Holm and Natvik (2021), MPC Heterogeneity and Household "
    "Balance Sheets, American Economic Journal: Macroeconomics 13(4), 1-54"
)

#: Within-year MPC of high-liquidity winners of large lottery prizes in
#: Norwegian administrative data. The abstract reports it as "slightly below
#: one-half"; 0.45 is that reading. It is a lower bound for this model in two
#: respects: the shocks here are persistent changes in debt service and
#: interest income rather than one-off windfalls, and Indian households have
#: thinner access to credit than Norwegian ones. Both push MPCs up.
MPC_HIGH_LIQUIDITY_FHN = 0.45

#: The hand-to-mouth end is not taken from the same source. FHN estimate that
#: low-liquidity winners of the smallest prizes spend all of it within the
#: year, an MPC of one. The model applies this endpoint to every shock a
#: hand-to-mouth household receives, not only to small windfalls, so it
#: stays at 0.70, below the source, as an assumption. ADR 017 reports how
#: the headline moves across 0.70 to 1.0.
MPC_LOW_LIQUIDITY_ASSUMED = 0.70


class BehaviourParameters(BaseModel):
    """Calibration for household decision rules.

    The high-liquidity end of the MPC gradient is sourced from Fagereng,
    Holm and Natvik (2021): even wealthy, liquid households spend slightly
    below half of a windfall within the year. The hand-to-mouth end is an
    assumption set below that source (see MPC_LOW_LIQUIDITY_ASSUMED). The
    gradient matters more than the level: it is what makes the identity of
    the affected household determine the aggregate response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- consumption ----
    mpc_low_wealth: float = Field(
        default=MPC_LOW_LIQUIDITY_ASSUMED,
        gt=0.0,
        le=1.0,
        description=(
            "Marginal propensity to consume for hand-to-mouth households. "
            "Assumed, below the source's estimate of one; see ADR 017."
        ),
    )
    mpc_high_wealth: float = Field(
        default=MPC_HIGH_LIQUIDITY_FHN,
        gt=0.0,
        le=1.0,
        description=(
            "Marginal propensity to consume for households with ample liquid "
            "wealth. Sourced: Fagereng, Holm and Natvik (2021). Was 0.10, "
            "which no source supported; see ADR 017."
        ),
    )
    mpc_wealth_scale: float = Field(
        default=0.5,
        gt=0.0,
        description="Months of income of liquid wealth at which MPC is halfway down.",
    )

    # ---- debt service ----
    debt_maturity_years: float = Field(
        default=15.0,
        gt=0.0,
        description="Average remaining term, used to amortise the payment.",
    )

    # ---- employment ----
    okun_coefficient: float = Field(
        default=0.4,
        ge=0.0,
        description="Percentage point rise in unemployment per point of lost output growth.",
    )
    job_loss_income_gradient: float = Field(
        default=1.8,
        ge=0.0,
        description=(
            "How much more likely the lowest income quintile is to lose work "
            "than the highest. Recessions are not distributed evenly."
        ),
    )
    job_finding_rate: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
        description="Monthly probability an unemployed household finds work.",
    )
    unemployment_income_replacement: float = Field(default=0.35, ge=0.0, le=1.0)

    # ---- expectations ----
    expectation_learning_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Weight on the latest forecast error. Adaptive learning in the "
            "sense of Evans and Honkapohja, not rational expectations."
        ),
    )

    seed: int = Field(default=42, ge=0)

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PeriodOutcome(BaseModel):
    """What happened to the population in one period.

    Arrays are per household so distributional questions stay answerable.
    Aggregates alone would defeat the purpose of simulating a million
    households.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    period: int
    consumption: Any = Field(description="(n,) real consumption.")
    disposable_income: Any = Field(description="(n,) income after debt service.")
    debt_service: Any = Field(description="(n,) annual debt payments.")
    interest_income: Any = Field(description="(n,) income from liquid wealth.")
    became_unemployed: Any = Field(description="(n,) boolean, lost work this period.")
    lending_rate: float = Field(default=0.0, description="Rate banks charged.")
    deposit_rate: float = Field(default=0.0, description="Rate banks paid.")

    def aggregate(self) -> dict[str, float]:
        return {
            "total_consumption": float(self.consumption.sum()),
            "mean_consumption": float(self.consumption.mean()),
            "total_debt_service": float(self.debt_service.sum()),
            "total_interest_income": float(self.interest_income.sum()),
            "job_losses": int(self.became_unemployed.sum()),
            "lending_rate": self.lending_rate,
            "deposit_rate": self.deposit_rate,
        }


# ---- channels -------------------------------------------------------------


def marginal_propensity_to_consume(
    population: Population, parameters: BehaviourParameters
) -> np.ndarray:
    """MPC as a decreasing function of liquid wealth relative to income.

    A household with no buffer spends a windfall immediately because it has
    unmet needs and no capacity to smooth. A household with a year of
    income in the bank saves most of it. The transition is smooth rather
    than a threshold, since hand-to-mouth status is a matter of degree.

    Wealth is measured in months of income rather than in currency, so the
    rule does not need recalibrating when the price level changes.
    """
    monthly_income = np.maximum(population.income / 12.0, 1.0)
    buffer_months = np.maximum(population.wealth, 0.0) / monthly_income

    decay = np.exp(-buffer_months / parameters.mpc_wealth_scale)
    return parameters.mpc_high_wealth + (
        parameters.mpc_low_wealth - parameters.mpc_high_wealth
    ) * decay


def bank_rates(
    policy_rate: float, baseline_policy_rate: float, banks: BankingSystem | None
) -> tuple[float, float, float]:
    """Resolve what a household actually pays and receives.

    Returns the current lending rate, the baseline lending rate that
    fixed-rate borrowers are still paying, and the deposit rate.

    Resolved once per period rather than inside each channel, so the
    banking layer stays out of every downstream function while still
    determining what people face. Without a banking system the spreads are
    flat constants, which is the pre-Layer-2 behaviour and is visibly
    cruder.
    """
    if banks is None:
        return (
            policy_rate + FALLBACK_LENDING_SPREAD,
            baseline_policy_rate + FALLBACK_LENDING_SPREAD,
            max(policy_rate - FALLBACK_DEPOSIT_SPREAD, 0.0),
        )

    current = banks.effective_rates(policy_rate, baseline_policy_rate)
    at_baseline = banks.effective_rates(baseline_policy_rate, baseline_policy_rate)
    return (
        current["lending_rate"],
        at_baseline["lending_rate"],
        current["deposit_rate"],
    )


def debt_service(
    population: Population,
    lending_rate: float,
    parameters: BehaviourParameters,
    *,
    baseline_lending_rate: float,
) -> np.ndarray:
    """Annual debt payments at the rate banks are actually charging.

    Takes a lending rate rather than a policy rate. That distinction is
    the point of the banking layer: a hundred basis point repo rise
    reaches Indian borrowers as roughly eighty, and the wedge is bank
    margin rather than something the household sees.

    Floating-rate borrowers see the change; fixed-rate borrowers keep
    paying the baseline rate until they refinance. That insulation is why
    the fixed-floating mix determines how hard a tightening bites, and why
    India, where mortgages are predominantly floating, transmits faster
    than the United States.

    The payment is a standard amortising annuity, so a rate rise increases
    the payment on the whole outstanding balance rather than only on new
    borrowing.
    """
    effective = np.where(
        population.debt_is_floating, lending_rate, baseline_lending_rate
    )

    n_years = parameters.debt_maturity_years
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            np.abs(effective) < 1e-9,
            1.0 / n_years,
            effective / (1.0 - np.power(1.0 + effective, -n_years)),
        )
    return population.debt * factor


def interest_income(population: Population, deposit_rate: float) -> np.ndarray:
    """Income earned on liquid wealth at the rate banks are paying.

    Deposit rates move less than lending rates, and that asymmetry is now
    produced by the banking layer rather than assumed here. Borrowers pay
    most of a policy increase while savers receive part of it.
    """
    return np.maximum(population.wealth, 0.0) * deposit_rate


def job_loss_probability(
    population: Population,
    income_growth: float,
    parameters: BehaviourParameters,
    *,
    baseline_growth: float,
) -> np.ndarray:
    """Probability of losing work this period, graded by income.

    Okun's law converts the output shortfall into an unemployment rise.
    That rise is then distributed unevenly: the lowest income quintile
    faces a multiple of the highest quintile's risk, because recessions
    concentrate in cyclical, lower paid and less secure work.

    Modelling job loss as uniform would understate the distributional cost
    of a contraction substantially, and would make monetary policy look
    close to distributionally neutral when the evidence says otherwise.
    """
    shortfall = max(baseline_growth - income_growth, 0.0)
    average_hazard = parameters.okun_coefficient * shortfall

    if average_hazard <= 0.0:
        return np.zeros(len(population))

    # Rank within the labour force only. Ranking over everyone would place
    # retirees, whose income is reduced by construction, at the bottom of
    # the distribution, so the gradient would be measured against a
    # denominator that includes people who cannot lose a job.
    in_labour_force = population.is_in_labour_force
    n_active = int(in_labour_force.sum())
    if n_active == 0:
        return np.zeros(len(population))

    hazard = np.zeros(len(population))
    active_income = population.income[in_labour_force]
    rank = np.argsort(np.argsort(active_income)) / max(n_active - 1, 1)

    gradient = parameters.job_loss_income_gradient
    relative = gradient - (gradient - 1.0) * rank  # highest at low income
    relative = relative / relative.mean()

    hazard[in_labour_force] = average_hazard * relative
    return np.clip(hazard, 0.0, 1.0)


def update_expectations(
    population: Population, realised_inflation: float, parameters: BehaviourParameters
) -> np.ndarray:
    """Adaptive learning: revise expectations toward the forecast error.

    Households do not solve the model. They notice they were wrong and
    adjust part of the way, in the tradition of Evans and Honkapohja. This
    is a weaker model of expectation formation than rational expectations
    and a stronger one than static beliefs, and unlike rational
    expectations it does not require households to know the true data
    generating process.
    """
    error = realised_inflation - population.expected_inflation
    return population.expected_inflation + parameters.expectation_learning_rate * error


# ---- the period step ------------------------------------------------------


def step(
    population: Population,
    period: int,
    path: ShockPath,
    parameters: BehaviourParameters,
    rng: np.random.Generator,
    banks: BankingSystem | None = None,
) -> tuple[Population, PeriodOutcome]:
    """Advance the population one period along the shock path.

    Returns a new population and the period's outcome. The input is not
    mutated, so a scenario can be branched without the branches
    interfering, which matters when running the same population against
    several shock paths.
    """
    state = path.at(period)
    policy_rate = state[MacroVariable.POLICY_RATE]
    inflation = state[MacroVariable.INFLATION]
    income_growth = state[MacroVariable.INCOME_GROWTH]

    baseline_rate = path.baselines[MacroVariable.POLICY_RATE]
    baseline_growth = path.baselines[MacroVariable.INCOME_GROWTH]

    updated = population.copy()

    # ---- employment transitions ----
    hazard = job_loss_probability(
        population, income_growth, parameters, baseline_growth=baseline_growth
    )
    draws = rng.random(len(population))
    lost_job = population.is_employed & (draws < hazard)

    found_job = population.employment == EmploymentStatus.UNEMPLOYED
    found_job &= rng.random(len(population)) < parameters.job_finding_rate

    updated.employment[lost_job] = EmploymentStatus.UNEMPLOYED
    updated.employment[found_job] = EmploymentStatus.EMPLOYED

    # ---- income ----
    growth = 1.0 + income_growth / 12.0  # annual rate applied monthly
    updated.income = population.income * growth
    updated.income[lost_job] *= parameters.unemployment_income_replacement
    updated.income[found_job] /= max(parameters.unemployment_income_replacement, 1e-6)

    # ---- interest flows, priced by the banking layer ----
    lending, baseline_lending, deposit = bank_rates(policy_rate, baseline_rate, banks)

    payments = debt_service(
        updated, lending, parameters, baseline_lending_rate=baseline_lending
    )
    receipts = interest_income(updated, deposit)
    disposable = np.maximum(updated.income + receipts - payments, 0.0)

    # ---- consumption ----
    mpc = marginal_propensity_to_consume(updated, parameters)
    consumption = mpc * disposable

    # Saving accumulates into wealth; inflation erodes both wealth and debt.
    saving = disposable - consumption
    real_erosion = 1.0 / (1.0 + inflation / 12.0)
    updated.wealth = (updated.wealth + saving / 12.0) * real_erosion
    updated.debt = updated.debt * real_erosion

    # ---- beliefs ----
    updated.expected_inflation = update_expectations(updated, inflation, parameters)

    updated._validate()

    outcome = PeriodOutcome(
        period=period,
        consumption=consumption,
        disposable_income=disposable,
        debt_service=payments,
        interest_income=receipts,
        became_unemployed=lost_job,
        lending_rate=lending,
        deposit_rate=deposit,
    )
    return updated, outcome


def simulate(
    population: Population,
    path: ShockPath,
    parameters: BehaviourParameters | None = None,
    banks: BankingSystem | None = None,
) -> tuple[Population, list[PeriodOutcome]]:
    """Run the population along the whole shock path.

    The random draw is seeded from the parameters, so a simulation is a
    deterministic function of its population, its shock path, its
    calibration and its banking system. Two runs with the same four inputs
    produce identical output, which is what makes a result reproducible
    rather than illustrative.
    """
    parameters = parameters or BehaviourParameters()
    rng = np.random.default_rng(parameters.seed)

    current = population.copy()
    outcomes: list[PeriodOutcome] = []

    for period in range(len(path)):
        current, outcome = step(current, period, path, parameters, rng, banks)
        outcomes.append(outcome)

    log.info(
        "simulation_completed",
        n_households=len(population),
        periods=len(path),
        shock=path.shock_name,
        total_job_losses=sum(int(o.became_unemployed.sum()) for o in outcomes),
        banks=None if banks is None else len(banks.banks),
        seed=parameters.seed,
    )
    return current, outcomes


def counterfactual(
    population: Population,
    shocked: ShockPath,
    parameters: BehaviourParameters | None = None,
    banks: BankingSystem | None = None,
) -> tuple[list[PeriodOutcome], list[PeriodOutcome]]:
    """Run the same population with and without the shock.

    The baseline path holds every macro variable at its pre-shock level.
    Comparing the two isolates the effect of the shock from the ordinary
    dynamics of ageing, saving and job turnover that would have happened
    anyway. Reporting the shocked path alone would attribute all of that
    to the policy.
    """
    parameters = parameters or BehaviourParameters()

    flat = {
        variable: np.full(len(shocked), shocked.baselines[variable])
        for variable in shocked.paths
    }
    baseline_path = shocked.model_copy(
        update={"paths": flat, "shock_name": f"{shocked.shock_name}_baseline"}
    )

    _, baseline_outcomes = simulate(population, baseline_path, parameters, banks)
    _, shocked_outcomes = simulate(population, shocked, parameters, banks)
    return baseline_outcomes, shocked_outcomes
