"""Income by employment status, held in one place (ADR 038).

Two layers disagreed about what a household earns when nobody in it is
working: 0.25 at generation, 0.35 in behaviour, and neither declared as a
parameter. These tests fix the three things that made that possible — one
object rather than two numbers, income derived from status rather than
adjusted on the way in and out of work, and a share that cannot be zero.

The defect worth keeping in mind is the asymmetric one. The behaviour layer
multiplied income by its figure on job loss and divided by the same figure on
job finding. That cancels only if both layers agree and the household started
in work; a household generated unemployed at 0.25 and divided by 0.35 landed
on 0.71 of an identical household that never lost a job, and stayed there.
`test_a_household_generated_unemployed_reaches_the_employed_level` is that
case.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from moirai.engine.economy.behaviour import BehaviourParameters, step
from moirai.engine.economy.households import (
    UNEMPLOYED_EARNINGS_SHARE_INDIA,
    EmploymentStatus,
    NonEmploymentIncomeParameters,
    Population,
    PopulationParameters,
    generate_population,
)
from moirai.engine.economy.shock_path import MacroVariable, ShockPath

#: A macro state that does nothing: no growth, no inflation, no shortfall, so
#: nothing moves income except the employment transition under test.
STILL = {
    MacroVariable.POLICY_RATE: 0.065,
    MacroVariable.INFLATION: 0.0,
    MacroVariable.INCOME_GROWTH: 0.0,
    MacroVariable.UNEMPLOYMENT: 0.072,
}


def _quiet_path(periods: int = 2) -> ShockPath:
    paths = {variable: np.full(periods, level) for variable, level in STILL.items()}
    return ShockPath(
        shock_name="still", horizon=periods - 1, paths=paths, baselines=STILL
    )


def _one_household(status: EmploymentStatus, income: float = 100.0) -> Population:
    """A single debt-free household, so only the income channel is live."""
    return Population(
        age=np.array([35], dtype=np.int16),
        income=np.array([income]),
        wealth=np.array([10.0]),
        debt=np.array([0.0]),
        debt_is_floating=np.array([False]),
        employment=np.array([status], dtype=np.int8),
        expected_inflation=np.array([0.05]),
        risk_aversion=np.array([2.0]),
    )


# ---- one object, not two numbers ----


def test_generation_and_behaviour_share_one_default() -> None:
    """The defect itself: 0.25 at generation against 0.35 in behaviour."""
    assert (
        PopulationParameters().non_employment_income
        == BehaviourParameters().non_employment_income
    )


def test_the_unemployed_share_is_the_household_measure() -> None:
    """ADR 040 supersedes the 0.57 this record first used.

    0.57 is what the earner loses; 0.83 is what the household keeps, because
    pension, rent, business income and transfers do not stop when a job does.
    The model scales total household income, so it needs the second.
    """
    assert pytest.approx(0.83) == UNEMPLOYED_EARNINGS_SHARE_INDIA
    assert pytest.approx(0.83) == NonEmploymentIncomeParameters().unemployed


def test_neither_layer_still_carries_a_lone_replacement_rate() -> None:
    assert "unemployment_income_replacement" not in BehaviourParameters.model_fields


# ---- the lookup ----


def test_being_in_work_is_the_share_everything_else_is_measured_against() -> None:
    shares = NonEmploymentIncomeParameters().shares
    employed = np.array([EmploymentStatus.EMPLOYED], dtype=np.int8)
    assert shares(employed)[0] == 1.0


def test_each_status_gets_its_own_share() -> None:
    parameters = NonEmploymentIncomeParameters(
        unemployed=0.5, retired=0.4, out_of_labour_force=0.3
    )
    every = np.array(
        [
            EmploymentStatus.UNEMPLOYED,
            EmploymentStatus.EMPLOYED,
            EmploymentStatus.RETIRED,
            EmploymentStatus.OUT_OF_LABOUR_FORCE,
        ],
        dtype=np.int8,
    )
    assert parameters.shares(every).tolist() == [0.5, 1.0, 0.4, 0.3]


def test_the_lookup_preserves_the_order_it_was_given() -> None:
    """A gather, so a population in any order must come back aligned."""
    parameters = NonEmploymentIncomeParameters(unemployed=0.5)
    mixed = np.array(
        [EmploymentStatus.EMPLOYED, EmploymentStatus.UNEMPLOYED, EmploymentStatus.EMPLOYED],
        dtype=np.int8,
    )
    assert parameters.shares(mixed).tolist() == [1.0, 0.5, 1.0]


def test_a_share_of_zero_is_refused_rather_than_guarded() -> None:
    """Zero income cannot be restored by a ratio, so it is rejected up front."""
    for field in ("unemployed", "retired", "out_of_labour_force"):
        with pytest.raises(ValidationError):
            NonEmploymentIncomeParameters(**{field: 0.0})


def test_a_share_above_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        NonEmploymentIncomeParameters(unemployed=1.5)


# ---- generation ----


def test_generated_income_scales_with_the_share_for_that_status_alone() -> None:
    """Halving the unemployed share halves their income and nobody else's."""
    base = PopulationParameters(n_households=20_000, seed=7)
    halved = base.model_copy(
        update={
            "non_employment_income": NonEmploymentIncomeParameters(
                unemployed=base.non_employment_income.unemployed / 2.0
            )
        }
    )
    first, second = generate_population(base), generate_population(halved)

    unemployed = np.asarray(first.employment) == EmploymentStatus.UNEMPLOYED
    employed = np.asarray(first.employment) == EmploymentStatus.EMPLOYED
    assert unemployed.any()

    assert second.income[unemployed].sum() == pytest.approx(
        first.income[unemployed].sum() / 2.0
    )
    assert second.income[employed].sum() == pytest.approx(first.income[employed].sum())


# ---- the transition ----


def _slump() -> ShockPath:
    """Income growth far below baseline, so the job-loss hazard reaches one."""
    baselines = {**STILL, MacroVariable.INCOME_GROWTH: 0.10}
    paths = {variable: np.full(2, level) for variable, level in STILL.items()}
    return ShockPath(
        shock_name="slump", horizon=1, paths=paths, baselines=baselines
    )


def test_losing_a_job_scales_income_by_the_unemployed_share() -> None:
    """A certain job loss, so the share is the only thing that moved income."""
    share = NonEmploymentIncomeParameters().unemployed
    population = _one_household(EmploymentStatus.EMPLOYED, income=100.0)
    parameters = BehaviourParameters(okun_coefficient=20.0, job_finding_rate=1e-12)

    updated, outcome = step(
        population, 0, _slump(), parameters, np.random.default_rng(0), None
    )

    assert updated.employment[0] == EmploymentStatus.UNEMPLOYED
    assert bool(outcome.became_unemployed[0])
    assert updated.income[0] == pytest.approx(100.0 * share)


def test_losing_a_job_and_finding_one_returns_to_the_original_income() -> None:
    """The round trip the old multiply-and-divide was meant to guarantee."""
    population = _one_household(EmploymentStatus.EMPLOYED, income=100.0)
    lost, _ = step(
        population,
        0,
        _slump(),
        BehaviourParameters(okun_coefficient=20.0, job_finding_rate=1e-12),
        np.random.default_rng(0),
        None,
    )
    assert lost.employment[0] == EmploymentStatus.UNEMPLOYED

    regained, _ = step(
        lost,
        0,
        _quiet_path(),
        BehaviourParameters(okun_coefficient=0.0, job_finding_rate=1.0),
        np.random.default_rng(0),
        None,
    )
    assert regained.employment[0] == EmploymentStatus.EMPLOYED
    assert regained.income[0] == pytest.approx(100.0)


def test_no_shortfall_means_no_job_loss_and_no_change_in_income() -> None:
    population = _one_household(EmploymentStatus.EMPLOYED, income=100.0)
    parameters = BehaviourParameters(okun_coefficient=0.0, job_finding_rate=1e-12)

    updated, _ = step(
        population, 0, _quiet_path(), parameters, np.random.default_rng(0), None
    )
    assert updated.employment[0] == EmploymentStatus.EMPLOYED
    assert updated.income[0] == pytest.approx(100.0)


def test_a_household_generated_unemployed_reaches_the_employed_level() -> None:
    """The asymmetry: it used to land on 0.25 / 0.35 = 0.71 of it, for good.

    The household's recorded income already reflects the unemployed share, so
    finding work must gross it up by exactly that share and no other figure.
    """
    share = NonEmploymentIncomeParameters().unemployed
    population = _one_household(EmploymentStatus.UNEMPLOYED, income=100.0 * share)
    parameters = BehaviourParameters(okun_coefficient=0.0, job_finding_rate=1.0)

    updated, _ = step(
        population, 0, _quiet_path(), parameters, np.random.default_rng(0), None
    )

    assert updated.employment[0] == EmploymentStatus.EMPLOYED
    assert updated.income[0] == pytest.approx(100.0)


def test_finding_work_uses_the_same_share_generation_used() -> None:
    """Were the two to disagree again, this household would miss the level."""
    share = 0.5
    shares = NonEmploymentIncomeParameters(unemployed=share)
    population = _one_household(EmploymentStatus.UNEMPLOYED, income=100.0 * share)
    parameters = BehaviourParameters(
        okun_coefficient=0.0, job_finding_rate=1.0, non_employment_income=shares
    )

    updated, _ = step(
        population, 0, _quiet_path(), parameters, np.random.default_rng(0), None
    )
    assert updated.income[0] == pytest.approx(100.0)


def test_a_household_that_stays_put_keeps_its_income() -> None:
    """share / share is one for every status, including the inactive ones."""
    parameters = BehaviourParameters(okun_coefficient=0.0, job_finding_rate=1e-12)
    for status in EmploymentStatus:
        population = _one_household(status, income=100.0)
        updated, _ = step(
            population, 0, _quiet_path(), parameters, np.random.default_rng(1), None
        )
        if updated.employment[0] == status:
            assert updated.income[0] == pytest.approx(100.0), status.name


def test_income_stays_finite_for_every_status() -> None:
    """The division is why this matters: no status may produce inf or nan."""
    population = generate_population(PopulationParameters(n_households=5_000, seed=3))
    parameters = BehaviourParameters(job_finding_rate=1.0)
    updated, _ = step(
        population, 0, _quiet_path(), parameters, np.random.default_rng(0), None
    )
    assert np.all(np.isfinite(updated.income))
    assert np.all(updated.income >= 0.0)
