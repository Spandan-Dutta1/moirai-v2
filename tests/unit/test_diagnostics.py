"""Tests for specification diagnostics.

The technique is construction: build residuals with a known defect and
check the corresponding test catches it, then build clean residuals and
check it does not. A diagnostic that never fires is as useless as one that
always does, so both directions matter.
"""

import numpy as np
import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.causal.diagnostics import (
    CheckOutcome,
    DiagnosticReport,
    Severity,
    arch_lm_test,
    cusum_test,
    diagnose,
    granger_causality,
    jarque_bera_test,
    ljung_box_per_equation,
    portmanteau_test,
)
from moirai.engine.causal.var import estimate_var

NAMES = ("output", "prices", "rate")

# Well-specified VAR(1): residuals should be clean by construction.
CLEAN_A = np.array(
    [
        [
            [0.50, 0.10, -0.20],
            [0.15, 0.60, -0.10],
            [0.20, 0.25, 0.55],
        ]
    ]
)
SIGMA = np.eye(3)


def simulate_var1(n_obs: int = 1_500, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burn_in = 200
    total = n_obs + burn_in
    shocks = rng.multivariate_normal(np.zeros(3), SIGMA, size=total)
    series = np.zeros((total, 3))
    for t in range(1, total):
        series[t] = CLEAN_A[0] @ series[t - 1] + shocks[t]
    return series[burn_in:]


def simulate_var3(n_obs: int = 1_500, seed: int = 5) -> np.ndarray:
    """A VAR(3). Fitting a VAR(1) to it leaves serial correlation behind."""
    coefficients = np.array(
        [
            [[0.30, 0.05, -0.10], [0.05, 0.30, -0.05], [0.10, 0.10, 0.30]],
            [[0.20, 0.05, -0.05], [0.05, 0.20, -0.05], [0.05, 0.10, 0.20]],
            [[0.15, 0.00, -0.05], [0.00, 0.15, 0.00], [0.05, 0.05, 0.15]],
        ]
    )
    rng = np.random.default_rng(seed)
    burn_in = 200
    total = n_obs + burn_in
    shocks = rng.multivariate_normal(np.zeros(3), SIGMA, size=total)
    series = np.zeros((total, 3))
    for t in range(3, total):
        value = np.zeros(3)
        for lag in range(3):
            value = value + coefficients[lag] @ series[t - lag - 1]
        series[t] = value + shocks[t]
    return series[burn_in:]


@pytest.fixture(scope="module")
def clean_var():
    """Correctly specified: VAR(1) fitted to VAR(1) data."""
    return estimate_var(simulate_var1(), NAMES, n_lags=1)


@pytest.fixture(scope="module")
def underfitted_var():
    """Misspecified: VAR(1) fitted to VAR(3) data."""
    return estimate_var(simulate_var3(), NAMES, n_lags=1, require_stable=False)


# --- CheckOutcome semantics -------------------------------------------------

def make_outcome(pvalue: float, severity: Severity = Severity.CRITICAL) -> CheckOutcome:
    return CheckOutcome(
        name="example",
        statistic=1.0,
        pvalue=pvalue,
        null_hypothesis="nothing is wrong",
        severity=severity,
    )


def test_high_pvalue_passes():
    """Passing means failing to reject, which is what these tests want."""
    assert make_outcome(0.5).passed is True
    assert make_outcome(0.5).rejects_null is False


def test_low_pvalue_fails():
    assert make_outcome(0.001).passed is False
    assert make_outcome(0.001).rejects_null is True


def test_significance_boundary_is_strict():
    assert make_outcome(0.05).rejects_null is False
    assert make_outcome(0.0499).rejects_null is True


def test_significance_level_is_configurable():
    outcome = CheckOutcome(
        name="x",
        statistic=1.0,
        pvalue=0.03,
        null_hypothesis="n",
        severity=Severity.ADVISORY,
        significance=0.01,
    )
    assert outcome.passed is True


def test_outcome_serialises_for_the_ledger():
    payload = make_outcome(0.5).to_ledger_dict()
    assert payload["passed"] is True
    assert payload["severity"] == "critical"
    assert "null_hypothesis" in payload


# --- portmanteau -----------------------------------------------------------

def test_portmanteau_passes_on_a_correct_specification(clean_var):
    assert portmanteau_test(clean_var, n_lags=12).passed is True


def test_portmanteau_catches_an_underfitted_model(underfitted_var):
    """The most important behaviour in the module: too few lags is caught."""
    assert portmanteau_test(underfitted_var, n_lags=12).passed is False


def test_portmanteau_is_critical(clean_var):
    assert portmanteau_test(clean_var).severity is Severity.CRITICAL


def test_portmanteau_reports_degrees_of_freedom(clean_var):
    outcome = portmanteau_test(clean_var, n_lags=12)
    assert "degrees of freedom" in outcome.detail


def test_portmanteau_requires_more_test_lags_than_model_lags(clean_var):
    """Otherwise the statistic has no positive degrees of freedom."""
    with pytest.raises(EngineError, match="must exceed model lags"):
        portmanteau_test(clean_var, n_lags=1)


def test_portmanteau_rejects_more_lags_than_observations(clean_var):
    with pytest.raises(EngineError, match="cannot test"):
        portmanteau_test(clean_var, n_lags=10_000)


# --- per-equation Ljung-Box ------------------------------------------------

def test_ljung_box_returns_one_outcome_per_equation(clean_var):
    assert len(ljung_box_per_equation(clean_var)) == 3


def test_ljung_box_names_each_equation(clean_var):
    names = [o.name for o in ljung_box_per_equation(clean_var)]
    assert names == ["ljung_box[output]", "ljung_box[prices]", "ljung_box[rate]"]


def test_ljung_box_passes_on_a_correct_specification(clean_var):
    assert all(o.passed for o in ljung_box_per_equation(clean_var, n_lags=12))


def test_ljung_box_localises_a_failure(underfitted_var):
    """The joint test says something is wrong; this says where."""
    outcomes = ljung_box_per_equation(underfitted_var, n_lags=12)
    assert any(not o.passed for o in outcomes)


# --- ARCH-LM ---------------------------------------------------------------

def test_arch_passes_on_homoskedastic_residuals(clean_var):
    assert all(o.passed for o in arch_lm_test(clean_var, n_lags=5))


def test_arch_is_advisory(clean_var):
    """OLS stays consistent under heteroskedasticity; robust errors suffice."""
    assert all(o.severity is Severity.ADVISORY for o in arch_lm_test(clean_var))


def test_arch_detects_volatility_clustering():
    """Build an explicit ARCH process and confirm it is caught."""
    rng = np.random.default_rng(11)
    n = 1_500
    series = np.zeros((n, 3))
    variance = 1.0
    for t in range(1, n):
        variance = 0.2 + 0.75 * series[t - 1, 0] ** 2
        series[t, 0] = rng.normal(0, np.sqrt(variance))
        series[t, 1] = rng.normal(0, 1)
        series[t, 2] = rng.normal(0, 1)

    var = estimate_var(series, NAMES, n_lags=1, require_stable=False)
    assert arch_lm_test(var, n_lags=5)[0].passed is False


def test_arch_returns_one_outcome_per_equation(clean_var):
    assert len(arch_lm_test(clean_var)) == 3


# --- Jarque-Bera -----------------------------------------------------------

def test_jarque_bera_passes_on_normal_residuals(clean_var):
    assert all(o.passed for o in jarque_bera_test(clean_var))


def test_jarque_bera_is_advisory(clean_var):
    """The bootstrap does not assume normality, which is why it was chosen."""
    assert all(o.severity is Severity.ADVISORY for o in jarque_bera_test(clean_var))


def test_jarque_bera_detects_fat_tails():
    rng = np.random.default_rng(13)
    n = 1_500
    series = np.zeros((n, 3))
    for t in range(1, n):
        series[t, 0] = rng.standard_t(df=3)  # heavy tails
        series[t, 1] = rng.normal(0, 1)
        series[t, 2] = rng.normal(0, 1)

    var = estimate_var(series, NAMES, n_lags=1, require_stable=False)
    assert jarque_bera_test(var)[0].passed is False


def test_jarque_bera_reports_moments(clean_var):
    assert "skewness" in jarque_bera_test(clean_var)[0].detail


# --- CUSUM -----------------------------------------------------------------

def test_cusum_passes_on_a_stable_sample(clean_var):
    assert all(o.passed for o in cusum_test(clean_var))


def test_cusum_is_critical(clean_var):
    """A VAR spanning a break describes an average of two regimes."""
    assert all(o.severity is Severity.CRITICAL for o in cusum_test(clean_var))


def test_cusum_returns_one_outcome_per_equation(clean_var):
    assert len(cusum_test(clean_var)) == 3


def test_cusum_reports_corridor_breaches(clean_var):
    assert "outside the corridor" in cusum_test(clean_var)[0].detail


# --- Granger causality -----------------------------------------------------

def test_granger_detects_a_real_predictive_relationship():
    """Construct x driving y, and confirm the test finds it."""
    rng = np.random.default_rng(17)
    n = 1_500
    series = np.zeros((n, 3))
    for t in range(1, n):
        series[t, 0] = 0.5 * series[t - 1, 0] + rng.normal(0, 1)
        series[t, 1] = 0.8 * series[t - 1, 0] + rng.normal(0, 1)  # driven by 0
        series[t, 2] = 0.3 * series[t - 1, 2] + rng.normal(0, 1)

    var = estimate_var(series, NAMES, n_lags=1)
    assert granger_causality(var, "output", "prices", series).rejects_null is True


def test_granger_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(19)
    n = 1_500
    series = np.zeros((n, 3))
    for t in range(1, n):
        series[t] = 0.4 * series[t - 1] + rng.normal(0, 1, 3)  # diagonal only

    var = estimate_var(series, NAMES, n_lags=1)
    assert granger_causality(var, "output", "prices", series).passed is True


def test_granger_is_directional():
    """x predicting y does not imply y predicts x."""
    rng = np.random.default_rng(23)
    n = 1_500
    series = np.zeros((n, 3))
    for t in range(1, n):
        series[t, 0] = 0.5 * series[t - 1, 0] + rng.normal(0, 1)
        series[t, 1] = 0.8 * series[t - 1, 0] + rng.normal(0, 1)
        series[t, 2] = 0.3 * series[t - 1, 2] + rng.normal(0, 1)

    var = estimate_var(series, NAMES, n_lags=1)
    forward = granger_causality(var, "output", "prices", series)
    backward = granger_causality(var, "prices", "output", series)
    assert forward.pvalue < backward.pvalue


def test_granger_detail_warns_against_causal_reading(clean_var):
    """The name misleads; the result should not."""
    data = simulate_var1()
    outcome = granger_causality(clean_var, "output", "prices", data)
    assert "not evidence of causation" in outcome.detail


def test_granger_is_advisory(clean_var):
    data = simulate_var1()
    outcome = granger_causality(clean_var, "output", "prices", data)
    assert outcome.severity is Severity.ADVISORY


def test_granger_rejects_self_causation(clean_var):
    data = simulate_var1()
    with pytest.raises(EngineError, match="cannot Granger-cause itself"):
        granger_causality(clean_var, "output", "output", data)


def test_granger_rejects_an_unknown_variable(clean_var):
    data = simulate_var1()
    with pytest.raises(EngineError, match="not in this VAR"):
        granger_causality(clean_var, "nope", "prices", data)


def test_granger_rejects_mismatched_data(clean_var):
    with pytest.raises(EngineError, match="columns, expected"):
        granger_causality(clean_var, "output", "prices", np.zeros((100, 5)))


# --- full report -----------------------------------------------------------

def test_report_collects_every_test(clean_var):
    report = diagnose(clean_var)
    names = {o.name.split("[")[0] for o in report.outcomes}
    assert names == {"portmanteau", "ljung_box", "arch_lm", "jarque_bera", "cusum"}


def test_clean_model_is_usable(clean_var):
    assert diagnose(clean_var).is_usable is True


def test_underfitted_model_is_not_usable(underfitted_var):
    """The verdict that matters: the pipeline refuses to vouch for it."""
    assert diagnose(underfitted_var).is_usable is False


def test_critical_and_advisory_failures_are_separated(underfitted_var):
    report = diagnose(underfitted_var)
    assert all(o.severity is Severity.CRITICAL for o in report.critical_failures)
    assert all(o.severity is Severity.ADVISORY for o in report.advisory_failures)


def test_advisory_failures_do_not_block_usability():
    """Non-normal residuals are common in macro data and not disqualifying."""
    report = DiagnosticReport(
        variables=NAMES,
        n_lags=1,
        n_observations=100,
        outcomes=(make_outcome(0.001, Severity.ADVISORY),),
    )
    assert report.is_usable is True
    assert len(report.advisory_failures) == 1


def test_summary_names_critical_failures(underfitted_var):
    assert "CRITICAL" in diagnose(underfitted_var).summary()


def test_summary_is_clean_when_everything_passes(clean_var):
    assert diagnose(clean_var).summary() == "All specification checks passed."


def test_report_serialises_for_the_ledger(clean_var):
    payload = diagnose(clean_var).to_ledger_dict()
    assert payload["is_usable"] is True
    assert len(payload["outcomes"]) == 13  # 1 + 3 + 3 + 3 + 3
    assert payload["n_lags"] == 1


def test_report_is_frozen(clean_var):
    report = diagnose(clean_var)
    with pytest.raises(Exception):
        report.n_lags = 99  # type: ignore[misc]


def test_diagnosis_does_not_mutate_the_var(clean_var):
    before = clean_var.residuals.copy()
    diagnose(clean_var)
    assert np.array_equal(clean_var.residuals, before)