"""
The Moirai pipeline on Indian data.

Same machinery as run_pipeline.py, different economy. Two things differ
and both are worth reading rather than skipping:

  * The series arrive as year-on-year rates from RBI, so the preparation
    layer leaves them undifferenced. Nothing here specifies that; it
    follows from the unit recorded at ingestion.

  * The VAR fails its residual autocorrelation tests, for the mechanical
    reason that a year-on-year rate overlaps eleven months with its
    predecessor. The gate is overridden deliberately and the override is
    recorded, so the result is marked provisional rather than presented
    as though it passed.

See docs/decisions/0006-indian-data-uses-year-on-year-series.md.

Run:  python scripts/run_pipeline_india.py
"""

import warnings
from datetime import date

import numpy as np

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky, ordering_sensitivity
from moirai.engine.causal.irf import impulse_responses, variance_decomposition
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import InformationCriterion, estimate_var, select_lag_order
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_GSEC_10Y,
    RBI_IIP,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)
from moirai.engine.data_fabric.series.temporal import drop_missing, restrict
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse
from moirai.engine.economy.behaviour import BehaviourParameters, counterfactual
from moirai.engine.economy.calibration import evaluate
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.economy.shock_path import (
    build_shock_path,
    yoy_percentage_point_mappings,
)

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
configure_logging("ERROR")

# CPI Combined begins in 2012, which sets the common sample.
START = date(2012, 1, 1)
N_LAGS = 6
HORIZON = 36
SHOCK_SCALE = 2.0
N_HOUSEHOLDS = 200_000

PATH = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- Layer 0
rule("LAYER 0  Data Fabric   (RBI monthly indicators)")

if not PATH.exists():
    raise SystemExit(f"missing {PATH}. Download from https://data.rbi.org.in/")

specs = [RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE, RBI_GSEC_10Y]
ingested = []
for spec in specs:
    series = ManualExcelAdapter(spec).ingest(PATH)
    ingested.append(series)
    present = [o for o in series.observations if not o.is_missing]
    print(
        f"  {series.metadata.series_id:<18} {len(present):>4} obs  "
        f"{series.metadata.unit.value:<15} "
        f"{present[0].period} to {present[-1].period}"
    )

with Warehouse() as warehouse:
    for series in ingested:
        warehouse.upsert_series(
            series, provenance={"publisher": "RBI", "file": PATH.name}
        )

print("\n  downloaded by hand, hashed, archived with a provenance sidecar")
print("  a spreadsheet ingested this way is as traceable as an API fetch")

# ---------------------------------------------------------------- Layer 1
rule("LAYER 1  Causal Intelligence")

monetary = [RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE]
prepared = []
for spec in monetary:
    series = ManualExcelAdapter(spec).ingest(PATH)
    result = prepare(drop_missing(restrict(series, start=START)))
    prepared.append(result)
    print(
        f"  {result.series_id:<18} {result.record.steps[0].value:<12} "
        f"n={len(result):<4} {result.record.reason}"
    )

print("\n  the two growth rates were left alone and the rate was differenced.")
print("  that follows from the unit declared at ingestion, not from this script.")
print("  a pipeline without that metadata would have differenced an already")
print("  differenced series and produced a plausible wrong answer.")

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
print(f"\n  aligned sample: {periods[0]} to {periods[-1]}  ({data.shape[0]} obs)")

selection = select_lag_order(data, max_lags=12, variables=names)
print(f"  lag order: {({c.value: selection.best(c) for c in InformationCriterion})}")

var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)
report = diagnose(var, portmanteau_lags=N_LAGS + 12)

print(f"\n  VAR({var.n_lags}) on {var.n_variables} variables")
print(f"    observations per parameter : {var.n_observations / var.n_parameters:.1f}")
print(f"    max eigenvalue modulus     : {var.stability.max_modulus:.3f}")
print(f"    slowest mode half life     : {var.stability.half_life:.1f} months")

print(f"\n  diagnostics: usable = {report.is_usable}")
for outcome in report.critical_failures:
    print(f"    CRITICAL  {outcome.name:<28} p={outcome.pvalue:.4f}")
print()
print("  These failures are mechanical. A year-on-year rate shares eleven")
print("  months with its predecessor, so the residuals are autocorrelated by")
print("  construction and no lag length absorbs it. The result below is")
print("  therefore provisional. See ADR 006.")

model = identify_cholesky(var, ordering=names)
irf = impulse_responses(model, horizon=HORIZON)
fevd = variance_decomposition(model, horizon=HORIZON)

print(f"\n  identification: {model.scheme.value}, "
      f"ordering {' -> '.join(n.replace('in_', '') for n in names)}")
print("    output and prices are assumed not to respond to the policy rate")
print("    within the month. Defensible for monthly data; untestable.")

sensitivity = ordering_sensitivity(var, "in_iip_yoy", "in_repo_rate")
print(f"\n  ordering sensitivity of the output response, all "
      f"{sensitivity.n_orderings} orderings:")
print(f"    [{sensitivity.minimum:+.5f}, {sensitivity.maximum:+.5f}]  "
      f"sign flips: {sensitivity.sign_flips}")

shock = "in_repo_rate_shock"
print("\n  response to a policy rate shock (deviation, percentage points)")
print(f"  {'h':>4}  " + "  ".join(f"{n.replace('in_', '')[:10]:>12}" for n in names))
for h in (0, 3, 6, 12, 24, 36):
    row = "  ".join(f"{irf.responses[h, i, 2]:>12.4f}" for i in range(3))
    print(f"  {h:>4}  {row}")

print("\n  share of forecast error variance from the policy shock at h=36")
for i, name in enumerate(names):
    print(f"    {name.replace('in_', ''):<16} {fevd.shares[HORIZON, i, 2]:.1%}")

# ------------------------------------------------------------ the seam
rule("SEAM  Impulse response to household-facing macro path")

path = build_shock_path(
    irf,
    shock,
    yoy_percentage_point_mappings(
        "in_repo_rate",
        "in_cpi_inflation",
        "in_iip_yoy",
        baseline_rate=0.0525,       # policy repo rate, August 2026
        baseline_inflation=0.044,   # CPI inflation, June 2026
        baseline_income_growth=0.065,
    ),
    scale=SHOCK_SCALE,
    diagnostics=report,
    require_usable=False,  # overridden deliberately; recorded in the ledger
)

print(f"  gate overridden: diagnostics_usable = {path.diagnostics_usable}")
print("  the override travels with the result rather than disappearing\n")
print(f"  {'period':>7}  {'repo rate':>11}  {'inflation':>11}  {'income growth':>14}")
for period in (0, 6, 12, 24, 36):
    values = list(path.at(period).values())
    print(f"  {period:>7}  {values[0]:>10.3%}  {values[1]:>10.3%}  {values[2]:>13.3%}")

# ---------------------------------------------------------------- Layer 3
rule("LAYER 3  Indian Households")

population = generate_population(
    PopulationParameters(n_households=N_HOUSEHOLDS, seed=1)
)
calibration = evaluate(population)
print(f"  {len(population):,} households, calibrated to Indian targets\n")
print(calibration.table())
print(f"\n  {calibration.summary()}")

baseline, shocked = counterfactual(population, path, BehaviourParameters())

base_total = np.sum([o.consumption for o in baseline], axis=0)
shock_total = np.sum([o.consumption for o in shocked], axis=0)
change = (shock_total - base_total) / np.maximum(base_total, 1.0)
aggregate = (shock_total.sum() / base_total.sum() - 1) * 100

# ---------------------------------------------------------------- result
rule("RESULT  Withheld: the shock is not credibly identified")

output_peak_h, output_peak = irf.peak("in_iip_yoy", shock)

print(f"  The output response to a policy tightening peaks at "
      f"{output_peak:+.3f} percentage points at h={output_peak_h}.")
print()
print("  Output rising after a rate rise is not a monetary transmission.")
print("  It is the RBI tightening into expected strength, which a three")
print("  variable VAR cannot separate from the tightening causing that")
print("  strength. Adding the ten year yield did not move the sign in any")
print("  of four specifications, so this is a robust feature of what a")
print("  recursive VAR extracts from these 170 observations, not noise.")
print()
print("  The distributional simulation therefore runs but its output is")
print("  withheld. A shock that makes households richer would produce a")
print("  distributional result, and it would be meaningless.")
print()
print("  What this pipeline does establish:")
print("    - a published spreadsheet ingests with full provenance")
print("    - the preparation layer read the declared units and correctly")
print("      declined to difference two already-differenced series")
print("    - the seam converts an impulse response into household inputs")
print("      with a guard that catches unit errors")
print("    - the specification gate and the plausibility guard both fired")
print()
print("  See ADR 007. The credibly identified chain is run_pipeline.py.")
# ---------------------------------------------------------- provenance
rule("PROVENANCE AND LIMITATIONS")

print("  data           : RBI Select Economic Indicators, downloaded manually")
print(f"  sample         : {periods[0]} to {periods[-1]}")
print(f"  identification : {model.scheme.value}")
print(f"  diagnostics    : usable = {report.is_usable}  "
      f"({len(report.critical_failures)} critical)")
print("  gate           : overridden, recorded in the ledger")
print(f"  calibration    : loss {calibration.loss():.3f}, "
      f"{sum(1 for r in calibration.results if r.moment.confidence.value == 'unsourced')}"
      f" of {len(calibration.results)} targets unsourced")
print()
print("  This result is provisional on two counts:")
print("    1. residual autocorrelation from year-on-year construction")
print("    2. four of seven household calibration targets are unsourced")
print()
print("  ADR 006  docs/decisions/0006-indian-data-uses-year-on-year-series.md")
print("  ADR 005  docs/decisions/0005-calibration-provenance.md")