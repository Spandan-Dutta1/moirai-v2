"""
The complete Moirai pipeline, end to end.

    real data -> bitemporal storage -> stationarity -> VAR
              -> identification -> impulse responses
              -> macro shock path -> heterogeneous households
              -> who gains and who loses

Every stage records what it assumed. The specification gate refuses to
build a shock path from a VAR that failed critical diagnostics, so a
misspecified model cannot quietly become a distributional claim.

Run:  python scripts/run_pipeline.py
"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky, ordering_sensitivity
from moirai.engine.causal.irf import impulse_responses, variance_decomposition
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse
from moirai.engine.economy.behaviour import BehaviourParameters, counterfactual
from moirai.engine.economy.calibration import evaluate
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.economy.shock_path import build_shock_path, monetary_mappings

configure_logging("ERROR")

# The Great Moderation. Chosen because the full 1960-2019 sample fails
# specification tests: it spans the Great Inflation, the Volcker experiment
# and the zero lower bound, which is several economies rather than one.
START, END = "1985-01-01", "2007-06-01"
CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
N_LAGS = 12
HORIZON = 36
SHOCK_SCALE = 2.0
N_HOUSEHOLDS = 200_000


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- Layer 0
rule("LAYER 0  Data Fabric")

with FredAdapter() as fred:
    series = [
        fred.fetch_series(code, observation_start=START, observation_end=END)
        for code in CODES
    ]

with Warehouse() as warehouse:
    for item in series:
        warehouse.upsert_series(item, provenance={"pipeline": "run_pipeline"})
    stored = warehouse.list_series()

for item in series:
    print(
        f"  {item.metadata.series_id:<12} {len(item):>4} obs  "
        f"{item.metadata.frequency.value:<10} {item.metadata.unit.value:<16} "
        f"{item.metadata.seasonal_adjustment.value}"
    )
print(f"\n  warehouse now holds {len(stored)} series "
      f"({len(series)} fetched by this run)")
print("  raw responses archived with sha256 content hashes")


# ---------------------------------------------------------------- Layer 1
rule("LAYER 1  Causal Intelligence")

prepared = [prepare(item) for item in series]
for item in prepared:
    print(
        f"  {item.series_id:<12} {item.record.steps[0].value:<16} "
        f"n={len(item):<5} {item.record.reason}"
    )

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
print(f"\n  aligned sample: {periods[0]} to {periods[-1]}  ({data.shape[0]} obs)")

var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)
report = diagnose(var, portmanteau_lags=N_LAGS + 12)

print(f"\n  VAR({var.n_lags}) on {var.n_variables} variables")
print(f"    observations per parameter : {var.n_observations / var.n_parameters:.1f}")
print(f"    max eigenvalue modulus     : {var.stability.max_modulus:.3f}")
print(f"    shock half life            : {var.stability.half_life:.1f} months")
print(f"\n  diagnostics: {report.summary()}")
print(f"  usable     : {report.is_usable}")

model = identify_cholesky(var, ordering=names)
print(f"\n  identification: {model.scheme.value}")
print(f"    {model.assumptions}")
print(f"    reconstruction error: {model.reconstruction_error():.2e}")

sensitivity = ordering_sensitivity(var, "indpro", "fedfunds")
print("\n  ordering sensitivity of the output response to a policy shock")
print(f"    across all {sensitivity.n_orderings} orderings: "
      f"[{sensitivity.minimum:+.5f}, {sensitivity.maximum:+.5f}]")
print(f"    sign flips: {sensitivity.sign_flips}")
print("    the lower bound is the assumption, not an estimate: orderings")
print("    placing output before the rate force this response to zero")

irf = impulse_responses(model, horizon=HORIZON)
cumulative = irf.cumulate()
fevd = variance_decomposition(model, horizon=HORIZON)

print("\n  cumulative response to a monetary tightening (percent of level)")
print(f"  {'h':>4}  " + "  ".join(f"{n[:9]:>10}" for n in names))
for h in (6, 12, 24, 36):
    row = "  ".join(f"{cumulative.responses[h, i, 2] * 100:>10.3f}" for i in range(3))
    print(f"  {h:>4}  {row}")

print("\n  note: the price level response is positive, which is the price")
print("  puzzle (Eichenbaum 1992). Adding a commodity price index, the")
print("  standard fix, did not resolve it on this sample and cost")
print("  specification validity. The anomaly is small and left stated.")

print("\n  share of forecast error variance explained by the policy shock at h=36")
for i, name in enumerate(names):
    print(f"    {name:<12} {fevd.shares[HORIZON, i, 2]:.1%}")
print("    a clearly signed effect can still be a small one")
# ------------------------------------------------------------ the seam
rule("SEAM  Impulse response to household-facing macro path")

path = build_shock_path(
    irf,
    "fedfunds_shock",
    monetary_mappings("fedfunds", "cpiaucsl", "indpro"),
    scale=SHOCK_SCALE,
    diagnostics=report,
    require_usable=True,
)

print(f"  shock scaled to {SHOCK_SCALE} standard deviations")
print("  gate passed: the VAR is specification-usable\n")
print(f"  {'period':>7}  {'policy rate':>12}  {'inflation':>11}  {'income growth':>14}")
for period in (0, 6, 12, 24, 36):
    state = path.at(period)
    values = list(state.values())
    print(f"  {period:>7}  {values[0]:>11.3%}  {values[1]:>10.3%}  {values[2]:>13.3%}")

# ---------------------------------------------------------------- Layer 3
rule("LAYER 3  Heterogeneous Households")

population = generate_population(
    PopulationParameters(n_households=N_HOUSEHOLDS, seed=1)
)
calibration = evaluate(population)

print(f"  {len(population):,} households, structure of arrays\n")
print(calibration.table())
print(f"\n  {calibration.summary()}")
print("  parameters fitted to declared targets, not chosen")

baseline, shocked = counterfactual(population, path, BehaviourParameters())

base_total = np.sum([o.consumption for o in baseline], axis=0)
shock_total = np.sum([o.consumption for o in shocked], axis=0)
change = (shock_total - base_total) / np.maximum(base_total, 1.0)

aggregate = (shock_total.sum() / base_total.sum() - 1) * 100
extra_losses = sum(int(o.became_unemployed.sum()) for o in shocked) - sum(
    int(o.became_unemployed.sum()) for o in baseline
)

# ---------------------------------------------------------------- result
rule("RESULT  Who bears a monetary tightening?")

print(f"  aggregate consumption change : {aggregate:+.3f}%")
print(f"  additional job losses        : {extra_losses:,}")
print()

quintile = population.quantile_groups(population.income, 5)
print(f"  {'income quintile':>16}  {'consumption':>13}  {'% rate exposed':>15}")
for q in range(5):
    mask = quintile == q
    print(
        f"  {q + 1:>16}  {change[mask].mean() * 100:>12.3f}%  "
        f"{population.is_rate_exposed[mask].mean():>14.1%}"
    )
print()

print(f"  {'by exposure':>26}  {'consumption':>13}  {'households':>12}")
for label, mask in [
    ("floating-rate borrowers", population.is_rate_exposed),
    ("fixed-rate borrowers", population.is_indebted & ~population.debt_is_floating),
    ("net savers", ~population.is_indebted),
]:
    print(
        f"  {label:>26}  {change[mask].mean() * 100:>12.3f}%  "
        f"{int(mask.sum()):>12,}"
    )

spread = change[population.is_rate_exposed].mean() - change[~population.is_indebted].mean()
print()
print(f"  spread between borrowers and savers: {spread * 100:.2f} percentage points")
print()
print("  A representative household nets these to roughly the aggregate above")
print("  and concludes that monetary policy barely moves consumption. The")
print("  aggregate is small because it is a transfer, and the transfer is")
print("  the finding.")

# ---------------------------------------------------------- provenance
rule("PROVENANCE")

print("  data vintage      : fetched live, content-hashed, archived")
print(f"  sample            : {START} to {END}, chosen on specification grounds")
print(f"  identification    : {model.scheme.value}, ordering {' -> '.join(names)}")
print(f"  diagnostics       : {report.is_usable} ({len(report.advisory_failures)} advisory)")
print(f"  shock scale       : {SHOCK_SCALE} standard deviations")
print(f"  population seed   : {population.parameters.seed}")
print(f"  calibration loss  : {calibration.loss():.3f}")
print(f"  unsourced targets : "
      f"{sum(1 for r in calibration.results if r.moment.confidence.value == 'unsourced')}"
      f" of {len(calibration.results)}")
print()
print("  Results depending on unsourced calibration targets are provisional.")
print("  See docs/decisions/0005-calibration-provenance.md")