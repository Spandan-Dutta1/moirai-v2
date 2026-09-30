"""
The complete Moirai pipeline, end to end.

    real data -> bitemporal storage -> stationarity -> VAR
              -> identification -> impulse responses
              -> the Fed's move -> the central bank game -> the RBI's response
              -> macro shock path -> commercial banks
              -> heterogeneous households -> who gains and who loses

The headline is the chain the project exists to trace, the declared
HEADLINE_SCENARIO: US inflation moves the Fed, the network of central banks
responds, and Indian households bear the part of the RBI's move the Fed
caused. Every link is solved, not chosen. The pipeline used to feed the
Fed's own rate path, scaled to an arbitrary two standard deviations,
straight to Indian households; that version is kept at the end, labelled
as what it is, so the two can be compared. See ADR 016.

Every stage records what it assumed. The specification gate refuses to
build a shock path from a VAR that failed critical diagnostics, so a
misspecified model cannot quietly become a distributional claim.

A household never faces the policy rate. It faces what its bank charges,
so the banking layer sits between the macro path and the household
response, and the pass-through it applies is derived from bank
characteristics calibrated against the RBI's published figures.

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
from moirai.engine.economy.shock_path import (
    MacroVariable,
    build_shock_path,
    monetary_mappings,
)
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.commercial_banks import (
    INDIAN_BANKING_SYSTEM,
    OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH,
)
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    FED_TO_INDIA_EXCHANGE_FLOOR,
    SpilloverMatrix,
)
from moirai.engine.scenarios import HEADLINE_SCENARIO, run_scenario

configure_logging("WARNING")

# The Great Moderation. Chosen because the full 1960-2019 sample fails
# specification tests: it spans the Great Inflation, the Volcker experiment
# and the zero lower bound, which is several economies rather than one.
START, END = "1985-01-01", "2007-06-01"
CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
N_LAGS = 12
HORIZON = 36
#: Only for the comparison at the end: the old headline's arbitrary scale.
COMPARISON_SHOCK_SCALE = 2.0
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
print(f"    slowest mode half life     : {var.stability.half_life:.1f} months")
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


# ------------------------------------------------------------ Layer 1a
rule("LAYER 1a  The central bank game")

scenario = HEADLINE_SCENARIO
central_banks = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)

population = generate_population(
    PopulationParameters(n_households=N_HOUSEHOLDS, seed=1)
)
calibration = evaluate(population)
banks = INDIAN_BANKING_SYSTEM

result = run_scenario(
    scenario,
    central_banks,
    spillovers,
    irf,
    shock_name="fedfunds_shock",
    rate_variable="fedfunds",
    price_variable="cpiaucsl",
    output_variable="indpro",
    population=population,
    banking_system=banks,
    diagnostics=report,
    calibration_loss=calibration.loss(),
)

print(f"  scenario: {scenario.name}")
print(f"  {scenario.description}")
for condition in scenario.conditions:
    print(f"  condition: {condition.bank} inflation at {condition.inflation:.1%}")
print(f"  spillover matrix: {spillovers.confidence.value}")
print()
print(f"  {'bank':<24} {'rate':>9} {'move':>9} {'unprompted':>11} {'caused':>9}")
conditioned = {b.name: b for b in scenario.apply_to(central_banks)}
for name, rate in result.equilibrium.rates.items():
    current = conditioned[name].current_rate
    reference = result.reference_equilibrium.rates[name]
    print(
        f"  {name:<24} {rate:>8.3%} {(rate - current) * 10_000:>+8.0f}bp "
        f"{(reference - current) * 10_000:>+10.0f}bp "
        f"{(rate - reference) * 10_000:>+8.0f}bp"
    )
print("  unprompted: the move each bank makes with no conditions at all")
print("  caused    : the rest, what the US inflation condition produced")


# ------------------------------------------------------------ the seam
rule("SEAM  The RBI's caused move to a household-facing macro path")

path = result.path
print(f"  {result.shock.note}")
print("  gate passed: the VAR is specification-usable")
for channel in scenario.held_channels:
    print(f"  held at baseline: {channel.variable.value}")
if scenario.horizon_cap is not None:
    print(f"  horizon capped at {scenario.horizon_cap.periods} months")
print("  the shape is the US transmission, the size is the RBI's (ADR 010)\n")
print(f"  {'period':>7}  {'policy rate':>12}  {'inflation':>11}  {'income growth':>14}")
for period in sorted({0, 6, 12, 24, path.horizon}):
    values = path.at(period)
    print(
        f"  {period:>7}  {values[MacroVariable.POLICY_RATE]:>11.3%}  "
        f"{values[MacroVariable.INFLATION]:>10.3%}  "
        f"{values[MacroVariable.INCOME_GROWTH]:>13.3%}"
    )


# ---------------------------------------------------------------- Layer 2
rule("LAYER 2  Commercial banks")

print(f"  {len(banks.banks)} banks across three RBI groups\n")
for group in sorted({b.group for b in banks.banks}, key=lambda g: g.value):
    members = banks.by_group(group)
    print(
        f"    {group.value:<9} {len(members):>2} banks, "
        f"{banks.group_share(group):>6.1%} of credit, "
        f"pass-through "
        f"{banks.weighted_lending_pass_through(tightening=True, group=group):.0%}"
    )

baseline_policy = path.baselines[MacroVariable.POLICY_RATE]
peak_period, _ = path.peak(MacroVariable.POLICY_RATE)
policy_move = (path.get(MacroVariable.POLICY_RATE)[peak_period] - baseline_policy) * 10_000

print(f"\n  at the peak of the path, a {policy_move:.0f}bp policy move reaches:")
print(f"    borrowers as {result.lending_rate_change_bp:>5.0f}bp")
print(f"    savers as    {result.deposit_rate_change_bp:>5.0f}bp")
print("    the wedge accrues to the banking system as margin")
print()
print("  Pass-through is derived from each bank's external benchmark share,")
print("  retail deposit dependence and stress, and calibrated against the")
print("  RBI's published transmission figures by bank group.")


# ---------------------------------------------------------------- Layer 3
rule("LAYER 3  Heterogeneous Households")

print(f"  {len(population):,} households, structure of arrays\n")
print(calibration.table())
print(f"\n  {calibration.summary()}")
print("  parameters fitted to declared targets, not chosen")


# ---------------------------------------------------------------- result
rule("RESULT  Who in India bears a US tightening?")

change = result.change_by_household
print(f"  HEADLINE  floating-rate borrowers against net savers: "
      f"{result.spread * 100:.2f} percentage points\n")

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
print()

quintile = population.quantile_groups(population.income, 5)
print(f"  {'income quintile':>16}  {'consumption':>13}  {'% rate exposed':>15}")
for q in range(5):
    mask = quintile == q
    print(
        f"  {q + 1:>16}  {change[mask].mean() * 100:>12.3f}%  "
        f"{population.is_rate_exposed[mask].mean():>14.1%}"
    )
print(f"\n  additional job losses: {result.extra_job_losses:,}")

# Two inputs the evidence bounds but does not pin down, each reported at
# both ends rather than at one. Deposit pass-through: the mechanism predicts
# 31 percent where 97 was observed (ADRs 009, 018). The Fed-to-India
# exchange coefficient: 0.42 by default, consistent with the rupee's
# twelve-month response to Fed surprises, against a floor of 0.17 from its
# one-day response (ADR 019).
observed_banks = banks.model_copy(
    update={"deposit_pass_through_override": OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH}
)
exchange_floor = spillovers.with_exchange(RBI.name, FED.name, FED_TO_INDIA_EXCHANGE_FLOOR)
exchange_default = spillovers.exchange[
    spillovers.index_of(RBI.name), spillovers.index_of(FED.name)
]


def headline_with(spill, banking):
    return run_scenario(
        scenario,
        central_banks,
        spill,
        irf,
        shock_name="fedfunds_shock",
        rate_variable="fedfunds",
        price_variable="cpiaucsl",
        output_variable="indpro",
        population=population,
        banking_system=banking,
        diagnostics=report,
        calibration_loss=calibration.loss(),
    )


mechanism_share = banks.weighted_deposit_pass_through(tightening=True)
rows = [
    (f"as reported (exchange {exchange_default:.2f}, deposits {mechanism_share:.0%})", result),
    (f"deposits as observed 2022-24 ({OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH:.0%})",
     headline_with(spillovers, observed_banks)),
    (f"exchange at the evidence floor ({FED_TO_INDIA_EXCHANGE_FLOOR:.2f})",
     headline_with(exchange_floor, banks)),
    ("both", headline_with(exchange_floor, observed_banks)),
]

print("\n  THE HEADLINE ACROSS WHAT THE EVIDENCE ALLOWS")
print(f"  {'':<46} {'RBI caused':>10} {'aggregate':>10} {'savers':>9} {'spread':>8}")
for label, run in rows:
    print(
        f"  {label:<46} {run.shock.deviation_bp:>+8.1f}bp "
        f"{run.aggregate_consumption_change * 100:>+9.3f}%"
        f" {run.saver_change * 100:>+8.3f}% {run.spread * 100:>6.2f}pp"
    )
spreads = [run.spread * 100 for _, run in rows]
print()
print(f"  spread range: {max(spreads):.2f} to {min(spreads):.2f} percentage points")
print()
print("  Deposit pass-through moves the aggregate and the savers, not the spread:")
print("  the spread is carried by what floating-rate borrowers pay. The exchange")
print("  coefficient moves everything, because it sets how much of the Fed's move")
print("  the RBI imports. The one-day rupee response to Fed surprises puts a floor")
print("  under it; the twelve-month response is consistent with the default but")
print("  too noisy to confirm. So the finding is a range, and the transfer from")
print("  borrowers to savers is its shape at every point in it.")


# ----------------------------------------------------------- comparison
rule("FOR COMPARISON  The Fed's own path, delivered straight to households")

direct_path = build_shock_path(
    irf,
    "fedfunds_shock",
    monetary_mappings("fedfunds", "cpiaucsl", "indpro"),
    scale=COMPARISON_SHOCK_SCALE,
    diagnostics=report,
    require_usable=True,
)
direct_base, direct_shock = counterfactual(
    population, direct_path, BehaviourParameters(), banks
)
d_base = np.sum([o.consumption for o in direct_base], axis=0)
d_shock = np.sum([o.consumption for o in direct_shock], axis=0)
d_change = (d_shock - d_base) / np.maximum(d_base, 1.0)
d_borrowers = d_change[population.is_rate_exposed].mean()
d_savers = d_change[~population.is_indebted].mean()

print(f"  the pipeline's former headline: the Fed's rate path at "
      f"{COMPARISON_SHOCK_SCALE} standard deviations,")
print("  a size chosen because it produced a visible response, fed to Indian")
print("  households with no central bank game and no RBI in between\n")
print(f"  {'':<26} {'chain':>10} {'Fed direct':>12}")
print(f"  {'aggregate consumption':<26} {result.aggregate_consumption_change * 100:>9.3f}% "
      f"{(d_shock.sum() / d_base.sum() - 1) * 100:>11.3f}%")
print(f"  {'floating-rate borrowers':<26} {result.borrower_change * 100:>9.3f}% "
      f"{d_borrowers * 100:>11.3f}%")
print(f"  {'net savers':<26} {result.saver_change * 100:>9.3f}% {d_savers * 100:>11.3f}%")
print(f"  {'spread (pp)':<26} {result.spread * 100:>10.2f} {(d_borrowers - d_savers) * 100:>12.2f}")
print()
print("  The direct version is not the chain. It skips the game and the RBI,")
print("  so it measures what the Fed's path would do if Indian households")
print("  faced it themselves. Read it as an upper bound, not a scenario.")


# ---------------------------------------------------------- provenance
rule("PROVENANCE")

print("  data vintage      : fetched live, content-hashed, archived")
print(f"  sample            : {START} to {END}, chosen on specification grounds")
print(f"  identification    : {model.scheme.value}, ordering {' -> '.join(names)}")
print(f"  diagnostics       : {report.is_usable} ({len(report.advisory_failures)} advisory)")
print(f"  headline scenario : {scenario.name}, shock origin {scenario.shock_origin}")
print(f"  shock size        : {result.shock.scale:.2f} standard deviations, "
      f"solved by the game ({result.shock.concept})")
print(f"  spillover matrix  : {spillovers.confidence.value}")
print(f"  deposit pass-thru : mechanism {mechanism_share:.0%}, bounded at the observed "
      f"{OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH:.0%} (ADR 009, 018)")
print(f"  Fed-India exchange: {exchange_default:.2f}, bounded at the evidence floor "
      f"{FED_TO_INDIA_EXCHANGE_FLOOR:.2f} (ADR 019)")
print(f"  banking system    : {len(banks.banks)} banks, "
      f"pass-through calibrated to RBI bulletin figures")
print(f"  population seed   : {population.parameters.seed}")
print(f"  calibration loss  : {calibration.loss():.3f}")
print(f"  unsourced targets : "
      f"{sum(1 for r in calibration.results if r.moment.confidence.value == 'unsourced')}"
      f" of {len(calibration.results)}")
print()
print("  Results depending on unsourced calibration targets are provisional.")
print("  See docs/decisions/0005-calibration-provenance.md, 0016, 0018 and 0019.")
