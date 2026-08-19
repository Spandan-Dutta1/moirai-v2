"""The full chain: conditions, game, transmission, households."""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.economy.behaviour import BehaviourParameters, counterfactual
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.economy.policy_shock import path_from_game
from moirai.engine.economy.shock_path import monetary_mappings
from moirai.engine.financial.central_banks import FED, RBI, analytic_nash

configure_logging("ERROR")

# ---- Layer 1a: the game decides the rate ---------------------------------
shocked_fed = FED.model_copy(update={"current_inflation": 0.045})
solution = analytic_nash(shocked_fed, RBI)

print("LAYER 1a  the policy game")
print(f"  Fed inflation {shocked_fed.current_inflation:.1%} against a "
      f"{FED.inflation_target:.1%} target")
print(f"  equilibrium rate : {solution.rates[FED.name]:.3%}")
print(f"  current rate     : {FED.current_rate:.3%}")
print(f"  the game says move {(solution.rates[FED.name] - FED.current_rate) * 10000:+.0f}bp")
print()

# ---- Layer 1b: the VAR says how it transmits -----------------------------
with FredAdapter() as fred:
    prepared = [
        prepare(
            fred.fetch_series(
                code, observation_start="1985-01-01", observation_end="2007-06-01"
            )
        )
        for code in ("INDPRO", "CPIAUCSL", "FEDFUNDS")
    ]

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=12, periods=periods)
report = diagnose(var, portmanteau_lags=24)
model = identify_cholesky(var, ordering=names)
irf = impulse_responses(model, horizon=36)

print("LAYER 1b  estimated transmission")
print(f"  VAR({var.n_lags}), diagnostics usable: {report.is_usable}")
print(f"  one sd shock moves the funds rate "
      f"{irf.path('fedfunds', 'fedfunds_shock')[0] * 100:.0f}bp on impact")
print()

# ---- the seam ------------------------------------------------------------
path, shock = path_from_game(
    solution,
    FED,
    irf,
    "fedfunds_shock",
    "fedfunds",
    monetary_mappings(
        "fedfunds",
        "cpiaucsl",
        "indpro",
        baseline_rate=FED.current_rate,
        baseline_inflation=shocked_fed.current_inflation,
        baseline_income_growth=0.02,
    ))

print("SEAM  game rate into shock scale")
print(f"  {shock.note}")
print()
print(f"  {'period':>7}  {'policy rate':>12}  {'inflation':>11}  {'income growth':>14}")
for period in (0, 6, 12, 24, 36):
    values = list(path.at(period).values())
    print(f"  {period:>7}  {values[0]:>11.3%}  {values[1]:>10.3%}  {values[2]:>13.3%}")
print()

# ---- Layer 3: households -------------------------------------------------
population = generate_population(PopulationParameters(n_households=200_000, seed=1))
baseline, shocked = counterfactual(population, path, BehaviourParameters())

base = np.sum([o.consumption for o in baseline], axis=0)
after = np.sum([o.consumption for o in shocked], axis=0)
change = (after - base) / np.maximum(base, 1.0)

print("LAYER 3  households")
print(f"  aggregate consumption: {(after.sum() / base.sum() - 1) * 100:+.3f}%")
print()
for label, mask in [
    ("floating-rate borrowers", population.is_rate_exposed),
    ("fixed-rate borrowers", population.is_indebted & ~population.debt_is_floating),
    ("net savers", ~population.is_indebted),
]:
    print(f"  {label:<26} {change[mask].mean() * 100:+7.3f}%")
print()

spread = change[population.is_rate_exposed].mean() - change[~population.is_indebted].mean()
print(f"  spread: {spread * 100:.2f} percentage points")
print()
print("Every number above traces back to the Fed's mandate and an inflation")
print("reading, rather than to a shock size chosen by the analyst.")