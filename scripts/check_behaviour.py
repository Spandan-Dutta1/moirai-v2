"""How does a Fed tightening affect households, and which ones?"""

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
from moirai.engine.economy.shock_path import build_shock_path, monetary_mappings

configure_logging("ERROR")

# ---- Layer 1: estimate the shock -----------------------------------------
CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
with FredAdapter() as fred:
    prepared = [
        prepare(
            fred.fetch_series(
                code, observation_start="1985-01-01", observation_end="2007-06-01"
            )
        )
        for code in CODES
    ]

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=12, periods=periods)
report = diagnose(var, portmanteau_lags=24)
model = identify_cholesky(var, ordering=names)
irf = impulse_responses(model, horizon=36)

path = build_shock_path(
    irf,
    "fedfunds_shock",
    monetary_mappings("fedfunds", "cpiaucsl", "indpro"),
    scale=2.0,  # a two standard deviation tightening
    diagnostics=report,
)

print(f"shock: {path.shock_name}, scale {path.scale_factor}x, usable {report.is_usable}")
print(f"peak policy rate deviation: {path.peak(list(path.paths)[0])[1]:+.3%}")
print()

# ---- Layer 3: run the households -----------------------------------------
population = generate_population(PopulationParameters(n_households=200_000, seed=1))
parameters = BehaviourParameters()

baseline, shocked = counterfactual(population, path, parameters)

# ---- aggregate effect ----------------------------------------------------
base_total = sum(o.consumption.sum() for o in baseline)
shock_total = sum(o.consumption.sum() for o in shocked)
print("AGGREGATE")
print(f"  consumption change: {(shock_total / base_total - 1) * 100:+.3f}%")
print(f"  extra job losses  : {sum(int(o.became_unemployed.sum()) for o in shocked) - sum(int(o.became_unemployed.sum()) for o in baseline):,}")
print()

# ---- distributional effect -----------------------------------------------
quintile = population.quantile_groups(population.income, 5)
base_by_household = np.sum([o.consumption for o in baseline], axis=0)
shock_by_household = np.sum([o.consumption for o in shocked], axis=0)
change = (shock_by_household - base_by_household) / np.maximum(base_by_household, 1.0)

print("BY INCOME QUINTILE")
print(f"{'quintile':>9}  {'consumption':>13}  {'% floating':>11}  {'mean income':>13}")
for q in range(5):
    mask = quintile == q
    print(
        f"{q + 1:>9}  {change[mask].mean() * 100:>12.3f}%  "
        f"{population.is_rate_exposed[mask].mean():>10.1%}  "
        f"{population.income[mask].mean():>13,.0f}"
    )
print()

print("BY EXPOSURE")
for label, mask in [
    ("floating-rate borrowers", population.is_rate_exposed),
    ("fixed-rate borrowers", population.is_indebted & ~population.debt_is_floating),
    ("net savers", ~population.is_indebted),
]:
    print(f"  {label:26s} {change[mask].mean() * 100:+7.3f}%   ({mask.sum():,} households)")