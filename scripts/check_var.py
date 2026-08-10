"""Estimate a VAR on real Indian data."""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import (
    InformationCriterion,
    estimate_var,
    forecast,
    select_lag_order,
)
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("INFO")

SERIES = ["INDCPIALLMINMEI", "INTDSRINM193N"]

prepared = []
with FredAdapter() as fred:
    for code in SERIES:
        prepared.append(prepare(fred.fetch_series(code, observation_start="2000-01-01")))

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)

print()
print("sample :", periods[0], "to", periods[-1], f"({data.shape[0]} obs)")
print()

selection = select_lag_order(data, max_lags=12, variables=names)
print("lag order by criterion:")
for criterion in InformationCriterion:
    print(f"  {criterion.value:5s} -> {selection.best(criterion)}")
print("  criteria agree:", selection.criteria_agree)
print()

result = estimate_var(data, names, criterion=InformationCriterion.BIC, periods=periods)

print("estimated VAR")
print("  lags        :", result.n_lags)
print("  observations:", result.n_observations)
print("  params/eqn  :", result.n_parameters)
print("  df          :", result.degrees_of_freedom)
print()
print("stability")
print("  max modulus :", round(result.stability.max_modulus, 4))
print("  stable      :", result.stability.is_stable)
half_life = result.stability.half_life
print("  half life   :", None if half_life is None else round(half_life, 2), "periods")
print()

print("residual correlation (why identification is still needed)")
sigma = result.sigma_u
correlation = sigma / np.outer(np.sqrt(np.diag(sigma)), np.sqrt(np.diag(sigma)))
print("  corr(u1, u2):", round(float(correlation[0, 1]), 4))
print()

predictions = forecast(result, horizon=6, history=data)
print("6-period forecast")
for step, row in enumerate(predictions, start=1):
    print(f"  h={step}  " + "  ".join(f"{value:+.5f}" for value in row))