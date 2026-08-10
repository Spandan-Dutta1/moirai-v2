"""Validate the VAR implementation against the canonical monetary system.

Industrial production, CPI, and the federal funds rate: monthly, seasonally
adjusted, the benchmark specification in Christiano, Eichenbaum and Evans.
Reproducing known behaviour on canonical data is how an implementation
earns the right to be trusted on novel data.

The sample ends in 2019. The 2020 collapse is a ten-sigma outlier that
breaks linear models, and handling it needs dummies or a nonlinear
specification rather than being quietly averaged in.
"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import (
    InformationCriterion,
    estimate_var,
    select_lag_order,
)
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("WARNING")  # quiet: we want the numbers, not the plumbing

SERIES = {
    "INDPRO": "Industrial production (index, SA)",
    "CPIAUCSL": "CPI all items (index, SA)",
    "FEDFUNDS": "Federal funds rate (percent)",
}

prepared = []
with FredAdapter() as fred:
    for code, label in SERIES.items():
        series = fred.fetch_series(
            code, observation_start="1960-01-01", observation_end="2019-12-01"
        )
        result = prepare(series)
        print(f"{label:38s} {result.record.steps[0].value:16s} n={len(result)}")
        prepared.append(result)

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)

print()
print(f"sample: {periods[0]} to {periods[-1]}  ({data.shape[0]} observations)")
print()

selection = select_lag_order(data, max_lags=12, variables=names)
print("lag order:", {c.value: selection.best(c) for c in InformationCriterion})
print("criteria agree:", selection.criteria_agree)
print()

result = estimate_var(data, names, n_lags=6, periods=periods)

print(f"VAR({result.n_lags}) on {result.n_variables} variables")
print(f"  observations : {result.n_observations}")
print(f"  params/eqn   : {result.n_parameters}")
print(f"  obs/param    : {result.n_observations / result.n_parameters:.1f}")
print()

print("stability")
print(f"  max modulus  : {result.stability.max_modulus:.4f}")
print(f"  stable       : {result.stability.is_stable}")
print(f"  half life    : {result.stability.half_life:.1f} months")
print()

sigma = result.sigma_u
scale = np.sqrt(np.diag(sigma))
correlation = sigma / np.outer(scale, scale)

print("residual correlations")
short = [n[:8] for n in names]
print("            " + "".join(f"{n:>10s}" for n in short))
for i, name in enumerate(short):
    row = "".join(f"{correlation[i, j]:>10.3f}" for j in range(len(names)))
    print(f"  {name:10s}{row}")
print()
print("Non-zero off-diagonals are the identification problem: these shocks")
print("move together, so none can be varied on its own without an assumption.")