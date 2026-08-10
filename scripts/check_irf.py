"""Impulse responses for the canonical monetary VAR."""


from moirai.core.logging import configure_logging
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import (
    bootstrap_bands,
    impulse_responses,
    variance_decomposition,
)
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("WARNING")

CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]

prepared = []
with FredAdapter() as fred:
    for code in CODES:
        prepared.append(
            prepare(
                fred.fetch_series(
                    code, observation_start="1960-01-01", observation_end="2019-12-01"
                )
            )
        )

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=6, periods=periods)
model = identify_cholesky(var, ordering=names)

HORIZON = 36
irf = impulse_responses(model, horizon=HORIZON)

print(f"VAR({var.n_lags}), Cholesky ordering {' -> '.join(names)}")
print(f"horizon {HORIZON} months")
print()

print("RESPONSE TO A MONETARY POLICY SHOCK (one std dev tightening)")
print(f"{'h':>4}  {'indpro':>10}  {'cpiaucsl':>10}  {'fedfunds':>10}")
for h in (0, 3, 6, 12, 18, 24, 36):
    row = "  ".join(f"{irf.responses[h, i, 2]:>10.5f}" for i in range(3))
    print(f"{h:>4}  {row}")
print()

for variable in names:
    when, value = irf.peak(variable, "fedfunds_shock")
    print(f"  peak {variable[:9]:10s}: {value:+.5f} at h={when}")
print()

# Cumulative: indpro and cpiaucsl were log-differenced, so levels are the sum.
cumulative = irf.cumulate()
print("CUMULATIVE (level response for differenced variables)")
for h in (12, 24, 36):
    ip = cumulative.responses[h, 0, 2] * 100
    cpi = cumulative.responses[h, 1, 2] * 100
    print(f"  h={h:>2}:  industrial production {ip:+.3f}%   price level {cpi:+.3f}%")
print()

fevd = variance_decomposition(model, horizon=HORIZON)
print("VARIANCE DECOMPOSITION at h=36 (share of forecast error)")
print(f"{'variable':>10}  " + "  ".join(f"{n[:9]:>10}" for n in irf.shock_names))
for i, variable in enumerate(names):
    row = "  ".join(f"{fevd.shares[36, i, j]:>10.3f}" for j in range(3))
    print(f"{variable[:9]:>10}  {row}")
print()
for variable in names:
    shock, share = fevd.dominant_shock(variable)
    print(f"  {variable[:9]:10s} mostly explained by {shock} ({share:.1%})")
print()

print("BOOTSTRAP BANDS (200 draws, this takes a moment)")
bands = bootstrap_bands(model, horizon=24, n_draws=200, seed=42)
print(f"  successful: {bands.n_successful}/{bands.n_draws}")
print()

low, point, high = bands.band("indpro", "fedfunds_shock", 0.90)
significant = bands.excludes_zero("indpro", "fedfunds_shock", 0.90)
print("  industrial production response to a monetary shock, 90% band")
print(f"{'h':>4}  {'lower':>10}  {'point':>10}  {'upper':>10}   excludes 0")
for h in (0, 3, 6, 12, 18, 24):
    print(
        f"{h:>4}  {low[h]:>10.5f}  {point[h]:>10.5f}  {high[h]:>10.5f}"
        f"   {'yes' if significant[h] else 'no'}"
    )