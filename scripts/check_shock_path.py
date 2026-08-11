"""Estimate a VAR, identify it, and translate the IRF into a household-facing path."""

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.economy.shock_path import (
    MacroVariable,
    build_shock_path,
    monetary_mappings,
)

configure_logging("WARNING")

CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]

prepared = []
with FredAdapter() as fred:
    for code in CODES:
        prepared.append(
            prepare(
                fred.fetch_series(
                    code, observation_start="1985-01-01", observation_end="2007-06-01"
                )
            )
        )

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)

# More lags this time: 6 failed the portmanteau test.
var = estimate_var(data, names, n_lags=12, periods=periods)
# Test lags must exceed model lags for the statistic to have degrees of
# freedom. With 12 model lags, testing at 24 leaves 12 * k^2 = 108.
report = diagnose(var, portmanteau_lags=24)
print("diagnostics:", report.summary())
print("usable     :", report.is_usable)
print()

model = identify_cholesky(var, ordering=names)
responses = impulse_responses(model, horizon=36)

mappings = monetary_mappings(
    rate_variable="fedfunds",
    price_variable="cpiaucsl",
    output_variable="indpro",
)

path = build_shock_path(
    responses,
    "fedfunds_shock",
    mappings,
    scale=1.0,
    diagnostics=report,
    require_usable=True,
)

print("MACRO PATH AFTER A ONE STANDARD DEVIATION TIGHTENING")
print(f"{'period':>7}  {'policy rate':>12}  {'inflation':>11}  {'income growth':>14}")
for period in (0, 3, 6, 12, 24, 36):
    state = path.at(period)
    print(
        f"{period:>7}  {state[MacroVariable.POLICY_RATE]:>11.3%}  "
        f"{state[MacroVariable.INFLATION]:>10.3%}  "
        f"{state[MacroVariable.INCOME_GROWTH]:>13.3%}"
    )
print()

for variable in path.paths:
    period, deviation = path.peak(variable)
    print(f"  peak {variable.value:16s} {deviation:+.4%} at period {period}")
print()

doubled = path.rescale(2.0)
print("scaled to two standard deviations:")
print(f"  peak policy rate {doubled.peak(MacroVariable.POLICY_RATE)[1]:+.4%}")
print()
print("ledger entry records the assumption and the diagnostics:")
ledger = path.to_ledger_dict()
print(f"  scheme            : {ledger['scheme']}")
print(f"  diagnostics usable: {ledger['diagnostics_usable']}")