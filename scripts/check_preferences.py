"""Estimate the Fed's revealed preferences from its actual decisions.

Fitted to the change in the policy rate rather than its level. An earlier
version fitted the level and the optimiser found that a lagged dependent
variable explains ninety percent of a highly persistent series, so it drove
the smoothing parameter to its bound and the policy coefficients to zero.
Explaining changes is harder and the R-squared is much lower, which is the
honest measure of how much of policy a simple rule accounts for.
"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import to_array
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.financial.central_banks import FED
from moirai.engine.financial.preferences import compare_detrending

configure_logging("ERROR")

# The Great Moderation, where the Taylor rule is known to fit well.
START, END = "1985-01-01", "2007-06-01"

with FredAdapter() as fred:
    funds = fred.fetch_series("FEDFUNDS", observation_start=START, observation_end=END)
    cpi = fred.fetch_series("CPIAUCSL", observation_start=START, observation_end=END)
    ip = fred.fetch_series("INDPRO", observation_start=START, observation_end=END)

_, rate_values = to_array(funds)
cpi_periods, cpi_values = to_array(cpi)
_, ip_values = to_array(ip)

# Year-on-year inflation from the index; drop the first twelve months.
inflation = np.log(cpi_values[12:] / cpi_values[:-12])
rates = rate_values[12:] / 100.0
output = np.log(ip_values[12:])

n = min(len(rates), len(inflation), len(output))
rates, inflation, output = rates[:n], inflation[:n], output[:n]

# Aggregate to quarterly. The monthly specification failed to identify the
# inflation response: month-to-month rate changes are dominated by meeting
# timing rather than by gaps that barely move over a month.
from moirai.engine.financial.preferences import to_quarterly

q_rates = to_quarterly(rates)
q_inflation = to_quarterly(inflation)
q_output = to_quarterly(output)

print(f"sample: {cpi_periods[12]} to {cpi_periods[12 + n - 1]}")
print(f"  {n} months aggregated to {len(q_rates)} quarters")
print(f"  mean funds rate {q_rates.mean():.2%}, mean inflation {q_inflation.mean():.2%}")
print()

estimates = compare_detrending(
    "Federal Reserve",
    q_rates,
    q_inflation,
    q_output,
    inflation_target=FED.inflation_target,
    neutral_real_rate=0.02,
    sample_start=str(cpi_periods[12]),
    sample_end=str(cpi_periods[12 + n - 1]),
)

print("ESTIMATED POLICY RULE")
print(f"  target = r* + pi + a*(pi - pi*) + b*gap")
print(f"  observed change = speed * (target - previous rate)")
print()
print(f"  {'detrending':<12} {'a (infl)':>9} {'b (out)':>9} {'speed':>7} "
      f"{'b/a':>7} {'R2':>7} {'interior':>9}")
print("  " + "-" * 64)
for name, estimate in estimates.items():
    print(
        f"  {name:<12} {estimate.inflation_weight:>9.3f} "
        f"{estimate.output_weight:>9.3f} {estimate.smoothing_weight:>7.3f} "
        f"{estimate.output_to_inflation_ratio:>7.3f} "
        f"{estimate.r_squared:>7.3f} {str(estimate.converged):>9}"
    )
print()

linear = estimates["linear"]

print("TAYLOR PRINCIPLE")
print(f"  satisfied: {linear.satisfies_taylor_principle}")
print("  A bank failing it raises nominal rates by less than inflation, so")
print("  the real rate falls when inflation rises. Clarida, Gali and Gertler")
print("  argued the pre-Volcker Fed failed this and that it explains the")
print("  Great Inflation.")
print()

print("AGAINST THE CANONICAL TAYLOR RULE")
print(f"  {'':<22} {'a':>7} {'b':>7} {'b/a':>7}")
print(f"  {'Taylor (1993)':<22} {0.5:>7.3f} {0.5:>7.3f} {1.0:>7.3f}")
print(
    f"  {'estimated, linear':<22} {linear.inflation_weight:>7.3f} "
    f"{linear.output_weight:>7.3f} {linear.output_to_inflation_ratio:>7.3f}"
)
print()

print("AGAINST THE ASSUMED WEIGHTS")
print(f"  assumed output weight in central_banks.py : {FED.output_weight:.3f}"
      f"  [{FED.weight_confidence.value}]")
print(f"  estimated output response                 : {linear.output_weight:.3f}"
      f"  [{linear.confidence.value}]")
print()
print("  These are not the same quantity. The assumed weight enters a loss")
print("  function; the estimated response is a reduced-form coefficient in a")
print("  rule. They are related through the transmission parameters, so the")
print("  comparison is directional rather than exact.")
print()

spread = abs(estimates["linear"].output_weight - estimates["hp"].output_weight)
print(f"DETRENDING DISAGREEMENT: {spread:.3f}")
if spread > 0.2:
    print("  large: the output gap construction is doing real work, so the")
    print("  estimate should be quoted as a range rather than a point")
else:
    print("  small: the estimate is not sensitive to the detrending choice")
print()

if not all(e.converged for e in estimates.values()):
    print("WARNING: at least one estimate sits at a parameter bound, so it is")
    print("not an interior optimum and should not be quoted as an estimate.")