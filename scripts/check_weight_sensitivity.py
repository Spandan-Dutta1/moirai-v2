"""Does the Fed-RBI result survive plausible variation in assumed weights?"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import (
    FED,
    RBI,
    analytic_nash,
    sensitivity_report,
    weight_sensitivity,
)

configure_logging("ERROR")

# The scenario that matters: a US inflation shock forcing the Fed to tighten.
shocked_fed = FED.model_copy(update={"current_inflation": 0.045})
baseline = analytic_nash(shocked_fed, RBI)

print("BASELINE  (Fed inflation 4.5%)")
for name, rate in baseline.rates.items():
    print(f"  {name:<24} {rate:.3%}")
print()

print("=" * 76)
print("THE CENTRAL ASSUMPTION: the RBI's external weight")
print("=" * 76)
print("  Its published concern with capital flows is documented.")
print("  The magnitude, 0.40, is not. So: does it matter?\n")

sensitivity = weight_sensitivity(
    shocked_fed,
    RBI,
    vary="external_weight",
    on=RBI.name,
    values=np.linspace(0.0, 0.8, 9),
    observe=RBI.name,
)

print(f"  {'external weight':>16}  {'RBI rate':>10}  {'move':>9}")
for value, rate, move in zip(
    sensitivity.values, sensitivity.rates, sensitivity.moves, strict=True
):
    marker = "  <- assumed" if abs(value - 0.40) < 1e-9 else ""
    print(f"  {value:>16.2f}  {rate:>9.3%}  {move:>+8.0f}bp{marker}")

print()
print(f"  spread across the range: {sensitivity.spread_bp:.0f}bp")
print(f"  sign flips             : {sensitivity.sign_flips}")
print(f"  robust                 : {sensitivity.is_robust}")
print()

print("=" * 76)
print("DOES THE RBI's WEIGHT CHANGE WHAT THE FED DOES?")
print("=" * 76)

cross = weight_sensitivity(
    shocked_fed,
    RBI,
    vary="external_weight",
    on=RBI.name,
    values=np.linspace(0.0, 0.8, 9),
    observe=FED.name,
)
print(f"  Fed rate across the range: "
      f"{cross.rate_range[0]:.3%} to {cross.rate_range[1]:.3%}  "
      f"({cross.spread_bp:.0f}bp)")
print()
print("  A small spread here is the asymmetry, quantified. What the RBI")
print("  cares about barely moves the Fed; the reverse is not true.")
print()

print("=" * 76)
print("EVERY ASSUMED WEIGHT")
print("=" * 76)

report = sensitivity_report(shocked_fed, RBI)
print(f"  {report['n_robust']}/{report['n_checks']} checks robust")
print(f"  worst spread : {report['worst_spread_bp']:.0f}bp")
print(f"  any sign flip: {report['any_sign_flip']}")
print()

if report["fragile"]:
    print("  FRAGILE RESULTS")
    for item in report["fragile"]:
        print(
            f"    {item['parameter']:<40} -> {item['bank']:<24} "
            f"{item['spread_bp']:>6.0f}bp  flips={item['sign_flips']}"
        )
    print()
    print("  These conclusions depend on a weight nobody has measured and")
    print("  should be reported as ranges rather than as point estimates.")
else:
    print("  No fragile results: every conclusion survives the full range.")