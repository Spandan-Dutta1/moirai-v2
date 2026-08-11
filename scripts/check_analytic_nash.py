"""Exact equilibria, and what happens when the Fed is forced to tighten."""

from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import (
    FED,
    RBI,
    analytic_cooperative,
    analytic_nash,
    coordination_value,
)

configure_logging("ERROR")


def report(label: str, fed, rbi) -> None:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"  Fed inflation {fed.current_inflation:.1%} vs target {fed.inflation_target:.1%}")
    print(f"  RBI inflation {rbi.current_inflation:.1%} vs target {rbi.inflation_target:.1%}"
          f"  (band {rbi.tolerance_lower:.0%}-{rbi.tolerance_upper:.0%})")
    print()

    nash = analytic_nash(fed, rbi)
    cooperative = analytic_cooperative(fed, rbi)
    value = coordination_value(fed, rbi)

    print(f"  {'':26} {'NASH':>10}  {'COOPERATIVE':>13}  {'move':>8}")
    for bank in (fed, rbi):
        move = (nash.rates[bank.name] - bank.current_rate) * 10_000
        print(
            f"  {bank.name:<26} {nash.rates[bank.name]:>9.3%}  "
            f"{cooperative.rates[bank.name]:>12.3%}  {move:>+7.0f}bp"
        )

    print(f"\n  condition number      : {nash.condition_number:.1f} "
          f"(well conditioned: {nash.is_well_conditioned})")
    print(f"  gain from coordinating: {value['total_gain']:.6f}")
    for name, gain in value["gain_by_bank"].items():
        print(f"    {name:<24} {gain:+.6f}")
    if value["someone_loses"]:
        print("    one party loses from coordination, which is why it is not")
        print("    self-enforcing and needs a commitment mechanism")


report("CURRENT CONDITIONS  (August 2026)", FED, RBI)

# A US inflation shock forces the Fed to tighten, which is when the RBI
# actually faces its trade-off between the rupee and domestic growth.
shocked_fed = FED.model_copy(update={"current_inflation": 0.045})
report("US INFLATION SHOCK  (Fed inflation to 4.5%)", shocked_fed, RBI)

# And the case where India is also above its band.
stressed_rbi = RBI.model_copy(update={"current_inflation": 0.068})
report("BOTH ABOVE TARGET  (RBI breaches its 6% ceiling)", shocked_fed, stressed_rbi)

print(f"\n{'=' * 74}")
print("The RBI's response to a Fed tightening is the asymmetry this models.")
print("Its external weight is 0.40, derived from its own published concern")
print("with capital flows. The Fed's is 0.00. So a Fed move forces an RBI")
print("response, and an RBI move does not force a Fed one.")