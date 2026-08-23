"""Five central banks as a network."""


from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    SpilloverMatrix,
    network_nash,
    network_stackelberg,
    transmission_ranking,
)

configure_logging("ERROR")

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
spillovers = SpilloverMatrix.from_tiers(DEFAULT_TIERS)

print("SYSTEMIC TIERS")
for name, tier in DEFAULT_TIERS.items():
    print(f"  {name:<24} {tier.value:<10} "
          f"out {tier.outward_strength:.2f}  in {tier.inward_sensitivity:.2f}")
print()

print("SPILLOVER MATRIX  (effect of column on row, demand channel)")
short = [n.split()[-1][:8] for n in spillovers.names]
print("  " + " " * 10 + "".join(f"{s:>10}" for s in short))
for i, name in enumerate(short):
    row = "".join(f"{spillovers.demand[i, j]:>10.3f}" for j in range(len(short)))
    print(f"  {name:<10}{row}")
print()
print(f"  symmetric: {spillovers.is_symmetric}  (it should not be)")
print()

print("  outward influence (what each bank exports)")
for name, value in sorted(
    spillovers.outward_influence().items(), key=lambda kv: -kv[1]
):
    print(f"    {name:<24} {value:.3f}")
print()
print("  inward exposure (what each economy absorbs)")
for name, value in sorted(spillovers.inward_exposure().items(), key=lambda kv: -kv[1]):
    print(f"    {name:<24} {value:.3f}")
print()

print("=" * 70)
print("SIMULTANEOUS EQUILIBRIUM")
print("=" * 70)
equilibrium = network_nash(BANKS, spillovers)
banks_by_name = {b.name: b for b in BANKS}
moves = equilibrium.moves_bp(banks_by_name)

print(f"  {'bank':<24} {'current':>9} {'equilibrium':>12} {'move':>8}")
for name, rate in equilibrium.rates.items():
    print(
        f"  {name:<24} {banks_by_name[name].current_rate:>8.2%} "
        f"{rate:>11.3%} {moves[name]:>+7.0f}bp"
    )
print()
print(f"  condition number: {equilibrium.condition_number:.1f} "
      f"(well conditioned: {equilibrium.is_well_conditioned})")
print()

print("=" * 70)
print("FED LEADS")
print("=" * 70)
led = network_stackelberg(BANKS, spillovers, FED.name)
for name, rate in led.rates.items():
    difference = (rate - equilibrium.rates[name]) * 10_000
    print(f"  {name:<24} {rate:>9.3%}   {difference:>+6.0f}bp vs simultaneous")
print()

print("=" * 70)
print("TRANSMISSION: a two point US inflation shock")
print("=" * 70)
ranking = transmission_ranking(BANKS, spillovers, FED.name)
print(f"  {'bank':<24} {'response':>10} {'pass-through':>14}")
for name, response in sorted(
    ranking["responses_bp"].items(), key=lambda kv: -abs(kv[1])
):
    through = ranking["pass_through"].get(name)
    shown = f"{through:.1%}" if through is not None else "origin"
    print(f"  {name:<24} {response:>+9.0f}bp {shown:>14}")
print()
print("  The Fed exports its stance; the RBI absorbs it. That asymmetry is")
print("  the tiering, and the tiering is an assumption marked as one.")
print()
print(f"  spillover matrix confidence: {spillovers.confidence.value}")