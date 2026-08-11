"""Solve the Fed and RBI as a Stackelberg pair."""

from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import (
    FED,
    MAJOR_CENTRAL_BANKS,
    RBI,
    SpilloverParameters,
    analyse_pair,
    build_policy_game,
)
from moirai.engine.financial.game import (
    nash_equilibria,
    payoff_surface_is_flat,
    stackelberg_equilibrium,
)

configure_logging("ERROR")

print("PUBLISHED MANDATES")
print(f"  {'bank':<24} {'target':>8}  {'band':>12}  {'mandate':<20} {'w_out':>6} {'w_ext':>6}")
for bank in MAJOR_CENTRAL_BANKS:
    band = (
        f"{bank.tolerance_lower:.0%}-{bank.tolerance_upper:.0%}"
        if bank.has_band
        else "none"
    )
    print(
        f"  {bank.name:<24} {bank.inflation_target:>7.1%}  {band:>12}  "
        f"{bank.mandate.value:<20} {bank.output_weight:>6.2f} {bank.external_weight:>6.2f}"
    )
print()
print("  the Fed is the only dual mandate; only 15-20% of central banks have one")
print("  the RBI is the only one with a legislated tolerance band")
print()

print("=" * 78)
print("FED AND RBI")
print("=" * 78)

game = build_policy_game(FED, RBI, title="Fed-RBI")
print(f"  {game.n_profiles} rate profiles on the grid\n")

equilibria = nash_equilibria(game)
print(f"NASH  ({len(equilibria)} equilibria)")
for outcome in equilibria[:3]:
    actions = {k: f"{v:.2%}" for k, v in outcome.actions.items()}
    print(f"  {actions}")
print()

stackelberg = stackelberg_equilibrium(game, FED.name)
print("STACKELBERG  (Fed moves first)")
for name, rate in stackelberg.actions.items():
    print(f"  {name:<24} {rate:.2%}")
print(f"  {stackelberg.note}")
print()

result = analyse_pair(FED, RBI)
gain = result["cooperation"]
print("COOPERATION")
print(f"  cooperative actions: "
      f"{ {k: f'{v:.2%}' for k, v in gain['cooperative_actions'].items()} }")
print(f"  nash actions       : "
      f"{ {k: f'{v:.2%}' for k, v in gain['nash_actions'].items()} }")
print(f"  gain from coordinating: {gain['gain']:.4f}")
print()

print("PAYOFF SURFACE")
for bank in (FED, RBI):
    flat = payoff_surface_is_flat(game, bank.name)
    print(f"  {bank.name:<24} flat: {flat}")
print("  a flat surface means the equilibrium is tie-breaking, not a choice")
print()

print("ASYMMETRY CHECK")
print(f"  RBI external weight : {RBI.external_weight}  ({RBI.weight_confidence.value})")
print(f"  Fed external weight : {FED.external_weight}  ({FED.weight_confidence.value})")
print()
print("  " + RBI.weight_note[:200] + "...")