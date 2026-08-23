"""Does the modelled banking system reproduce RBI's published transmission?"""

from moirai.core.logging import configure_logging
from moirai.engine.financial.commercial_banks import (
    EASING_TARGETS,
    INDIAN_BANKING_SYSTEM,
    BankGroup,
    evaluate_transmission,
)

configure_logging("ERROR")

system = INDIAN_BANKING_SYSTEM

print("THE SYSTEM")
print(f"  {len(system.banks)} banks, {system.total_assets / 1e5:,.1f} lakh crore assets")
for group in BankGroup:
    banks = system.by_group(group)
    if banks:
        print(f"  {group.value:<10} {len(banks):>2} banks, "
              f"{system.group_share(group):>6.1%} of system credit")
print()

print("BANK CHARACTERISTICS")
print(f"  {'bank':<16} {'group':<9} {'EBLR':>6} {'retail':>7} {'NPA':>6} "
      f"{'CAR':>6} {'stress':>7}")
for bank in system.banks:
    print(
        f"  {bank.name:<16} {bank.group.value:<9} {bank.eblr_share:>6.2f} "
        f"{bank.retail_deposit_share:>7.2f} {bank.npa_ratio:>6.1%} "
        f"{bank.capital_ratio:>6.1%} {bank.stress:>7.2f}"
    )
print()

print("PASS-THROUGH BY GROUP")
print(f"  {'group':<10} {'lend tight':>11} {'lend ease':>10} "
      f"{'dep tight':>10} {'dep ease':>9}")
for group in BankGroup:
    if not system.by_group(group):
        continue
    print(
        f"  {group.value:<10} "
        f"{system.weighted_lending_pass_through(tightening=True, group=group):>11.1%} "
        f"{system.weighted_lending_pass_through(tightening=False, group=group):>10.1%} "
        f"{system.weighted_deposit_pass_through(tightening=True, group=group):>10.1%} "
        f"{system.weighted_deposit_pass_through(tightening=False, group=group):>9.1%}"
    )
print()
print("  Lending pass-through is higher when tightening than easing: banks")
print("  raise rates faster than they cut them. That asymmetry is why a")
print("  tightening cycle hurts more than an easing cycle helps.")
print()

print("AGAINST THE PUBLISHED FIGURES")
print(f"  source: {EASING_TARGETS[0].source}")
print()
evaluation = evaluate_transmission(system)
print(f"  {'group':<10} {'target':>8} {'modelled':>10} {'gap':>8}  {'':4} channel")
print("  " + "-" * 56)
for result in evaluation["results"]:
    for channel in ("lending", "deposit"):
        mark = "ok" if result[f"{channel}_passed"] else "MISS"
        print(
            f"  {result['group']:<10} {result[f'{channel}_target']:>8.2f} "
            f"{result[f'{channel}_modelled']:>10.2f} "
            f"{result[f'{channel}_gap']:>+8.2f}  {mark:<4} {channel}"
        )
print()
print(f"  {evaluation['n_passed']}/{evaluation['n_checks']} within tolerance, "
      f"loss {evaluation['loss']:.4f}")
print()

print("WHAT A HOUSEHOLD FACES  (repo 5.25% to 6.25%)")
before = system.effective_rates(0.0525, 0.0525)
after = system.effective_rates(0.0625, 0.0525)
print(f"  {'':16} {'before':>9} {'after':>9} {'change':>9}")
for label, key in (("lending rate", "lending_rate"), ("deposit rate", "deposit_rate")):
    print(
        f"  {label:<16} {before[key]:>9.3%} {after[key]:>9.3%} "
        f"{(after[key] - before[key]) * 10000:>+8.0f}bp"
    )
print()
print(f"  of a 100bp repo rise, borrowers absorb "
      f"{(after['lending_rate'] - before['lending_rate']) * 10000:.0f}bp and savers "
      f"receive {(after['deposit_rate'] - before['deposit_rate']) * 10000:.0f}bp")
print("  the difference accrues to the banking system as margin")