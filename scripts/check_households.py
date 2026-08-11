"""Generate a population and inspect its distribution."""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.economy.households import (
    PopulationParameters,
    generate_population,
)

configure_logging("WARNING")

population = generate_population(PopulationParameters(n_households=1_000_000))

print(population)
print()
for key, value in population.summary().items():
    print(f"  {key:26s} {value}")
print()

print("INCOME QUINTILES")
quintile = population.quantile_groups(population.income, 5)
header = f"{'group':>8}  {'mean income':>14}  {'mean wealth':>14}"
print(header + f"  {'% indebted':>11}  {'% floating':>11}")
for g in range(5):
    mask = quintile == g
    print(
        f"{g + 1:>8}  {population.income[mask].mean():>14,.0f}  "
        f"{population.wealth[mask].mean():>14,.0f}  "
        f"{population.is_indebted[mask].mean():>10.1%}  "
        f"{population.is_rate_exposed[mask].mean():>10.1%}"
    )
print()

print("WHO IS EXPOSED TO A RATE RISE")
exposed = population.subset(population.is_rate_exposed)
print(f"  households      : {len(exposed):,} ({len(exposed) / len(population):.1%})")
print(f"  mean income     : {exposed.income.mean():,.0f}")
print(f"  population mean : {population.income.mean():,.0f}")
print(f"  mean debt       : {exposed.debt.mean():,.0f}")
print(f"  mean debt/income: {np.median(exposed.debt_to_income):.2f}")
print(f"  mean age        : {exposed.age.mean():.1f}")
print()

print("Reproducibility check")
a = generate_population(PopulationParameters(n_households=10_000, seed=7))
b = generate_population(PopulationParameters(n_households=10_000, seed=7))
print("  same seed gives identical population:", np.array_equal(a.income, b.income))
c = generate_population(PopulationParameters(n_households=10_000, seed=8))
print("  different seed differs             :", not np.array_equal(a.income, c.income))