"""How well does the current calibration reproduce its declared targets?"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.economy.calibration import (
    URBAN_INDIA_MOMENTS,
    Confidence,
    evaluate,
    search,
)
from moirai.engine.economy.households import PopulationParameters, generate_population

configure_logging("ERROR")

print("DECLARED TARGETS")
for moment in URBAN_INDIA_MOMENTS:
    print(f"  {moment.name:<36} {moment.target:>7.3f}  [{moment.confidence.value}]")
    print(f"      source: {moment.source}")
print()

by_confidence = {c: 0 for c in Confidence}
for moment in URBAN_INDIA_MOMENTS:
    by_confidence[moment.confidence] += 1
print("target provenance:", {c.value: n for c, n in by_confidence.items()})
print()

print("=" * 92)
print("CURRENT CALIBRATION")
print("=" * 92)
current = generate_population(PopulationParameters(n_households=100_000, seed=1))
report = evaluate(current)
print(report.table())
print()
print(report.summary())
print()

print("=" * 92)
print("GRID SEARCH over income_wealth_correlation and log_wealth_sd")
print("=" * 92)
best, best_report, trials = search(
    {
        "log_wealth_mean": [8.0, 9.0, 10.0, 11.0, 12.0],
        "income_wealth_correlation": [0.5, 0.7, 0.85],
    },
    n_households=30_000,
)
print(f"evaluated {len(trials)} parameter combinations")
print()
print(best_report.table())
print()
print(f"best: log_wealth_mean {best.log_wealth_mean}, "
      f"income_wealth_correlation {best.income_wealth_correlation}")
print(best_report.summary())
print()

losses = np.array([loss for _, loss in trials])
print(f"loss range across the grid: {losses.min():.3f} to {losses.max():.3f}")
print("A narrow range would mean these moments do not identify these parameters.")