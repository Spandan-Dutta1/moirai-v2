"""The network against calendar 2022, with nothing tuned (ADR 012).

The declared scenario `historical_2022` starts every bank from its
January 2022 rate and gives it its 2022 average inflation in its own
target measure. This script recomputes those conditions from the data,
refuses to run if they disagree with the declaration, solves the network
and compares the solved moves with what the banks actually did.

Inflation is computed from index levels, not taken from a summary. Two
exceptions, both stated in the output:

    Japan   FRED's OECD CPI stops in mid-2021, so the 2022 average
            (2.5 percent, Statistics Bureau, all items) is declared.
    India   The target measure is CPI Combined, which the warehouse
            holds as the RBI publishes it, year on year. The OECD index
            on FRED is printed beside it and is a different measure.

The network is linear while no bank leaves its band, so the caused move
splits exactly into one contribution per bank's condition. The RBI does
leave its band here, so the solved rate is checked against the true loss.
"""

from datetime import date

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import to_array
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    NetworkEquilibrium,
    SpilloverMatrix,
    _losses_at,
    _reaction_system,
    network_nash,
)
from moirai.engine.scenarios import (
    HISTORICAL_2022,
    JANUARY_2022_BANKS,
    OBSERVED_2022_MOVES_BP,
    Condition,
)

configure_logging("ERROR")

#: Declared conditions are rounded to a thousandth of a percentage point.
DECLARED_TOLERANCE = 0.00001

SHORT = {
    FED.name: "Fed",
    ECB.name: "ECB",
    BANK_OF_JAPAN.name: "BoJ",
    BANK_OF_ENGLAND.name: "BoE",
    RBI.name: "RBI",
}


# ---- data ------------------------------------------------------------------


def fetch(code: str, start: str, end: str) -> list[tuple[date, float]]:
    with FredAdapter() as fred:
        series = fred.fetch_series(code, observation_start=start, observation_end=end)
    periods, values = to_array(series)
    return [(p, float(v)) for p, v in zip(periods, values, strict=True) if v == v]


def average_inflation_2022(code: str) -> tuple[float, float, float]:
    """Mean of monthly year-on-year rates, annual-average ratio, and peak."""
    levels = dict(fetch(code, "2021-01-01", "2022-12-31"))
    months = [date(2022, m, 1) for m in range(1, 13)]
    prior = [date(2021, m, 1) for m in range(1, 13)]
    missing = [d for d in months + prior if d not in levels]
    if missing:
        raise SystemExit(f"{code}: missing {missing[0]} and {len(missing) - 1} more")
    yoy = [levels[a] / levels[b] - 1 for a, b in zip(months, prior, strict=True)]
    annual = np.mean([levels[d] for d in months]) / np.mean([levels[d] for d in prior]) - 1
    return float(np.mean(yoy)), float(annual), float(max(yoy))


def rate_on(code: str, when: date) -> float:
    """The last observation on or before a date, as a decimal."""
    observations = fetch(code, "2021-11-01", when.isoformat())
    return observations[-1][1] / 100


declared = {c.bank: c.inflation for c in HISTORICAL_2022.conditions}

print("=" * 76)
print("INFLATION, 2022, COMPUTED FROM INDEX LEVELS")
print("=" * 76)
print(f"  {'bank':<24} {'series':<20} {'mean y/y':>9} {'annual':>8} {'peak':>7}")

measured: dict[str, float] = {}
for bank, code in (
    (FED, "PCEPI"),
    (ECB, "CP0000EZ19M086NEST"),
    (BANK_OF_ENGLAND, "GBRCPIALLMINMEI"),
):
    mean, annual, peak = average_inflation_2022(code)
    measured[bank.name] = mean
    print(f"  {bank.name:<24} {code:<20} {mean:>9.2%} {annual:>8.2%} {peak:>7.2%}")

measured[BANK_OF_JAPAN.name] = declared[BANK_OF_JAPAN.name] or 0.0
print(f"  {BANK_OF_JAPAN.name:<24} {'(declared)':<20} {measured[BANK_OF_JAPAN.name]:>9.2%}")
print("    Statistics Bureau of Japan, CPI all items; FRED's JPNCPIALLMINMEI ends 2021-06")

with Warehouse(read_only=True) as warehouse:
    india = warehouse.get_series("in_cpi_inflation")
    repo = {
        o.period: o.value / 100
        for o in warehouse.get_series("in_repo_rate").observations
        if o.value is not None
    }
india_2022 = [
    o.value / 100
    for o in india.observations
    if o.period.year == 2022 and o.value is not None
]
if len(india_2022) != 12:
    raise SystemExit(f"in_cpi_inflation has {len(india_2022)} months of 2022, not 12")
measured[RBI.name] = float(np.mean(india_2022))
oecd_mean, _, oecd_peak = average_inflation_2022("INDCPIALLMINMEI")
print(
    f"  {RBI.name:<24} {'in_cpi_inflation':<20} {measured[RBI.name]:>9.2%} "
    f"{'':>8} {max(india_2022):>7.2%}"
)
print("    CPI Combined as the RBI publishes it, year on year (no index held)")
print(f"    OECD INDCPIALLMINMEI, a different measure: mean {oecd_mean:.2%}, "
      f"peak {oecd_peak:.2%}")
print("  mean y/y is the condition applied; annual is the ratio of annual averages")

disagreements = [
    f"{name}: declared {declared[name]}, measured {value:.5f}"
    for name, value in measured.items()
    if abs((declared[name] or 0.0) - value) > DECLARED_TOLERANCE
]
if disagreements:
    raise SystemExit("historical_2022 disagrees with the data:\n  " + "\n  ".join(disagreements))
print("  the declared conditions of historical_2022 match the data")

# ---- starting rates and observed moves -------------------------------------

START, END = date(2022, 1, 1), date(2022, 12, 31)
starting = {
    FED.name: rate_on("DFEDTARU", START),
    ECB.name: rate_on("ECBDFR", START),
    BANK_OF_ENGLAND.name: 0.0025,
    BANK_OF_JAPAN.name: rate_on("IRSTCI01JPM156N", START),
    RBI.name: repo[date(2022, 1, 1)],
}
ending = {
    FED.name: rate_on("DFEDTARU", END),
    ECB.name: rate_on("ECBDFR", END),
    BANK_OF_ENGLAND.name: 0.0350,
    BANK_OF_JAPAN.name: rate_on("IRSTCI01JPM156N", date(2022, 12, 1)),
    RBI.name: repo[date(2022, 12, 1)],
}
observed_bp = {name: (ending[name] - starting[name]) * 10_000 for name in starting}

print()
print("=" * 76)
print("POLICY RATES, JANUARY TO DECEMBER 2022")
print("=" * 76)
print(f"  {'bank':<24} {'Jan 2022':>9} {'Dec 2022':>9} {'observed':>9}")
for name in starting:
    print(
        f"  {name:<24} {starting[name]:>9.2%} {ending[name]:>9.2%} "
        f"{observed_bp[name]:>+8.0f}bp"
    )
sonia = rate_on("IUDSOIA", START), rate_on("IUDSOIA", END)
print("  Bank Rate is not on FRED; 0.25 and 3.50 percent are the published rates.")
print(f"    SONIA cross-check: {sonia[0]:.3%} to {sonia[1]:.3%}")

banks = JANUARY_2022_BANKS
for b in banks:
    if abs(b.current_rate - starting[b.name]) > 1e-9:
        raise SystemExit(f"{b.name}: declared start {b.current_rate}, data {starting[b.name]}")
    if abs(OBSERVED_2022_MOVES_BP[b.name] - observed_bp[b.name]) > 0.5:
        raise SystemExit(
            f"{b.name}: declared move {OBSERVED_2022_MOVES_BP[b.name]}, "
            f"data {observed_bp[b.name]:.1f}"
        )
print("  the declared starting rates and observed moves match the data")

# ---- the game ----------------------------------------------------------------

spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
conditions = HISTORICAL_2022.conditions


def solve(active: tuple[Condition, ...]) -> NetworkEquilibrium:
    only = HISTORICAL_2022.model_copy(update={"conditions": active})
    return network_nash(only.apply_to(banks), spillovers)


reference = solve(())
equilibrium = solve(conditions)
contributions = {c.bank: solve((c,)) for c in conditions}

print()
print("=" * 76)
print("MODEL AGAINST OBSERVED")
print("=" * 76)
print(
    f"  {'bank':<24} {'model':>8} {'unprompted':>11} {'caused':>8} "
    f"{'observed':>9} {'ratio':>6}"
)
model_bp: dict[str, float] = {}
for b in banks:
    rate = equilibrium.rates[b.name]
    model_bp[b.name] = (rate - b.current_rate) * 10_000
    unprompted = (reference.rates[b.name] - b.current_rate) * 10_000
    caused = (rate - reference.rates[b.name]) * 10_000
    observed = observed_bp[b.name]
    ratio = f"{model_bp[b.name] / observed:>6.2f}" if abs(observed) >= 25 else f"{'-':>6}"
    print(
        f"  {b.name:<24} {model_bp[b.name]:>+7.0f}bp {unprompted:>+10.0f}bp "
        f"{caused:>+7.0f}bp {observed:>+8.0f}bp {ratio}"
    )
print(f"  condition number {equilibrium.condition_number:.1f}")

print()
print("  caused move split by whose condition produced it (bp)")
header = "".join(f"{SHORT[c.bank]:>9}" for c in conditions)
print(f"  {'bank':<24}{header} {'sum':>8} {'residual':>9}")
for b in banks:
    parts = [
        (contributions[c.bank].rates[b.name] - reference.rates[b.name]) * 10_000
        for c in conditions
    ]
    caused = (equilibrium.rates[b.name] - reference.rates[b.name]) * 10_000
    cells = "".join(f"{p:>+9.1f}" for p in parts)
    print(f"  {b.name:<24}{cells} {sum(parts):>+8.1f} {caused - sum(parts):>+9.2e}")

# ---- the RBI-to-Fed ratio and where it comes from --------------------------

names = spillovers.names
i, f = names.index(RBI.name), names.index(FED.name)
fed_on_fed = (contributions[FED.name].rates[FED.name] - reference.rates[FED.name]) * 10_000
fed_on_rbi = (contributions[FED.name].rates[RBI.name] - reference.rates[RBI.name]) * 10_000

ordered = tuple(next(b for b in banks if b.name == n) for n in names)
matrix, _ = _reaction_system(ordered, spillovers)
demand = np.asarray(spillovers.demand)
exchange = np.asarray(spillovers.exchange)
rbi = next(b for b in banks if b.name == RBI.name)
own_inflation = -(spillovers.own_inflation_effect + exchange[i].sum())
terms = {
    "exchange channel (inflation)": 2 * rbi.inflation_weight * own_inflation * exchange[i, f],
    "demand channel (output)": (
        2 * rbi.output_weight * spillovers.own_output_effect * demand[i, f]
    ),
    "external differential": -2 * rbi.external_weight / (len(names) - 1),
}
assert abs(sum(terms.values()) - matrix[i, f]) < 1e-12

print()
print("=" * 76)
print("THE RBI-TO-FED RATIO")
print("=" * 76)
print("  like for like, total move against total move:")
print(f"    observed : {observed_bp[RBI.name]:.0f} / {observed_bp[FED.name]:.0f} = "
      f"{observed_bp[RBI.name] / observed_bp[FED.name]:.1%}")
print(f"    model    : {model_bp[RBI.name]:.0f} / {model_bp[FED.name]:.0f} = "
      f"{model_bp[RBI.name] / model_bp[FED.name]:.1%}")
print("  the Fed's share of the RBI's move, which the data cannot show:")
print(f"    model    : {fed_on_rbi:+.1f} / {fed_on_fed:+.1f} = {fed_on_rbi / fed_on_fed:.1%}")
print()
print("  RBI best-response slope to the Fed's rate, dr_RBI / dr_Fed, by channel")
slope = -matrix[i, f] / matrix[i, i]
for label, term in terms.items():
    print(f"    {label:<30} {-term / matrix[i, i]:>+7.3f}")
print(f"    {'total':<30} {slope:>+7.3f}")
print(f"  parameters: exchange[RBI,Fed] {exchange[i, f]:.3f}, demand[RBI,Fed] "
      f"{demand[i, f]:.3f}, external weight {rbi.external_weight:.2f} over "
      f"{len(names) - 1} banks, smoothing {rbi.smoothing_weight:.2f}")

# ---- the band --------------------------------------------------------------


def with_rbi_at(inflation: float) -> tuple[Condition, ...]:
    return tuple(
        c if c.bank != RBI.name else c.model_copy(update={"inflation": inflation})
        for c in conditions
    )


def rbi_realised_inflation(active: tuple[Condition, ...]) -> float:
    """The RBI's inflation at equilibrium, computed as the solver's loss computes it."""
    solved = solve(active)
    conditioned = HISTORICAL_2022.model_copy(update={"conditions": active}).apply_to(banks)
    by_name = {b.name: b for b in conditioned}
    moves = np.array([solved.rates[n] - by_name[n].current_rate for n in names])
    value = by_name[RBI.name].current_inflation - spillovers.own_inflation_effect * moves[i]
    value -= float(sum(exchange[i, m] * (moves[i] - moves[m]) for m in range(len(names))))
    return value


rbi_inflation = rbi_realised_inflation(conditions)

# Affine in the RBI's own condition while the solver is linear.
low, high = 0.05, 0.10
pi_low = rbi_realised_inflation(with_rbi_at(low))
pi_high = rbi_realised_inflation(with_rbi_at(high))
assert RBI.tolerance_upper is not None
threshold = low + (RBI.tolerance_upper - pi_low) * (high - low) / (pi_high - pi_low)

print()
print("=" * 76)
print("THE BAND AT EQUILIBRIUM")
print("=" * 76)
print(f"  RBI condition (2022 mean CPI)  : {measured[RBI.name]:.2%}")
print(f"  RBI inflation at equilibrium   : {rbi_inflation:.2%}")
print(f"  band                           : {RBI.tolerance_lower:.0%} to {RBI.tolerance_upper:.0%}")
print(f"  breaches at equilibrium        : {not RBI.within_band(rbi_inflation)}")
print(f"  condition above which it would : {threshold:.2%} (others at their 2022 conditions)")
print(f"  2022 peak CPI {max(india_2022):.2%} is a monthly peak; the model's condition is one")
print("    static level, so the annual mean is the comparable number")

conditioned = HISTORICAL_2022.apply_to(banks)
ordered_conditioned = tuple(next(b for b in conditioned if b.name == n) for n in names)
solved = np.array([equilibrium.rates[n] for n in names])


def rbi_loss(rate: float) -> float:
    trial = solved.copy()
    trial[i] = rate
    return _losses_at(ordered_conditioned, trial, spillovers)[RBI.name]


grid = solved[i] + np.arange(-3000, 3001) / 100_000
best = float(grid[int(np.argmin([rbi_loss(r) for r in grid]))])
print(f"  best reply under the true loss : {best:.3%} against solved {solved[i]:.3%} "
      f"({(best - solved[i]) * 10_000:+.1f}bp)")
print("  (ADR 011: the solver omits the band penalty; this checks the solved rate)")
