"""The fixed design every historical validation runs through (ADRs 012, 013).

A historical validation declares a scenario, the banks' starting state and
what the banks actually did. This module recomputes every condition,
starting rate and observed move from the data, refuses to run if any of
them disagrees with the declaration, solves the network untuned and
reports the result in one shape. Keeping the design in one place is what
makes it fixed: a second year cannot quietly use a different window, a
different average or a different comparison.

The design:

    window      one calendar year
    rates       the rate in force on 1 January and on 31 December
    condition   each bank's mean of the twelve monthly year-on-year
                inflation rates in its own target measure, computed from
                index levels where an index exists
    reference   the same network with every bank at its inflation target
                and a zero output gap
    solution    the simultaneous network equilibrium, sourced spillovers

The network is linear while no bank leaves its band, so the caused move
splits exactly into one contribution per bank's condition. Where the RBI
does leave its band, its solved rate is checked against its true loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import minimize_scalar

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
    CentralBank,
)
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    NetworkEquilibrium,
    SpilloverMatrix,
    _losses_at,
    _reaction_system,
    network_nash,
)
from moirai.engine.scenarios import Condition, Scenario

#: Declared conditions are rounded to a thousandth of a percentage point.
DECLARED_TOLERANCE = 0.00001

#: Observed moves are declared to the nearest tenth of a basis point.
OBSERVED_TOLERANCE_BP = 0.05

SHORT = {
    FED.name: "Fed",
    ECB.name: "ECB",
    BANK_OF_JAPAN.name: "BoJ",
    BANK_OF_ENGLAND.name: "BoE",
    RBI.name: "RBI",
}


# ---- where each number comes from -------------------------------------------


@dataclass(frozen=True)
class FredIndex:
    """A price index on FRED; inflation is computed from its levels."""

    code: str


@dataclass(frozen=True)
class WarehouseYearOnYear:
    """A published year-on-year rate in the warehouse, in percent."""

    series_id: str
    note: str


@dataclass(frozen=True)
class DeclaredInflation:
    """A figure that cannot be computed here, with its source."""

    source: str


@dataclass(frozen=True)
class FredRate:
    """The last observation on or before a date, in percent."""

    code: str
    on: date


@dataclass(frozen=True)
class WarehouseRate:
    """A monthly warehouse value, in percent, holding the month-end rate."""

    series_id: str
    month: date


@dataclass(frozen=True)
class DeclaredRate:
    """A rate not available here, with its source."""

    value: float
    source: str


InflationSource = FredIndex | WarehouseYearOnYear | DeclaredInflation
RateSource = FredRate | WarehouseRate | DeclaredRate


@dataclass(frozen=True)
class HistoricalValidation:
    """Everything a historical validation declares, and where it comes from."""

    year: int
    scenario: Scenario
    banks: tuple[CentralBank, ...]
    observed_bp: dict[str, float]
    inflation: dict[str, InflationSource]
    start: dict[str, RateSource]
    end: dict[str, RateSource]
    cross_checks: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


# ---- data ------------------------------------------------------------------


def fetch(code: str, start: str, end: str) -> list[tuple[date, float]]:
    with FredAdapter() as fred:
        series = fred.fetch_series(code, observation_start=start, observation_end=end)
    periods, values = to_array(series)
    return [(p, float(v)) for p, v in zip(periods, values, strict=True) if v == v]


def average_inflation(code: str, year: int) -> tuple[float, float, float]:
    """Mean of monthly year-on-year rates, annual-average ratio, and peak."""
    levels = dict(fetch(code, f"{year - 1}-01-01", f"{year}-12-31"))
    months = [date(year, m, 1) for m in range(1, 13)]
    prior = [date(year - 1, m, 1) for m in range(1, 13)]
    missing = [d for d in months + prior if d not in levels]
    if missing:
        raise SystemExit(f"{code}: missing {missing[0]} and {len(missing) - 1} more")
    yoy = [levels[a] / levels[b] - 1 for a, b in zip(months, prior, strict=True)]
    annual = np.mean([levels[d] for d in months]) / np.mean([levels[d] for d in prior]) - 1
    return float(np.mean(yoy)), float(annual), float(max(yoy))


def warehouse_year(series_id: str, year: int) -> list[float]:
    with Warehouse(read_only=True) as warehouse:
        series = warehouse.get_series(series_id)
    values = [
        o.value / 100
        for o in series.observations
        if o.period.year == year and o.value is not None
    ]
    if len(values) != 12:
        raise SystemExit(f"{series_id} has {len(values)} months of {year}, not 12")
    return values


def read_rate(source: RateSource) -> float:
    """A policy rate as a decimal."""
    if isinstance(source, DeclaredRate):
        return source.value
    if isinstance(source, FredRate):
        start = date(source.on.year - 1, 11, 1).isoformat()
        return fetch(source.code, start, source.on.isoformat())[-1][1] / 100
    with Warehouse(read_only=True) as warehouse:
        series = warehouse.get_series(source.series_id)
    match = next(
        (o for o in series.observations if o.period == source.month and o.value is not None),
        None,
    )
    if match is None or match.value is None:
        raise SystemExit(f"{source.series_id} has no value for {source.month}")
    return match.value / 100


def describe(source: RateSource) -> str:
    if isinstance(source, DeclaredRate):
        return f"declared: {source.source}"
    if isinstance(source, FredRate):
        return f"{source.code} on {source.on}"
    return f"{source.series_id}, {source.month:%B %Y} month-end"


# ---- the run ---------------------------------------------------------------


def run(spec: HistoricalValidation) -> None:
    configure_logging("ERROR")
    year = spec.year
    declared = {c.bank: c.inflation for c in spec.scenario.conditions}

    print("=" * 76)
    print(f"INFLATION, {year}, IN EACH BANK'S TARGET MEASURE")
    print("=" * 76)
    print(f"  {'bank':<24} {'series':<20} {'mean y/y':>9} {'annual':>8} {'peak':>7}")

    measured: dict[str, float] = {}
    for bank in spec.banks:
        source = spec.inflation[bank.name]
        if isinstance(source, FredIndex):
            mean, annual, peak = average_inflation(source.code, year)
            measured[bank.name] = mean
            print(f"  {bank.name:<24} {source.code:<20} {mean:>9.2%} {annual:>8.2%} "
                  f"{peak:>7.2%}")
        elif isinstance(source, WarehouseYearOnYear):
            values = warehouse_year(source.series_id, year)
            measured[bank.name] = float(np.mean(values))
            print(f"  {bank.name:<24} {source.series_id:<20} {measured[bank.name]:>9.2%} "
                  f"{'':>8} {max(values):>7.2%}")
            print(f"    {source.note}")
        else:
            measured[bank.name] = declared[bank.name] or 0.0
            print(f"  {bank.name:<24} {'(declared)':<20} {measured[bank.name]:>9.2%}")
            print(f"    {source.source}")

    for label, code in spec.cross_checks:
        mean, _, peak = average_inflation(code, year)
        print(f"  cross-check, {label} ({code}): mean {mean:.2%}, peak {peak:.2%}")
    print("  mean y/y is the condition applied; annual is the ratio of annual averages")

    disagreements = [
        f"{name}: declared {declared[name]}, measured {value:.5f}"
        for name, value in measured.items()
        if abs((declared[name] or 0.0) - value) > DECLARED_TOLERANCE
    ]
    if disagreements:
        raise SystemExit(
            f"{spec.scenario.name} disagrees with the data:\n  " + "\n  ".join(disagreements)
        )
    print(f"  the declared conditions of {spec.scenario.name} match the data")

    # ---- starting rates and observed moves ----
    starting = {b.name: read_rate(spec.start[b.name]) for b in spec.banks}
    ending = {b.name: read_rate(spec.end[b.name]) for b in spec.banks}
    observed_bp = {n: (ending[n] - starting[n]) * 10_000 for n in starting}

    print()
    print("=" * 76)
    print(f"POLICY RATES, 1 JANUARY TO 31 DECEMBER {year}")
    print("=" * 76)
    print(f"  {'bank':<24} {'start':>9} {'end':>9} {'observed':>9}")
    for b in spec.banks:
        n = b.name
        print(f"  {n:<24} {starting[n]:>9.3%} {ending[n]:>9.3%} {observed_bp[n]:>+8.1f}bp")
    for b in spec.banks:
        print(f"    {SHORT[b.name]}: {describe(spec.start[b.name])}; "
              f"{describe(spec.end[b.name])}")

    for b in spec.banks:
        if abs(b.current_rate - starting[b.name]) > 1e-9:
            raise SystemExit(
                f"{b.name}: declared start {b.current_rate}, data {starting[b.name]}"
            )
        if abs(spec.observed_bp[b.name] - observed_bp[b.name]) > OBSERVED_TOLERANCE_BP:
            raise SystemExit(
                f"{b.name}: declared move {spec.observed_bp[b.name]}, "
                f"data {observed_bp[b.name]:.2f}"
            )
    print("  the declared starting rates and observed moves match the data")

    for note in spec.notes:
        print(f"  {note}")

    # ---- the game ----
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    conditions = spec.scenario.conditions

    def solve(active: tuple[Condition, ...]) -> NetworkEquilibrium:
        only = spec.scenario.model_copy(update={"conditions": active})
        return network_nash(only.apply_to(spec.banks), spillovers)

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
    for b in spec.banks:
        rate = equilibrium.rates[b.name]
        model_bp[b.name] = (rate - b.current_rate) * 10_000
        unprompted = (reference.rates[b.name] - b.current_rate) * 10_000
        caused = (rate - reference.rates[b.name]) * 10_000
        observed = observed_bp[b.name]
        ratio = (
            f"{model_bp[b.name] / observed:>6.2f}" if abs(observed) >= 25 else f"{'-':>6}"
        )
        print(
            f"  {b.name:<24} {model_bp[b.name]:>+7.0f}bp {unprompted:>+10.0f}bp "
            f"{caused:>+7.0f}bp {observed:>+8.0f}bp {ratio}"
        )
    print(f"  condition number {equilibrium.condition_number:.1f}")
    print("  ratio is shown only where the observed move is at least 25bp")
    below_zero = [
        f"{SHORT[n]} {r:.2%}" for n, r in equilibrium.rates.items() if r < 0
    ]
    if below_zero:
        print(f"  solved below zero: {', '.join(below_zero)} (the model has no lower bound)")

    print()
    print("  caused move split by whose condition produced it (bp)")
    header = "".join(f"{SHORT[c.bank]:>9}" for c in conditions)
    print(f"  {'bank':<24}{header} {'sum':>8} {'residual':>9}")
    for b in spec.banks:
        parts = [
            (contributions[c.bank].rates[b.name] - reference.rates[b.name]) * 10_000
            for c in conditions
        ]
        caused = (equilibrium.rates[b.name] - reference.rates[b.name]) * 10_000
        cells = "".join(f"{p:>+9.1f}" for p in parts)
        print(f"  {b.name:<24}{cells} {sum(parts):>+8.1f} {caused - sum(parts):>+9.2e}")

    # ---- the RBI-to-Fed ratio and where it comes from ----
    names = spillovers.names
    i, f = names.index(RBI.name), names.index(FED.name)
    fed_on_fed = (contributions[FED.name].rates[FED.name] - reference.rates[FED.name]) * 1e4
    fed_on_rbi = (contributions[FED.name].rates[RBI.name] - reference.rates[RBI.name]) * 1e4

    ordered = tuple(next(b for b in spec.banks if b.name == n) for n in names)
    matrix, _ = _reaction_system(ordered, spillovers)
    demand = np.asarray(spillovers.demand)
    exchange = np.asarray(spillovers.exchange)
    rbi = next(b for b in spec.banks if b.name == RBI.name)
    own_inflation = -(spillovers.own_inflation_effect + exchange[i].sum())
    terms = {
        "exchange channel (inflation)": (
            2 * rbi.inflation_weight * own_inflation * exchange[i, f]
        ),
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
    if abs(observed_bp[FED.name]) >= 25:
        print(f"    observed : {observed_bp[RBI.name]:.0f} / {observed_bp[FED.name]:.0f} = "
              f"{observed_bp[RBI.name] / observed_bp[FED.name]:.1%}")
    else:
        print(f"    observed : {observed_bp[RBI.name]:.0f} / {observed_bp[FED.name]:.0f}, "
              f"undefined: the Fed did not move")
    print(f"    model    : {model_bp[RBI.name]:.0f} / {model_bp[FED.name]:.0f} = "
          f"{model_bp[RBI.name] / model_bp[FED.name]:.1%}")
    print("  the Fed's share of the RBI's move, which the data cannot show:")
    print(f"    model    : {fed_on_rbi:+.1f} / {fed_on_fed:+.1f} = "
          f"{fed_on_rbi / fed_on_fed:.1%}")
    print()
    print("  RBI best-response slope to the Fed's rate, dr_RBI / dr_Fed, by channel")
    slope = -matrix[i, f] / matrix[i, i]
    for label, term in terms.items():
        print(f"    {label:<30} {-term / matrix[i, i]:>+7.3f}")
    print(f"    {'total':<30} {slope:>+7.3f}")
    print(f"  parameters: exchange[RBI,Fed] {exchange[i, f]:.3f}, demand[RBI,Fed] "
          f"{demand[i, f]:.3f}, external weight {rbi.external_weight:.2f} over "
          f"{len(names) - 1} banks, smoothing {rbi.smoothing_weight:.2f}")

    # ---- the band ----
    def with_rbi_at(inflation: float) -> tuple[Condition, ...]:
        return tuple(
            c if c.bank != RBI.name else c.model_copy(update={"inflation": inflation})
            for c in conditions
        )

    def rbi_realised_inflation(active: tuple[Condition, ...]) -> float:
        """The RBI's inflation at equilibrium, as the solver's loss computes it."""
        solved = solve(active)
        scenario = spec.scenario.model_copy(update={"conditions": active})
        by_name = {b.name: b for b in scenario.apply_to(spec.banks)}
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
    print(f"  RBI condition ({year} mean CPI)  : {measured[RBI.name]:.2%}")
    print(f"  RBI inflation at equilibrium   : {rbi_inflation:.2%}")
    print(f"  band                           : {RBI.tolerance_lower:.0%} to "
          f"{RBI.tolerance_upper:.0%}")
    print(f"  breaches at equilibrium        : {not RBI.within_band(rbi_inflation)}")
    print(f"  condition above which it would : {threshold:.2%} "
          f"(others at their {year} conditions)")

    banks = spec.scenario.apply_to(spec.banks)
    ordered_conditioned = tuple(next(b for b in banks if b.name == n) for n in names)
    solved = np.array([equilibrium.rates[n] for n in names])

    def rbi_loss(rate: float) -> float:
        trial = solved.copy()
        trial[i] = rate
        return _losses_at(ordered_conditioned, trial, spillovers)[RBI.name]

    span = 0.05
    best = float(
        minimize_scalar(
            rbi_loss,
            bounds=(solved[i] - span, solved[i] + span),
            method="bounded",
            options={"xatol": 1e-9},
        ).x
    )
    at_edge = abs(abs(best - solved[i]) - span) < 1e-6
    print(f"  best reply under the true loss : {best:.3%} against solved {solved[i]:.3%} "
          f"({(best - solved[i]) * 10_000:+.1f}bp){' AT SEARCH BOUND' if at_edge else ''}")
    print("  (ADR 011: the solver omits the band penalty; this checks the solved rate)")
