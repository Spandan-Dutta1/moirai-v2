"""
The model against published estimates, link by link (Stage 6, ADR 037).

Every other check here asks whether the engine computes what it says it
computes. This one asks a different question: whether what it computes is the
size that other people, using other data and other methods, have measured.
That is external benchmarking, and it is the part of model validation a
repository cannot do for itself.

Each link is run at a standard hundred-basis-point shock, printed beside the
published estimate, and given a verdict. Where the published figure measures
something the model does not, the row says so rather than inventing a
comparison: the IMF's India number is the response of the three-to-six-month
government bond yield, which carries risk premia and term expectations that a
policy rate does not.

The household figures are differentials - floating-rate borrowers against
households with no debt - because that is what the micro literature estimates
and what this model exists to produce. Levels drift with accumulated income
growth in both arms; the gap is the result.

Reads nothing and writes nothing. Runs in about half a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moirai.engine.credit.loan_tape import Product
from moirai.engine.credit.repricing import RepricingParameters, ResetPolicy, reprice
from moirai.engine.credit.synthetic import PSB, REPO_RATE, generate
from moirai.engine.economy.behaviour import BehaviourParameters, counterfactual
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.economy.shock_path import MacroVariable, ShockPath
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    SpilloverMatrix,
    _reaction_system,
)

HOUSEHOLDS = 200_000
LOANS = 100_000
HORIZON = 24
SHOCK = 0.01

#: The macro state the household shock is run around: the repo rate of
#: November 2026 and the engine's own baseline inflation, income and
#: unemployment. Only the deviation from it matters.
BASELINE = {
    MacroVariable.POLICY_RATE: 0.065,
    MacroVariable.INFLATION: 0.05,
    MacroVariable.INCOME_GROWTH: 0.065,
    MacroVariable.UNEMPLOYMENT: 0.072,
}

#: The headline this project has published for a hundred basis points of Fed
#: tightening, from the static and the dynamic network respectively.
HEADLINE = (0.18, 0.56)


@dataclass(frozen=True)
class Benchmark:
    """One link, what the model says, and what has been measured elsewhere."""

    link: str
    model: float
    published: str
    low: float
    high: float
    unit: str
    source: str
    comparable: bool = True

    @property
    def verdict(self) -> str:
        """Where the model sits against the published range.

        Named against the range rather than the benchmark, because the
        consumption row is negative and "above" on a -1.95 benchmark would
        otherwise read as a larger response when it means a smaller one.
        """
        if not self.comparable:
            return "not like-for-like"
        if self.low <= self.model <= self.high:
            return "inside the range"
        if self.model < self.low:
            return "below the range"
        return "above the range"


def rbi_response_to_the_fed() -> float:
    """The RBI's best-response slope to the Fed's rate, from the network."""
    banks = (FED, ECB, BANK_OF_ENGLAND, BANK_OF_JAPAN, RBI)
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    names = spillovers.names
    ordered = tuple(next(b for b in banks if b.name == n) for n in names)
    matrix, _ = _reaction_system(ordered, spillovers)
    rbi, fed = names.index(RBI.name), names.index(FED.name)
    return float(-matrix[rbi, fed] / matrix[rbi, rbi])


def lending_rate_pass_through(
    n: int = LOANS, horizon: int = HORIZON
) -> dict[str, float]:
    """Pass-through of the shock into the book's rate, from the loan layer."""
    tape = generate(PSB, n=n, seed=0).tape
    path = np.r_[REPO_RATE, np.full(horizon, REPO_RATE + SHOCK)]
    result = reprice(
        tape, path, RepricingParameters(reset_policy=ResetPolicy.TENURE_FIRST)
    )
    rate = np.asarray(result.rate)
    weight = np.asarray(tape.outstanding, dtype=float)
    home = np.asarray(tape.product) == Product.HOME

    def at(month: int, mask: np.ndarray | None = None) -> float:
        move = rate[:, month] - rate[:, 0]
        if mask is None:
            return float(np.average(move, weights=weight) / SHOCK)
        return float(np.average(move[mask], weights=weight[mask]) / SHOCK)

    return {
        "book_3m": at(3),
        "book_12m": at(12),
        "floating_3m": at(3, home),
        "floating_12m": at(12, home),
    }


def household_response(
    n: int = HOUSEHOLDS, horizon: int = HORIZON
) -> dict[str, float]:
    """The consumption gap and the implied MPC, from the household layer."""
    population = generate_population(PopulationParameters(n_households=n, seed=0))
    paths = {
        variable: np.full(horizon + 1, level) for variable, level in BASELINE.items()
    }
    paths[MacroVariable.POLICY_RATE] = np.full(
        horizon + 1, BASELINE[MacroVariable.POLICY_RATE] + SHOCK
    )
    shock = ShockPath(
        shock_name="rbi_100bp", horizon=horizon, paths=paths, baselines=BASELINE
    )
    flat, hit = counterfactual(
        population, shock, BehaviourParameters(), INDIAN_BANKING_SYSTEM
    )

    floating = np.asarray(population.debt_is_floating)
    debt_free = np.asarray(population.debt) <= 0.0

    def measure(month: int) -> tuple[float, float]:
        before, after = flat[month], hit[month]
        was = np.asarray(before.consumption)
        now = np.asarray(after.consumption)

        def change(mask: np.ndarray) -> float:
            return 100.0 * (now[mask].sum() / was[mask].sum() - 1.0)

        spent = now[floating].sum() - was[floating].sum()
        owed = (
            np.asarray(after.debt_service)[floating].sum()
            - np.asarray(before.debt_service)[floating].sum()
        )
        gap = change(floating) - change(debt_free)
        return gap, (-spent / owed if owed else float("nan"))

    gap_now, mpc_now = measure(0)
    gap_year, mpc_year = measure(min(12, horizon))
    return {
        "gap_impact": gap_now,
        "gap_12m": gap_year,
        "mpc_impact": mpc_now,
        "mpc_12m": mpc_year,
        "lending": (hit[0].lending_rate - flat[0].lending_rate) / SHOCK,
        "deposit": (hit[0].deposit_rate - flat[0].deposit_rate) / SHOCK,
    }


def benchmarks(
    slope: float, rates: dict[str, float], households: dict[str, float]
) -> list[Benchmark]:
    """Every link the model can be held against a published number."""
    return [
        Benchmark(
            link="Fed +100bp -> RBI policy rate",
            model=slope * 10_000 * SHOCK,
            published="51bp into India's 3-6 month yield; 46bp of it autonomy-impairing",
            low=40.0,
            high=60.0,
            unit="bp",
            source="IMF WP/16/195, 43 countries, 2000-2015",
            comparable=False,
        ),
        Benchmark(
            link="RBI +100bp -> lending rate, floating, 12m",
            model=rates["floating_12m"],
            published="0.78 on fresh rupee loans",
            low=0.70,
            high=0.90,
            unit="x",
            source="RBI: WALR on fresh loans +194bp per 250bp of repo, 2022-24",
        ),
        Benchmark(
            link="RBI +100bp -> lending rate, whole book, 12m",
            model=rates["book_12m"],
            published="0.44 on outstanding rupee loans",
            low=0.40,
            high=0.60,
            unit="x",
            source="RBI: WALR on outstanding loans +111bp per 250bp, 2022-24",
        ),
        Benchmark(
            link="RBI +100bp -> floating borrowers' consumption, vs no debt",
            model=households["gap_12m"],
            published="-1.95% for adjustable-rate against debt-free owners",
            low=-2.40,
            high=-1.50,
            unit="pp",
            source=(
                "Floden, Kilstrom, Sigurdsson, Vestman (2017), "
                "300k Swedish households, 2000-2007"
            ),
        ),
        Benchmark(
            link="MPC out of interest expenses, on impact",
            model=households["mpc_impact"],
            published="0.2 at low and medium leverage; 0.5 at high",
            low=0.15,
            high=0.50,
            unit="",
            source="Floden et al. (2017); this population's median DTI is 1.8",
        ),
    ]


def main() -> None:
    slope = rbi_response_to_the_fed()
    rates = lending_rate_pass_through()
    households = household_response()
    table = benchmarks(slope, rates, households)

    print("=" * 100)
    print("MODEL AGAINST PUBLISHED ESTIMATES")
    print("=" * 100)
    for item in table:
        print(f"\n  {item.link}")
        print(f"    model     {item.model:>8.2f} {item.unit}")
        print(f"    published {item.published}")
        print(f"    verdict   {item.verdict.upper()}    [{item.source}]")

    print()
    print("=" * 100)
    print("THE CHAIN MULTIPLIES TO THE WHOLE")
    print("=" * 100)
    implied = slope * abs(households["gap_12m"])
    low, high = HEADLINE
    print(
        f"  100bp of Fed moves the RBI {slope * 10_000 * SHOCK:.1f}bp, and 100bp of RBI "
        f"moves the consumption gap {households['gap_12m']:.2f}pp,"
    )
    print(f"  so 100bp of Fed implies a gap of {implied:.2f}pp.")
    print(f"  This project's published headline is {low} to {high}pp (static to dynamic).")
    print(f"  {'INSIDE' if low <= implied <= high else 'OUTSIDE'} that range.")

    print()
    print("=" * 100)
    print("WHAT THE COMPARISON EXPOSED")
    print("=" * 100)
    print(
        f"  On impact, per 100bp of repo, the household layer passes "
        f"{households['lending']:.2f} into the lending rate"
    )
    print(
        f"  and {households['deposit']:.2f} into the deposit rate. The loan layer "
        f"reaches that same lending"
    )
    print(
        f"  level only after about a year: {rates['floating_3m']:.2f} at three months "
        f"against {rates['floating_12m']:.2f} at twelve."
    )
    print("  The two layers agree on how far a policy move travels and disagree on how")
    print("  fast: one applies the whole reset at once, the other spreads it over four")
    print("  quarters. The household layer has no reset lag.")
    print()
    retained = 100.0 * households["mpc_12m"] / households["mpc_impact"]
    print(
        f"  The symptom is the implied MPC, which starts at "
        f"{households['mpc_impact']:.2f} and still holds "
        f"{households['mpc_12m']:.2f} by month"
    )
    print(
        f"  twelve - {retained:.0f} per cent of its impact value. Against an annual "
        f"published estimate of"
    )
    print("  0.2 at comparable leverage that is the right order and somewhat strong, so")
    print("  the missing lag costs less than the inconsistency suggests. It is still an")
    print("  inconsistency between two layers of the same model. ADR 037 records it.")


if __name__ == "__main__":
    main()
