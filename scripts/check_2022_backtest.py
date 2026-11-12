"""The 2022-23 tightening, against what actually happened (Stage 4, ADR 035).

The April 2022 book through the real repo path, scored on the one outcome
public data gives: housing asset quality improved through 250bp of tightening.
Housing GNPA at the HFCs fell from 4.3 percent in March 2022 to 3.3 in March
2023 and 2.2 in March 2024, and retail GNPA from 2.1 percent in June 2022 to
1.2 in March 2024.

Three things the engine does not have drove that outcome, and only one of them
can be put back. The book grew by a third, which cuts any ratio mechanically,
and the engine has no credit growth. Which borrowers broke is not published.
But income is in the engine's hands: it was held constant, and India's nominal
per capita income grew about 10 percent a year over the period. So the test is
run across a range of income growth, and what is scored is the direction and
the break-even, not the level.

Two different comparisons are reported, because they answer different
questions and the first pre-registration confused them:

    shock   month 24 under the path, against month 24 with no shock. This
            attributes the shock and nothing else. Income growth barely moves
            it, because it falls out of both sides.
    level   month 24 under the path, against the book as it stood in April
            2022. This is what a GNPA ratio moves with, and it is the only one
            that can be scored against an observed change over time.

The hypotheses were written down before this script ran once and are recorded
in ADR 035, along with the correction to H2 and H3's measure:

    H1a  tenure-first produces at least half less stress than EMI-first
    H1b  with income held constant the model over-predicts: stress rises
    H2   with 10 percent income growth, under tenure-first, the level falls
    H3   the break-even income growth lies between 3 and 8 percent a year

Reads nothing and writes nothing. The book is synthetic; the ticket is not
period-sourced and scales rupee totals only, which a test pins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moirai.engine.credit.default import (
    DefaultParameters,
    lifetime_default,
    probability_of_default,
)
from moirai.engine.credit.ecl import EclParameters, assign_stages, expected_credit_loss
from moirai.engine.credit.loan_tape import LoanTape
from moirai.engine.credit.repricing import RepricingParameters, ResetPolicy, reprice
from moirai.engine.credit.synthetic import APRIL_2022, generate

LOANS = 100_000
HORIZON = 24

#: The repo rate from 1 April 2022, by month of the horizon. The MPC's actual
#: decisions: +40bp on 4 May 2022, +50 on 8 June, +50 on 5 August, +50 on
#: 30 September, +35 on 7 December, +25 on 8 February 2023, then held.
DECISIONS = {1: 0.0440, 2: 0.0490, 4: 0.0540, 6: 0.0590, 8: 0.0625, 10: 0.0650}

#: The debt-service criterion, which ADR 033 found is the one a rate shock can
#: reach. The doubling convention is inert here for the same reason.
CRITERION = EclParameters(sicr_dsti_threshold=0.50)

GROWTH = (0.0, 0.05, 0.10)


def repo_path() -> np.ndarray:
    """The actual path: 4.00 percent, stepping to 6.50 by month 10, then held."""
    path = np.full(HORIZON + 1, 0.04)
    for month, rate in DECISIONS.items():
        path[month:] = rate
    return path


@dataclass(frozen=True)
class State:
    """A book's debt service, probabilities and provision at one month."""

    debt_service: np.ndarray
    annual: np.ndarray
    lifetime: np.ndarray
    capped: np.ndarray
    provision: float


def at_month_zero(tape: LoanTape, defaults: DefaultParameters) -> State:
    """The book as it stood, which is what a change over time is measured from."""
    debt_service = np.asarray(tape.borrower_dsti, dtype=float)
    annual = probability_of_default(debt_service, tape.product, defaults)
    lifetime = lifetime_default(annual, tape.remaining_months)
    stage = assign_stages(
        tape,
        annual,
        annual,
        CRITERION,
        dsti_now=debt_service,
        dsti_reference=debt_service,
    )
    provision = expected_credit_loss(tape, stage, annual, lifetime, CRITERION).total()
    return State(debt_service, annual, lifetime, np.zeros(len(tape), dtype=bool), provision)


def at_horizon(
    tape: LoanTape,
    policy_path: np.ndarray,
    reset: ResetPolicy,
    growth: float,
    defaults: DefaultParameters,
    reference: State | None = None,
) -> State:
    """The book at month 24 under a path, staged against `reference` if given."""
    result = reprice(
        tape, policy_path, RepricingParameters(reset_policy=reset, income_growth=growth)
    )
    debt_service = np.asarray(result.borrower_dsti)[:, -1]
    annual = probability_of_default(debt_service, tape.product, defaults)
    lifetime = lifetime_default(annual, np.asarray(result.remaining_months)[:, -1])
    base = reference or State(debt_service, annual, lifetime, np.zeros(len(tape), bool), 0.0)
    stage = assign_stages(
        tape,
        annual,
        base.annual,
        CRITERION,
        dsti_now=debt_service,
        dsti_reference=base.debt_service,
    )
    provision = expected_credit_loss(tape, stage, annual, lifetime, CRITERION).total()
    return State(debt_service, annual, lifetime, np.asarray(result.tenure_capped), provision)


def run(tape: LoanTape, path: np.ndarray, reset: ResetPolicy, growth: float) -> dict[str, float]:
    """Both comparisons: the shock on its own, and the change in the level."""
    defaults = DefaultParameters()
    flat = np.full(HORIZON + 1, path[0])
    start = at_month_zero(tape, defaults)
    unshocked = at_horizon(tape, flat, reset, growth, defaults)
    shocked = at_horizon(tape, path, reset, growth, defaults, reference=unshocked)
    against_start = at_horizon(tape, path, reset, growth, defaults, reference=start)

    weight = np.asarray(tape.outstanding, dtype=float)
    return {
        "shock_change": shocked.provision / unshocked.provision - 1.0,
        "level_change": against_start.provision / start.provision - 1.0,
        "crossed": float(
            np.average(
                (shocked.debt_service > 0.50) & (unshocked.debt_service <= 0.50),
                weights=weight,
            )
        ),
        "capped": float(np.average(shocked.capped, weights=weight)),
        "dsti_start": float(np.average(start.debt_service, weights=weight)),
        "dsti_end": float(np.average(shocked.debt_service, weights=weight)),
    }


def break_even(tape: LoanTape, path: np.ndarray) -> float:
    """The income growth at which the level stops rising and starts falling."""
    low, high = 0.0, 0.40
    if run(tape, path, ResetPolicy.TENURE_FIRST, high)["level_change"] > 0.0:
        return float("nan")
    if run(tape, path, ResetPolicy.TENURE_FIRST, low)["level_change"] <= 0.0:
        return 0.0
    for _ in range(18):  # bisection to about 1bp
        middle = (low + high) / 2.0
        if run(tape, path, ResetPolicy.TENURE_FIRST, middle)["level_change"] > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def main() -> None:
    tape = generate(APRIL_2022, n=LOANS, seed=0).tape
    path = repo_path()
    print(f"{APRIL_2022.source}\n")
    print(
        f"  the actual path: {path[0]:.2%} to {path[-1]:.2%} over "
        f"{max(DECISIONS)} months, held to month {HORIZON}"
    )
    print(f"  {len(tape):,} home loans; debt-service SICR criterion (ADR 033)\n")

    print("=" * 92)
    print("THE RUN  'shock' is against no shock; 'level' is against the April 2022 book")
    print("=" * 92)
    print(
        f"  {'income':<8} {'reset':<13} {'capped':>7} {'crossed':>8}"
        f" {'shock':>9} {'level':>9} {'DSTI Apr-22':>12} {'DSTI Mar-24':>12}"
    )
    results: dict[tuple[float, ResetPolicy], dict[str, float]] = {}
    for growth in GROWTH:
        for reset in ResetPolicy:
            out = run(tape, path, reset, growth)
            results[(growth, reset)] = out
            print(
                f"  {growth:>7.0%} {reset.value:<13} {out['capped']:>6.1%}"
                f" {out['crossed']:>8.1%} {out['shock_change']:>+8.1%}"
                f" {out['level_change']:>+8.1%}"
                f" {out['dsti_start']:>12.3f} {out['dsti_end']:>12.3f}"
            )

    held_tenure = results[(0.0, ResetPolicy.TENURE_FIRST)]
    held_emi = results[(0.0, ResetPolicy.EMI_FIRST)]
    observed = results[(0.10, ResetPolicy.TENURE_FIRST)]

    print()
    print("=" * 92)
    print("THE VERDICTS, against the hypotheses written before this ran")
    print("=" * 92)
    h1a = (
        held_tenure["crossed"] <= 0.5 * held_emi["crossed"]
        and held_tenure["shock_change"] <= 0.5 * held_emi["shock_change"]
    )
    print(
        f"  H1a tenure-first at least halves the stress         {'PASS' if h1a else 'FAIL'}"
        f"   crossed {held_tenure['crossed']:.1%} vs {held_emi['crossed']:.1%},"
        f" shock {held_tenure['shock_change']:+.1%} vs {held_emi['shock_change']:+.1%}"
    )
    h1b = held_tenure["level_change"] > 0.10
    print(
        f"  H1b income held constant over-predicts              {'PASS' if h1b else 'FAIL'}"
        f"   level {held_tenure['level_change']:+.1%} against an observed improvement"
    )
    h2 = observed["level_change"] < 0.0 and observed["dsti_end"] < observed["dsti_start"]
    print(
        f"  H2  at 10 percent income growth the level falls     {'PASS' if h2 else 'FAIL'}"
        f"   level {observed['level_change']:+.1%},"
        f" DSTI {observed['dsti_start']:.3f} to {observed['dsti_end']:.3f}"
    )
    point = break_even(tape, path)
    h3 = 0.03 <= point <= 0.08
    print(
        f"  H3  break-even between 3 and 8 percent a year       {'PASS' if h3 else 'FAIL'}"
        f"   break-even {point:.2%}, observed about 10%"
    )

    print()
    print("  Scored against: housing GNPA 4.3% (Mar-22) to 2.2% (Mar-24), HFCs;")
    print("  retail GNPA 2.1% (Jun-22) to 1.2% (Mar-24); nominal per capita income")
    print("  +12.3% FY23 and +8.7% FY24. The level of NPAs is not scoreable here:")
    print("  the book grew a third and the engine has no credit growth.")


if __name__ == "__main__":
    main()
