"""Three synthetic books through a +25bp step and a +100bp stress (Stage 2.5).

The books are built to published aggregates (moirai.engine.credit.synthetic);
the shapes around those aggregates are assumed and named there, so these are
results on a synthetic book, not on a lender's.

Each book starts at the repo rate of 5.50 percent. The policy path steps up in
month 1 and stays there for 24 months, which covers every reset: EBLR loans
reprice within three months, internal-benchmark loans within twelve.

Reported for home loans (the floating part of each book), by outstanding value:

    capped        an extension hit age 70 or 30 years, so the EMI rose
    peak EMI      the average largest rise in the monthly payment
    pushed        DSTI crossed 50 percent during the path, from at or below it
    above         DSTI was already above 50 percent before any shock

Reads nothing and writes nothing.
"""

from __future__ import annotations

import numpy as np

from moirai.engine.credit.loan_tape import Product
from moirai.engine.credit.repricing import RepricingParameters, ResetPolicy, reprice
from moirai.engine.credit.synthetic import PROFILES, REPO_RATE, generate

HORIZON = 24
LOANS = 100_000


def share(flag: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(flag[mask], weights=weight[mask]))


def main() -> None:
    for profile in PROFILES:
        book = generate(profile, n=LOANS, seed=0)
        tape = book.tape
        home = tape.product == Product.HOME
        weight = tape.outstanding
        print(f"\n{profile.name}: {len(tape):,} loans, {home.sum():,} home; {profile.source}")
        print(
            f"  home share of value {share(home, np.ones_like(home), weight):.1%}; "
            f"home DSTI above 50% at start {share(tape.borrower_dsti > 0.5, home, weight):.1%}"
        )
        for bp in (25, 100):
            path = np.r_[REPO_RATE, np.full(HORIZON, REPO_RATE + bp / 10_000)]
            for policy in ResetPolicy:
                result = reprice(tape, path, RepricingParameters(reset_policy=policy))
                emi = np.asarray(result.emi)
                peak = emi.max(axis=1) / emi[:, 0] - 1.0
                print(
                    f"  +{bp:>3}bp {policy.value:<12}"
                    f" capped {share(result.tenure_capped, home, weight):6.1%}"
                    f"  peak EMI {share(peak, home, weight):+6.1%}"
                    f"  pushed {share(result.dsti_pushed_across, home, weight):6.1%}"
                    f"  above {share(result.dsti_above_at_start, home, weight):6.1%}"
                )


if __name__ == "__main__":
    main()
