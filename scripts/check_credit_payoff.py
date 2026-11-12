"""Does the credit layer earn its place? Three pre-registered tests (Stage 3).

The pass marks were written down before this script was run once, and are
recorded in ADR 033. They are:

    1  a +100bp path changes total provisions by more than 10 percent
    2  the RBI accommodating and the RBI defending give materially different
       provisions, so the macro chain matters to the credit answer
    3  the 2022 backtest ranks stressed segments correctly (Stage 4, not here)

Every provision is compared at the same month under two paths, flat and
shocked, so the difference is the shock and nothing else. The staging
reference is the unshocked run, which makes the significant-increase test ask
"whose risk did this shock raise", not "whose risk has risen since the loan
was written" - the accounting question a tape without months on book cannot
answer.

Both significant-increase criteria are reported, because they are not the same
test: doubling a probability needs debt service to rise by 47 percentage
points under the IMF's bands, which no rate shock does, while crossing the 50
percent level is exactly what a rate shock can cause.

The two RBI paths are the network game's own output, recorded in ADR 029: a
Fed tightening alone moves the RBI 13bp, and a twin tightening moves it 126bp.

Reads nothing and writes nothing. Results are on synthetic books.
"""

from __future__ import annotations

import numpy as np

from moirai.engine.credit.default import (
    DefaultParameters,
    lifetime_default,
    probability_of_default,
)
from moirai.engine.credit.ecl import (
    EclParameters,
    Stage,
    assign_stages,
    expected_credit_loss,
    weighted_provision,
)
from moirai.engine.credit.loan_tape import LoanTape
from moirai.engine.credit.repricing import RepricingParameters, ResetPolicy, reprice
from moirai.engine.credit.synthetic import PROFILES, REPO_RATE, generate

HORIZON = 24
LOANS = 100_000

#: The RBI's caused move under each scenario, from the network game (ADR 029).
RBI_PATHS = {"accommodate": 0.0013, "defend": 0.0126}
STRESS = 0.0100

CRITERIA = {
    "pd doubling": EclParameters(),
    "dsti crossing": EclParameters(sicr_dsti_threshold=0.50),
}


def path(size: float) -> np.ndarray:
    return np.r_[REPO_RATE, np.full(HORIZON, REPO_RATE + size)]


FLAT = path(0.0)


def run(
    tape: LoanTape,
    policy_path: np.ndarray,
    reset: ResetPolicy,
    defaults: DefaultParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Debt service, both probability horizons and the term, at the last month."""
    result = reprice(tape, policy_path, RepricingParameters(reset_policy=reset))
    debt_service = np.asarray(result.borrower_dsti)[:, -1]
    remaining = np.asarray(result.remaining_months)[:, -1]
    annual = probability_of_default(debt_service, tape.product, defaults)
    return debt_service, annual, lifetime_default(annual, remaining), remaining


def provision(
    tape: LoanTape,
    shock: float,
    reset: ResetPolicy,
    criterion: EclParameters,
    defaults: DefaultParameters,
) -> dict[str, float]:
    """Provisions at month 24 under the shock, against the same month unshocked."""
    base_dsti, base_annual, base_lifetime, _ = run(tape, FLAT, reset, defaults)
    dsti, annual, lifetime, _ = run(tape, path(shock), reset, defaults)

    reference_stage = assign_stages(
        tape,
        base_annual,
        base_annual,
        criterion,
        dsti_now=base_dsti,
        dsti_reference=base_dsti,
    )
    stressed_stage = assign_stages(
        tape,
        annual,
        base_annual,
        criterion,
        dsti_now=dsti,
        dsti_reference=base_dsti,
    )

    before = expected_credit_loss(tape, reference_stage, base_annual, base_lifetime, criterion)
    after = expected_credit_loss(tape, stressed_stage, annual, lifetime, criterion)
    summary = after.summary()
    return {
        "before": before.total(),
        "after": after.total(),
        "change": after.total() / before.total() - 1.0,
        "stage_two": summary["share_stage_2"],
        "floor_bound": summary["share_floor_bound"],
        "coverage": summary["coverage"],
    }


def main() -> None:
    defaults = DefaultParameters()
    books = {p.name: generate(p, n=LOANS, seed=0).tape for p in PROFILES}
    print(
        f"flat PD {defaults.flat_pd_12m:.4%} in the flat region; "
        f"{LOANS:,} loans per book; {HORIZON} months; synthetic books\n"
    )

    print("=" * 94)
    print("TEST 1  does a +100bp path move provisions by more than 10 percent?")
    print("=" * 94)
    header = f"  {'book':<11} {'reset':<13} {'criterion':<14}"
    print(f"{header} {'provision':>13} {'change':>8} {'stage 2':>8} {'floored':>8} {'cover':>7}")
    for name, tape in books.items():
        for reset in ResetPolicy:
            for label, criterion in CRITERIA.items():
                out = provision(tape, STRESS, reset, criterion, defaults)
                verdict = "PASS" if abs(out["change"]) > 0.10 else "fail"
                print(
                    f"  {name:<11} {reset.value:<13} {label:<14}"
                    f" {out['after']:>13,.0f} {out['change']:>+7.1%} {verdict}"
                    f" {out['stage_two']:>7.1%} {out['floor_bound']:>8.1%}"
                    f" {out['coverage']:>7.2%}"
                )

    print()
    print("=" * 94)
    print("TEST 2  does the RBI's choice change provisions? (+13bp accommodate, +126bp defend)")
    print("=" * 94)
    print(
        f"  {'book':<11} {'reset':<13} {'criterion':<14}"
        f" {'accommodate':>13} {'defend':>13} {'ratio':>7}"
    )
    for name, tape in books.items():
        for reset in ResetPolicy:
            for label, criterion in CRITERIA.items():
                totals = {
                    scenario: provision(tape, size, reset, criterion, defaults)["after"]
                    for scenario, size in RBI_PATHS.items()
                }
                ratio = totals["defend"] / totals["accommodate"]
                verdict = "PASS" if ratio > 1.10 else "fail"
                print(
                    f"  {name:<11} {reset.value:<13} {label:<14}"
                    f" {totals['accommodate']:>13,.0f} {totals['defend']:>13,.0f}"
                    f" {ratio:>6.2f}x {verdict}"
                )

    print()
    print("=" * 94)
    print("THE WEIGHTED PROVISION the RBI's directions ask for (three scenarios)")
    print("=" * 94)
    tape = books["psb"]
    criterion = CRITERIA["dsti crossing"]
    weights = {"accommodate": 0.5, "defend": 0.3, "stress": 0.2}  # ASSUMED
    results = {}
    for scenario, size in {**RBI_PATHS, "stress": STRESS}.items():
        base_dsti, base_annual, _, _ = run(tape, FLAT, ResetPolicy.EMI_FIRST, defaults)
        dsti, annual, lifetime, _ = run(tape, path(size), ResetPolicy.EMI_FIRST, defaults)
        stage = assign_stages(
            tape,
            annual,
            base_annual,
            criterion,
            dsti_now=dsti,
            dsti_reference=base_dsti,
        )
        results[scenario] = expected_credit_loss(tape, stage, annual, lifetime, criterion)
        print(
            f"  {scenario:<13} weight {weights[scenario]:.2f}"
            f"  provision {results[scenario].total():>14,.0f}"
        )
    print(
        f"  {'weighted':<13}              provision {weighted_provision(results, weights):>14,.0f}"
    )
    print("  weights are ASSUMED; the directions require three scenarios but set no weights")

    print()
    print("  TEST 3 (the 2022 backtest) belongs to Stage 4 and is not run here.")
    print(f"  Stage 3 exposure is nil on these books: {Stage.THREE.name.lower()} needs arrears,")
    print("  which a synthetic book has none of. A real tape carries them.")


if __name__ == "__main__":
    main()
