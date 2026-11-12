# 33. The credit layer, and what the pre-registered payoff test found

Status: accepted, 12 November 2026

## Context

Stage 2 showed what a policy move does to each borrower's payment. A lender
does not act on a payment, it acts on a provision, and from 1 April 2027 the
RBI's final expected-credit-loss directions fix how that provision is formed.
Stage 3 closes the chain: debt service to probability of default, probability
to stage, stage to expected credit loss under weighted scenarios.

Before building it, the user and I agreed a go/no-go test on the architecture
itself. The question was whether the chain reaches money: if the macro layers
and the loan layer make no material difference to what a lender books, then
the borrower half of Moirai is description rather than decision, and effort
belongs back in the macro half. The pass marks were written down first, in
`prediction_stage3.txt`, before the script was run once.

## Decision

1. **`default.py`** turns debt service into a 12-month probability. The shape
   is Nier, Popa, Shamloo and Voinea (IMF WP/19/182), on Romanian loan-level
   credit-register data: a threshold, not a slope.

   | Loans | Flat below | Rise in PD per 10pp of DSTI above it |
   |---|---|---|
   | Mortgage | 50% | +16% in [50, 90), +14% in [90, 120), +8% above 120 |
   | Consumer | 30% | +8% in [30, 50), +6% in [50, 70), +3% above 70 |

   The effect is multiplicative, so the bands scale a base level. The base
   level is calibrated, not published: `calibrate_flat_pd` solves for the
   level at which a book reproduces a lender's disclosed Stage 1 coverage.
   The default, 0.434%, is what the synthetic prime book's home loans give
   against Bajaj Housing Finance's 0.33% at the RBI's 65% fallback LGD. The
   three published lenders imply 0.26% to 0.45%. Direction for India comes
   from Bandyopadhyay and Saha (2009), 13,487 Indian housing loans.

2. **`ecl.py`** stages and provisions. Sourced from the directions: Stage 2
   presumed above 30 days past due; Stage 3 credit-impaired with the 90-day
   backstop, at borrower level; 12-month loss in Stage 1 and lifetime beyond;
   fallback LGD 65% secured and 70% unsecured; PD floor 0.03%; provisioning
   floors, which for individual housing are 0.25% in Stage 1 and 1.5% in
   Stage 2, and in Stage 3 rise 10/20/30/40/100% with years in arrears; at
   least three probability-weighted scenarios. A provision is the greater of
   the modelled loss and the floor, and the result records which decided it.

3. Two things are marked ASSUMED because no regulator sets them. The
   significant-increase-in-credit-risk test has no prescribed number, so two
   criteria are implemented and both are reported: a doubling of probability
   (the industry convention) and crossing the 50% debt-service level (the
   IMF threshold, as a qualitative criterion the directions permit). The
   scenario weights are an input to the run.

## What the test found

`scripts/check_credit_payoff.py`, 100,000 loans per book, 24 months,
provisions compared at the same month under a flat and a shocked path.

**Test 1: does a +100bp path move provisions by more than 10%?**

| Book | Reset | PD doubling | DSTI crossing |
|---|---|---|---|
| psb | tenure first | +0.1% | +2.4% |
| psb | EMI first | +1.4% | **+18.2%** |
| private | tenure first | +0.2% | +3.7% |
| private | EMI first | +1.7% | **+23.1%** |
| affordable | tenure first | +0.0% | +3.1% |
| affordable | EMI first | +0.5% | **+41.5%** |

**Test 2: does the RBI's own choice matter?** Provisions under the RBI
defending (+126bp, twin tightening) against the RBI accommodating (+13bp, Fed
tightening alone), both from the network game (ADR 029).

| Book | Reset | PD doubling | DSTI crossing |
|---|---|---|---|
| psb | EMI first | 1.02x | **1.17x** |
| private | EMI first | 1.02x | **1.20x** |
| affordable | EMI first | 1.01x | **1.46x** |
| all | tenure first | 1.00x | 1.03 to 1.06x |

**Test 3**, the 2022 backtest, belongs to Stage 4 and was not run.

### The finding that matters

The architecture reaches money, but only through one door. Under the
doubling convention the chain is inert: doubling a mortgage PD requires debt
service to rise by 10pp x ln(2)/ln(1.16) = **47 percentage points**, which no
rate shock produces. A +100bp move raises the EMI about 4% under EMI-first,
carrying debt service from roughly 0.59 to 0.62 and the PD by 4.5%. So a
lender using the common convention would report almost nothing from a rate
shock, however severe the borrower stress underneath.

Under the debt-service criterion the same shock moves 2.6% to 3.6% of
exposure into Stage 2 and provisions by 18% to 42%. The loans that migrate
carry a lifetime rather than a 12-month loss and a six-fold higher floor.

Two conclusions follow. The choice of SICR criterion matters more to reported
provisions than the size of the shock, and it is the one thing no regulator
specifies. And the reset policy is again decisive: under tenure-first every
configuration fails both tests, because stretching the term is what keeps
debt service below the level at which the provision changes. The 2022-23
practice the RBI objected to also happens to be the practice that keeps the
stress out of the accounts.

### A second finding: the floor often decides

On the mixed books, **36.7% to 37.2% of exposure** is provisioned at the
floor rather than by the model. It is the non-housing share: at a 0.434% PD,
a vehicle loan gives 0.28% coverage against a 0.40% floor and a personal loan
0.30% against a 1.0% floor. Home loans clear their 0.25% floor, but only
just, at 0.28%. For a third of a retail book a loss model does not decide the
Stage 1 provision at all.

### Predictions that were wrong

Every pass/fail verdict matched the pre-registration. Two magnitudes did not,
both too high:

- I predicted +40% to +80% for the prime books under EMI-first with the
  debt-service criterion. The outcome was +18% and +23%. I had assumed each
  migrating loan's provision rises about twentyfold; it rises less, because
  loans with a shorter remaining term carry a smaller lifetime loss and some
  were already above their floor.
- I predicted the defend case would be "several times" the accommodate case.
  It is 1.17 to 1.46 times.

The floor-bound share was predicted correctly at 0.37, as was the inertness
of the doubling criterion and the failure of every tenure-first case.

## Consequences

- 62 tests added (`test_default.py` 30, `test_ecl.py` 32), one script, and no
  change to any existing output.
- **The architecture passes**, conditional on the SICR criterion. The default
  stays the doubling convention, because that is what a lender's auditor
  expects, and the debt-service criterion is offered beside it. Any run that
  reports a provision impact must say which it used.
- Still assumed: the scenario weights; the SICR criterion itself; the
  constant-hazard lifetime PD, which has no seasoning because the tape has no
  months on book; the Stage 3 vintage, approximated from days past due; the
  base PD level for vehicle and personal loans, inherited from housing
  because no Indian lender publishes a comparable figure.
- Stage 3 exposure is nil on a synthetic book, which has no arrears. The
  arrears channel is the one a rate shock most plausibly works through, and
  it is exactly what a DSTI-to-arrears transition model would add. That, not
  F2 to F5, is the next thing that would materially improve the answer.
- Numbering: the parked date-format change (Stage 1.B) carries "ADR 033" in
  `manual_csv.py` and `test_date_formats.py`. Those references move to the
  next free number when that step is resumed.
- Next: Stage 4, the pre-registered 2022 backtest, which is Test 3.
