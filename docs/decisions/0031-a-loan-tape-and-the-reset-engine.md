# 31. A loan tape, the EMI reset engine, and a synthetic book built to published figures

Status: accepted, 8 November 2026

## Context

The product question asks which of a lender's borrowers break under the next
global or RBI shock, through which channel, and what that does to provisions.
The household layer cannot answer it. It holds debt as one number per
household, amortised over one maturity, and it has no benchmark, no reset date
and no lender rule. A policy move reaches an Indian borrower only when the loan
reprices, by as much as its benchmark passes on, and in the form chosen at the
reset. Stage 2 adds that layer as `moirai.engine.credit`.

## Decision

1. **The loan tape** (`loan_tape.py`): one row per loan, held as parallel
   arrays and validated on entry. A contradiction is refused, and the refusal
   names the loans. Negative amortisation is the exception: it is accepted and
   flagged, because it is the 2022-23 outcome itself. `borrower_income` is
   annual net income.
2. **The reset engine** (`repricing.py`): runs a monthly policy path through
   every loan. It reprices on each loan's own schedule, applies the
   benchmark's pass-through, extends tenure first or raises the EMI first,
   and caps an extension at the lender's age and tenure limits.
3. **The adapter** (`adapter.py`): turns a tape into a Population, one
   household per borrower. The four fields a tape cannot observe
   (employment, wealth, expected inflation, risk aversion) are imputed and
   named on the result.
4. **The synthetic book** (`synthetic.py`): until a lender shares a tape, the
   engine runs on a book built to published aggregates. Each aggregate is a
   test with a tolerance. Every shape the sources do not publish is named in
   `ASSUMED` and carried on the book.

## The defaults and where they come from

A literature and data search replaced the values first marked ASSUMED.

| Parameter | Value | Source |
|---|---|---|
| EBLR pass-through | 1.0 | Contractual: repo plus a fixed spread, reset at least quarterly (RBI, September 2019) |
| Internal (MCLR) pass-through | 0.62 | 1-year median MCLR moved 155bp for 250bp of repo, in the 2019-21 easing (RBI Bulletin, July 2021) and in the 2022-24 tightening (NIBM, 2024) |
| Reset policy | tenure first | 2022-23 practice (RBI statement, 10 August 2023; Home First earnings call; Business Standard). Since the August 2023 framework the borrower chooses, so forward runs report both |
| Maximum age at maturity | 70 | SBI: 30 years or age 70, whichever is earlier. HDFC is about 65. Set per lender |
| Maximum tenure | 360 months | SBI and HDFC |
| DSTI threshold | 0.5, net income | Mortgage default risk is flat below 50% and rises above it (Nier, Popa, Shamloo and Voinea, IMF WP/19/182) |

The synthetic book's targets come from these sources:

- **Repo:** 5.50% after the 7 October 2026 hike (RBI MPC).
- **MCLR:** median 8.61% (RBI, September 2026).
- **Product mix and tickets:** from CRIF High Mark (December 2025 and Q3 FY26).
- **EBLR share of floating loans:** 49.8% at public-sector banks, 88.6% at private banks (RBI, September 2025).
- **Age:** 38 on average at origination (HDFC).
- **Average FOIR:** 59% for a prime book (Bajaj Housing Finance), about 40% for an affordable one (Home First).

## What the search changed beyond the numbers

- **Two DSTI flags, not one.** Indian prime books start above 50%: Bajaj
  Housing Finance's average FOIR is 59%. A single flag would mark most of a
  prime book before any shock. The engine reports `dsti_above_at_start` and
  `dsti_pushed_across` separately.
- **The age limit is a lender rule, not retirement.** The household layer
  retires people at 60. The lender lends to 70. They are separate
  parameters.
- **Two imputed fields reach no decision.** Nothing in `behaviour.py` reads
  `risk_aversion`. `expected_inflation` is updated and reported but drives no
  decision. They are not calibrated, since no output depends on them.

## Result (`scripts/check_loan_book.py`)

Home loans, by outstanding value, over 24 months from a repo rate of 5.50%:

| Book | Above 50% DSTI at start | +100bp, tenure first | +100bp, EMI first |
|---|---|---|---|
| psb | 78.6% | 11.9% capped, 0.3% pushed across | EMI +4.0%, 4.2% pushed across |
| private | 79.7% | 14.0% capped, 0.5% pushed across | EMI +4.8%, 5.5% pushed across |
| affordable | 18.1% | 8.3% capped, 0.2% pushed across | EMI +3.0%, 3.6% pushed across |

The reset choice moves the share pushed across 50% DSTI by ten to fifteen
times on the same shock. Under tenure first the shock shows up as longer
loans, not as payment stress. The private book is hit harder than the
public-sector book because more of it is on EBLR.

The prediction written before the run expected 15-35% of loans capped and
peak EMIs up 3-6% under tenure first. Both were too high, because under
tenure first only capped loans see their EMI rise. They are recorded as wrong.

These are results on a synthetic book, about the mechanism, not estimates for
any lender.

## Consequences

- Main gains the credit package and one script. No existing output changes:
  the comparison run differed only in pytest (1384 to 1448) and in the new
  script.
- Still assumed, and replaced by a design partner's tape:
  - the shape of the DSTI distribution;
  - the joint distribution of age, tenure and ticket;
  - seasoning and spreads;
  - one loan per borrower;
  - fixed rates for vehicle and personal loans.
- Next refinements, each a step of its own:
  - **MCLR pass-through that grows with time.** The evidence is about 0.36 at
    15 months, 0.62 at two years and 0.70 at 30 months. It would be fitted as
    a partial-adjustment model on the RBI's monthly MCLR and repo series.
  - **A months-on-book field,** so the 30-year limit counts from origination.
  - **An employment-type field.** Self-employed borrowers default more in
    Indian loan-level data (Bandyopadhyay and Saha 2009).
- Stage 3 turns DSTI into a probability of default. The IMF slopes give the
  shape, Indian loan-level studies give the direction, and CRIF's arrears
  rates give the level.
