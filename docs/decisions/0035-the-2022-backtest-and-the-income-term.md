# 35. The 2022-23 backtest, and the term that was missing

Status: accepted, 12 November 2026

## Context

This is Test 3 of the payoff test agreed before Stage 3 (ADR 033): does the
model reproduce what happened when the RBI raised the repo rate 250bp between
May 2022 and February 2023?

The backtest as originally framed — "the model predicts stress rose; did it?"
— cannot be run, and finding that out was the first result of this stage.
Retail asset quality **improved** through the tightening. Housing GNPA at the
housing finance companies fell from 4.3 percent in March 2022 to 3.3 in March
2023 and 2.2 in March 2024. Retail GNPA fell from 2.1 percent in June 2022 to
1.2 in March 2024.

Three forces the engine does not carry dominate that path:

1. **The book grew by a third**, from Rs 25.1 trillion to Rs 33.3 trillion,
   which cuts any ratio mechanically. The engine has no credit growth.
2. **Income was held constant.** `repricing.py` said so explicitly. India's
   nominal per capita income grew 12.3 percent in FY23 and 8.7 in FY24.
3. **Tenure extension absorbed the shock**, which is the model's own finding.

Scoring the model on the level of NPAs would have tested those three forces
rather than the transmission, and could easily have produced a false pass.
Only the second can be put back, so the test became a test of direction and of
a break-even, not of magnitude.

## Decision

1. **An April 2022 book.** `BookProfile` gains the rate environment it was
   previously reading from module constants: `repo_rate`,
   `internal_reference_rate` and `eblr_spread`. The defaults are the existing
   values, so every earlier result is unchanged, which the comparison run
   confirms. `APRIL_2022` is a home-only book at a 4.00 percent repo rate,
   with 63.7 percent of floating housing loans on EBLR (58.2 against 33.1 at
   December 2021, RBI Bulletin of April 2022, Table 3) and spreads set to
   reproduce the 7.49 percent WALR on outstanding housing loans of February
   2022.

   Its ticket is not period-sourced, because the CRIF FY22 report is not
   reachable. This does not affect the test:
   `test_the_crossing_share_does_not_depend_on_the_ticket` pins that income is
   derived from the EMI and the target debt service, so scaling every loan
   scales the rupee totals and leaves debt service identical.

2. **Income growth in the reset engine.** `RepricingParameters.income_growth`,
   zero by default so the conventional held-constant stress test is unchanged.

3. **Two comparisons, not one**, because they answer different questions:
   *shock* is month 24 under the path against month 24 with no shock, which
   attributes the shock alone; *level* is month 24 under the path against the
   book as it stood in April 2022, which is what a GNPA ratio moves with and
   the only one scoreable against an observed change over time.

## Result

`scripts/check_2022_backtest.py`, 100,000 home loans, the actual repo
decisions, 24 months, debt-service SICR criterion.

| Income growth | Reset | Capped | Crossed 50% | Shock | Level | DSTI Apr-22 | DSTI Mar-24 |
|---|---|---|---|---|---|---|---|
| 0% | tenure first | 27.9% | 2.3% | +40.8% | +40.0% | 0.589 | 0.595 |
| 0% | EMI first | 0.0% | 9.6% | +105.9% | +105.3% | 0.589 | 0.645 |
| 5% | tenure first | 27.9% | 3.6% | +68.5% | +2.7% | 0.589 | 0.539 |
| 5% | EMI first | 0.0% | 15.2% | +170.9% | +29.8% | 0.589 | 0.585 |
| 10% | tenure first | 27.9% | 4.5% | +89.7% | **−9.8%** | 0.589 | 0.491 |
| 10% | EMI first | 0.0% | 19.9% | +232.8% | −5.3% | 0.589 | 0.533 |

| Hypothesis | Verdict | |
|---|---|---|
| H1a tenure first at least halves the stress | **PASS** | crossed 2.3% against 9.6%, shock +40.8% against +105.9% |
| H1b income held constant over-predicts | **PASS** | level +40.0% against an observed improvement |
| H2 at 10 percent income growth the level falls | **PASS** | level −9.8%, debt service 0.589 to 0.491 |
| H3 break-even between 3 and 8 percent a year | **PASS** | break-even 5.62 percent, observed about 10 percent |

### What this says

The model reproduces the direction of the one outcome public data gives, and
it does so for a reason rather than by construction. Held at a constant
income it predicts a 40 percent rise in provisions, which did not happen. Add
the income growth that did happen and it predicts a 10 percent fall, which is
the right direction. The break-even sits at 5.6 percent a year: comfortably
below the observed 10, but not trivially so. Had incomes grown only 4
percent, the model would have predicted deterioration, so this is a test the
period could have failed.

Two mechanisms carry it. Under tenure first, 27.9 percent of the book by value
hit the age-70 or 360-month limit at +250bp and had its EMI forced up anyway,
which is a large number and matches the contemporaneous reports of 20-year
loans becoming 24-year loans. And tenure first still roughly halves the stress
against EMI first, as at +100bp.

### The correction, recorded

H2 and H3 **failed on first run** and pass on a corrected measure. The first
implementation compared the shocked path against the unshocked path at the
same date for every hypothesis. That attributes a shock correctly but cannot
speak to a change in a level over time, and it produced the absurdity that
higher income growth appeared to raise provisions — because income growth
moves the unshocked baseline below the threshold, so more loans fall in the
band between the two paths.

The pre-registration as written asked for the level: H2 says "provisions fall
and mean debt service at month 24 is below **its starting value**". So the
correction aligned the code with the hypothesis, not the hypothesis with the
result, and the H3 band of 3 to 8 percent was set before the first run and
never changed. It is recorded here because a correction made after seeing a
failure has to be, whatever its justification.

An intermediate guess of mine was also wrong: reasoning about the break-even
by hand I expected about 1.3 percent, because I ignored that the lifetime
probability and the stage migration amplify the debt-service move. The
pre-registered band was right and the back-of-envelope was not.

## Consequences

- 7 tests added (4 in `test_synthetic.py`, 3 in `test_repricing.py`), one
  script. No existing output changes: the rate environment became a field with
  its current values as defaults, and `income_growth` defaults to zero.
- The "nothing is validated against outcomes" weakness is now closed with a
  caveat rather than open: the direction is validated, the magnitude is not,
  and the level of NPAs is not scoreable without credit growth.
- What would make this a strong test, in order: segment-level arrears over
  2022-24, which is not published; a debt-service-to-arrears transition, so
  the shock reaches provisions the way lenders see it; credit growth, so the
  ratio rather than the stock can be compared.
- The three agreed payoff tests are now all run. The architecture passes:
  conditional on the SICR criterion (ADR 033), and on direction rather than
  magnitude here.
- Next is Stage 6, packaging, by the decision of 12 November. F2 to F5 stay
  parked.
