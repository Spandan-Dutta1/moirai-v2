# 25. Borrowing capped at the lenders' limit

Date: 2026-10-01
Status: Accepted

## Context

ADR 024 found 1,442 of 200,000 households consuming nothing in every
period, and showed they inflate the original group measure about threefold.
Their debt service exceeds their income, and disposable income is floored
at zero.

The first remedy considered was a subsistence floor on consumption, set as
a share of income. It was rejected on measurement: a floor at 30 percent of
income cut floating-rate borrowers' response by two thirds, and at 50
percent it removed the rate effect entirely, because consumption is the MPC
times disposable income and a floor of that size binds for most
households. The result would have been a function of the floor chosen.

The cause is upstream, in the population. Debt-to-income is drawn from a
lognormal with median 1.8 and a standard deviation of one in logs:

| Debt-to-income among borrowers | Median | 90th percentile | 99th percentile | Maximum |
|---|---|---|---|---|
| Uncapped | 1.8 | 6.6 | 18.6 | 132 |

6.2 percent of borrowers have repayments above their income at the
baseline lending rate. No lender extends debt of a hundred times income.
Indian banks limit a borrower's fixed obligations to around half of
income, the fixed-obligation-to-income ratio.

## Decision

- `PopulationParameters.max_debt_to_income`, `None` by default so every
  validated result reproduces, caps a borrower's debt as a multiple of
  income.
- `LENDER_FOIR_LIMIT = 0.50` and `LENDER_DEBT_TO_INCOME_CAP = 4.22`: a
  fifteen-year annuity at the model's baseline lending rate of 8.25 percent
  costs 0.1186 of principal a year, so debt of 4.22 times income uses half
  of it.
- The random draws are unchanged. A capped population is the same
  households, with tail borrowers carrying no more than lenders extend;
  incomes, wealth and floating-rate status are identical.
- `run_pipeline.py` reports the headline on the capped population beside
  the original, with its count of households consuming nothing and its
  calibration loss.
- Four tests: no cap leaves the population identical; the cap changes only
  tail debt; 4.22 is the 50 percent limit at 8.25 percent over fifteen
  years; the cap lowers the calibration loss.

## Result

On the real data:

| Population | Spread, original measure | Spread, by group totals | Consuming nothing | Calibration loss |
|---|---|---|---|---|
| Uncapped (validated) | -0.61pp | -0.18pp | 1,442 | 2.720 |
| **Capped at 4.22x income** | **-0.20pp** | **-0.16pp** | **0** | **1.319** |

- **The zero-consumption households disappear.**
- **The two measures converge**, -0.20 against -0.16. With nobody at the
  floor there is nothing for the average of household changes to be
  distorted by, so the measurement problem of ADR 024 was a symptom of
  this.
- **The calibration loss halves.** The capped population matches the
  declared calibration targets better than the uncapped one, which the cap
  was not chosen to do. That is independent evidence the tail was an
  artefact of the lognormal draw.
- In a sandbox test with constant paths, a 13bp cut and a 13bp rise now
  have mirror-image effects on floating-rate borrowers (+0.12 and -0.17
  percent), where uncapped they were +12.9 and -0.60.

The comparison against the main branch showed `run_pipeline` as the only
script whose output changed, every existing line identical.

## Consequences

- The defensible headline is about -0.2 points: floating-rate borrowers
  cut consumption by about 0.2 points more than net savers. The original
  measure on the original population, -0.61, is kept and reproducible.
- The lender limit is a single number applied to every borrower. Real
  limits vary by lender, product and income, and informal borrowing is not
  subject to them (roadmap item F18).
