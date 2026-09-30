# 19. The Fed-to-India exchange coefficient against Indian data

Date: 2026-09-30
Status: Accepted

## Context

A sensitivity sweep of the headline scenario found one input dominating
the RBI's response to the Fed: the exchange cell of the spillover matrix,
`exchange[RBI, Fed]`, the points of Indian inflation a point of Fed
tightening produces through a weaker rupee. Halving it cut the RBI's
caused move from 13bp to 3bp; doubling it took the move to 28bp. The RBI's
own preference weights moved it by a few basis points at most.

The cell is 0.42, built in `SpilloverMatrix.from_literature` as the IMF's
estimate that a 1pp US tightening raises emerging market local currency
bond yields by 36bp, times an exchange ratio of 1.17 carried from the
tiered version. It is a bond yield response for emerging markets in
aggregate standing in for an inflation response in India. The docstring
says so. No part of it had been checked against Indian data.

## What was measured

The coefficient factors into two measurable pieces: the rupee's
depreciation per point of Fed tightening, times exchange rate pass-through
to Indian CPI.

**The rupee, on the day** (`scripts/check_rupee_fed_surprise.py`). Bauer
and Swanson (2023) surprises, orthogonalised to public information and
scaled to a 1pp move in expected US rates about a year ahead, from the San
Francisco Fed's update, against FRED's noon New York rupee fixing. The
window runs from the last fixing before each announcement to the first
after it, from the `Time` column, since most announcements come at 2pm
and the fixing at noon.

| Sample | Depreciation per 1pp | t |
|---|---|---|
| All, 1995-2023 | +1.28% | 1.93 |
| Scheduled only | +1.70% | 2.07 |
| Raw surprise | +1.28% | 2.27 |
| 1995-2007 | +0.59% | 1.15 |
| 2008-2023 | +2.71% | 1.66 |
| 2013-2023 | +4.35% | 2.96 |

Placebo windows one and two trading days earlier are insignificant in the
full sample (t = 1.25, 1.18) and in 2013-2023 (t = 0.54, -1.47). A method
check on synthetic data with a planted response recovered it and gave
null placebos.

The effect is undetectable while the rupee was tightly managed and large
after the taper tantrum, consistent with India's rising exposure to US
policy through capital flows.

**The rupee, over a year** (`scripts/check_india_fed_spillover.py`).
Jorda local projections on the monthly surprise, six lags of the outcome
and the surprise as controls, Newey-West errors. For 2013-2023 the
depreciation is -3.1% to -2.5% over the first three months, +6.0% at six,
+7.7% at nine and +10.8% at twelve (t = 1.6). None is significant: a
month's surprises are small, so they explain little of a month's movement.
The point estimates build rather than fade.

**Pass-through.** The RBI's Monetary Policy Report of October 2022 states
that a 5 percent depreciation from baseline could raise inflation by
around 20bp, so 0.04. A secondary source puts the RBI's estimate at about
0.07. The RBI's own figure is used.

**The demand channel.** Industrial production responses (OECD index
1995-2023, RBI IIP 2011-2023, pandemic excluded) are mostly negative, the
expected sign, and insignificant throughout. The RBI series is too short
and too noisy to use. The demand cell stays at 0.36, unmeasured.

## Result

| Depreciation used | Times RBI pass-through | Coefficient |
|---|---|---|
| One day, 2013-2023, precise | 4.35% x 0.04 | 0.17 |
| Twelve months, 2013-2023, imprecise | 10.8% x 0.04 | 0.43 |
| Default, from the literature | | 0.42 |

The precise evidence puts a floor under the coefficient at 0.17. The
imprecise evidence is consistent with the default and cannot confirm it.
The data does not support a single value inside that range.

## Decision

- The default stays at 0.42. It lies inside the range the evidence
  allows, at its upper end, and replacing it with a number the data cannot
  pin down would trade one assumption for another.
- `network.py` carries the evidence as constants with their sources:
  `RBI_EXCHANGE_RATE_PASS_THROUGH`,
  `RUPEE_DEPRECIATION_PER_FED_POINT_ONE_DAY` and
  `FED_TO_INDIA_EXCHANGE_FLOOR = 0.174`.
- `SpilloverMatrix.with_exchange` replaces one exchange cell and nothing
  else, for bounding.
- `run_pipeline.py` reports the headline at both ends of the exchange
  range and both ends of the deposit range (ADR 018).
- Both scripts are kept as the evidence, and eight tests pin the
  constants, the order of floor and default, and the method.

## The headline across the evidence

On the real data:

| | RBI caused | Aggregate | Savers | Spread |
|---|---|---|---|---|
| As reported (exchange 0.42, deposits 31%) | +12.9bp | -0.085% | -0.043% | -0.61pp |
| Deposits as observed (97%) | +12.9bp | -0.024% | +0.023% | -0.60pp |
| Exchange at the floor (0.17) | +4.0bp | -0.026% | -0.014% | -0.21pp |
| Both | +4.0bp | -0.007% | +0.006% | -0.20pp |

The spread lies between 0.20 and 0.61 percentage points. Deposit
pass-through barely moves it; the exchange coefficient sets where in that
range it falls, because it sets how much of the Fed's move the RBI
imports. At every point, floating-rate borrowers bear the tightening and
savers do not.

## Consequences

- The finding to quote is a range: floating-rate borrowers cut consumption
  by 0.2 to 0.6 points more than savers. Where it falls depends on how
  long the rupee's response to a Fed surprise lasts, which Indian data can
  bound from below but not yet measure.
- The aggregate is between roughly zero and -0.09 percent. It is not a
  finding.
- The coefficient's provenance moves from an emerging market bond yield
  proxy to a range anchored on India's own rupee response and the RBI's
  own pass-through estimate.
- Narrowing the range needs a sharper measure of persistence: daily local
  projections of the rupee over weeks, or Indian high-frequency surprises
  (Lakdawala and Sengupta) to separate the RBI's own response.
- The demand channel remains unmeasured and is the next input to test. It
  pulls the RBI the other way, so if it is also overstated the floor
  would rise.

## Data

`data/raw/manual/frbsf_monetary_policy_surprises.xlsx`, downloaded from
https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/
(update covering 1988-2023). FRED series DEXINUS, EXINUS and
INDPROINDMISMEI through the project's adapter, archived with content
hashes. The workbook is not tracked by git; it must be present in each
checkout's `data/raw/manual` for the evidence scripts to run.
