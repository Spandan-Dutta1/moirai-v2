# 29. The RBI starts at its observed inflation, reported beside the declared one

Status: accepted, 7 November 2026

## Context

The RBI in `central_banks.py` carries a declared starting inflation of 4.4%,
carried over from `run_pipeline_india.py`. The warehouse's latest CPI Combined
reading (`in_cpi_inflation`, May 2026) is 3.93%. A declared number written down
once goes stale; this one had.

The study notes recorded a fix under the name ADR 028 in a trial copy, with the
caused move unchanged at 12.92bp. It was never committed (ADR 028 became the
tooling decision) and is redone here, with a comparison run.

Where the starting inflation enters the chain:

- the network's inflation gap, `current_inflation - inflation_target`
  (`network.py`), in both the solve with the scenario's conditions and the
  reference solve without them;
- the seam, which holds Indian inflation at the bank's starting value
  (`policy_shock.py`, `baseline=bank.current_inflation`);
- the dynamic chain's inflation path (`dynamic_chain.py`);
- behaviour, which erodes wealth and debt by the inflation level each month
  (`behaviour.py`, `real_erosion = 1 / (1 + inflation / 12)`).

## Decision

1. The declared RBI is unchanged. A new module,
   `engine/financial/starting_state.py`, returns the bank at its observed
   starting inflation and a record of both values. The warehouse is passed in,
   so importing the engine never touches a database.
2. It refuses what cannot be a starting inflation: an index level, a series
   still holding several vintages, no values, a value above 50%. With no
   series in the warehouse (a fresh clone) it keeps the declared value and the
   record says so.
3. `run_pipeline.py` and `run_scenarios.py` start the RBI at the observed value
   and print one line: `Reserve Bank of India: starting inflation declared
   4.40%, observed 3.93% (May 2026, in_cpi_inflation); using observed`.
4. Scenario conditions stay declared as levels. "Indian inflation at 6.8%" is
   a statement about the world; its size as a shock is measured from where
   inflation starts, and that is now the observed start.

## Evidence

Comparison run, main (`53446a8`) against this branch: three differences,
`_pytest` (1,371 -> 1,383), `run_pipeline` and `run_scenarios`. Nothing else.

The headline (`imported_tightening`), whose conditions do not touch the RBI:

| | Declared 4.40% | Observed 3.93% |
|---|---|---|
| RBI caused move | +12.9bp | +12.9bp |
| RBI unprompted move | +6bp | -13bp |
| RBI rate, reference / with conditions | 5.31% / 5.44% | 5.12% / 5.25% |
| Inflation held in the seam | 4.400% | 3.930% |
| Spread, by group totals (ADR 024) | -0.18pp | -0.18pp |
| Spread, household average | -0.61pp | -0.62pp |
| Floating-rate borrowers, household average | -0.650% | -0.665% |
| Households consuming nothing | 1,442 | 1,483 |
| Evidence range, household average | -0.20 to -0.61 | -0.21 to -0.62 |
| Dynamic game spread, by totals | -0.049pp | -0.050pp |
| Per 100bp of Fed tightening, static / dynamic | -0.18 / -0.55 | -0.18 / -0.56 |

The caused move is unchanged because the network is linear while no bank
leaves its band, and the starting inflation enters both solves identically.
The household-average measure moves slightly because lower inflation erodes
debt less, so 41 more of the most indebted households reach zero consumption;
that measure is the one near-zero consumers distort, which is why ADR 024 made
group totals the headline. By totals, nothing moved.

Scenarios that declare the RBI's own inflation as a condition:

| Scenario | RBI caused, declared -> observed | RBI rate with conditions | Households |
|---|---|---|---|
| `twin_tightening` (RBI at 6.8%) | +107 -> +126bp | 6.380%, unchanged | shock 2.6 -> 3.1 SD; spread -3.68 -> -4.19pp; extra job losses 5,926 -> 6,910 |
| `global_tightening` | +84 -> +102bp | 6.147%, unchanged | spread -2.95pp, unchanged (its household shock originates at the Fed) |

Here the condition fixes the shocked solve, so the caused move absorbs the
whole fall in the reference rate (about 19bp). The stale start understated the
twin tightening shock: 6.8% is 2.87 points above an observed 3.93%, not 2.4
above a declared 4.4%.

Tests: 13 in `test_starting_state.py`, written first. Two pin the economics:
the caused move is independent of the starting level with no condition on the
RBI, and with one it changes by exactly the fall in the reference rate.

## Consequences

- The headline by group totals stands: -0.18pp per 100bp in the static game,
  -0.56pp in the dynamic one (was -0.55pp). Results quoted on the household-average measure change in the
  second decimal place, and twin tightening is a larger shock than reported.
- The starting state now follows the data: refreshing the RBI workbook can move
  the unprompted moves, the seam's held inflation and the twin and global
  scenarios. The printed line ties each run to the month it started from.
- Still declared and unsourced: the RBI's `current_income_growth` (6.5%), and
  `baseline_inflation=0.044` in `run_pipeline_india.py`, which is off the
  headline chain. Both are recorded for the model-consistency batch.
- Considered and rejected: updating the constant to 0.0393 (stale again next
  month, with no record of the change); declaring conditions as departures from
  the start (a scenario would no longer describe the world, and its meaning
  would change with the data).
