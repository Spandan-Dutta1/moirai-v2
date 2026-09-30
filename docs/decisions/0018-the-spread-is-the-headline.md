# 18. The spread is the headline, and the aggregate is bounded

Date: 2026-09-30
Status: Accepted

## Context

The pipeline reports two numbers from the household layer: the aggregate
change in consumption, and the spread between floating-rate borrowers and
net savers. It led with the aggregate.

ADR 009 found that the banking layer's deposit mechanism fails its
held-out test badly. It predicts that 31 percent of a tightening reaches
deposit rates; over May 2022 to November 2024 the weighted average rate on
fresh deposits rose 243bp against a 250bp repo increase, 97 percent. The
ADR concluded that "any result depending on deposit pass-through is
unreliable", including the saver side of the distributional finding.

Nothing in the pipeline showed which of its two numbers that failure
reaches, or by how much.

Separately, the three headline scripts ran with logging at ERROR, which
discarded every warning, including `still_not_stationary` for Indian CPI
and `specification_failure` for the Indian VAR.

## Decision

- `BankingSystem` gains `deposit_pass_through_override`, `None` by default,
  so every existing result is unchanged. When set, it replaces only the
  system-wide deposit pass-through on tightening moves, which is all the
  held-out observation describes. Lending, easing and group figures keep
  the mechanism.
- `OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH = 243 / 250` records that
  observation with its source. It is used only to bound results, never to
  calibrate: a test fails if the mechanism is ever moved within half a
  point of it, which keeps ADR 009's failure visible.
- `run_pipeline.py` leads with the spread, and runs the headline chain a
  second time with deposits repricing as observed, printing the aggregate,
  the savers' change and the spread at both.
- `run_pipeline.py`, `run_pipeline_india.py` and `run_scenarios.py` log at
  WARNING.

Seven tests cover the override: absent by default, confined to what the
observation describes, bounded to a share, and the held-out figure kept
apart from the mechanism.

## Result

On the real data, the headline chain at each deposit pass-through:

| Deposit pass-through | Aggregate | Net savers | Spread |
|---|---|---|---|
| Mechanism, 31% | -0.085% | -0.043% | -0.61pp |
| Observed 2022-24, 97% | -0.024% | +0.023% | -0.60pp |

The deposit failure moves the aggregate by a factor of three and flips the
sign of the savers' change. It moves the spread by 0.01 points.

The spread is carried by what floating-rate borrowers pay, which the
lending side determines, and the lending side passed its held-out test
(76 percent observed against 80 predicted). The aggregate and the saver
side are carried by deposits, which failed. So the headline is the number
resting on the part of the banking layer that validated, and the number
resting on the part that did not is reported as a range.

## Consequences

- **The finding to quote:** floating-rate borrowers cut consumption by
  about 0.6 points more than net savers, robust to deposit pass-through
  (-0.60 to -0.61) and to both MPC endpoints (-0.54 to -0.61, ADR 017).
- **The aggregate is not a finding.** It is between -0.02 and -0.09
  percent, and whether savers gain or lose depends on a mechanism known to
  be wrong.
- The pipelines now print their warnings, and the first run showed why
  that matters. The Indian pipeline shows its two specification warnings,
  as expected. The headline VAR shows one nobody had seen: on 1985-2007,
  log-differenced US CPI is `still_not_stationary` with verdict
  `inconclusive`. ADF rejects a unit root (p = 0.0046) while KPSS rejects
  stationarity, the pattern of a series drifting slowly downward, which is
  what inflation did across the Great Moderation. The headline holds the
  inflation channel at baseline for India (ADR 010), so the warning does
  not reach households directly, but it is a property of an input to the
  VAR the whole chain uses, and it had been discarded on every run.

## What did not move

The game, the network, the VAR, the calibration and every historical
validation. `run_scenarios.py` reports what it did; only its warnings are
now visible.
