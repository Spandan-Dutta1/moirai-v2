# 26. Policy inertia in the dynamic game

Date: 2026-10-01
Status: Accepted

## Context

ADR 023's dynamic game makes the RBI's caused move largest in the first
quarter (+16.3bp) and then fade into easing from the fourth quarter. ADR
022 measured the opposite shape: after a Fed tightening surprise India's
repo rate rises for about two quarters and stays up, at 27, 61, 99, 98 and
100 percent of its peak at 1, 3, 6, 9 and 12 months with Indian CPI
controlled.

Nothing in the game makes a bank move gradually. Its smoothing term
penalises distance from the current rate, not the size of each step, so a
bank's best response to a shock is to move at once. Real central banks
change rates in steps and avoid reversals; estimated policy rules put
large weight on the previous rate.

## Decision

`DynamicParameters.inertia_weight`, zero by default, adds a cost on the
squared change in each bank's rate from the previous quarter. With it on,
last quarter's moves join the state and each bank's rule responds to its
own previous move as well as to the economy. With it off the state is
unchanged, so ADR 023's results reproduce exactly.

**Calibration on a published estimate, not on the Indian data.** The
weight is set so that the Fed's equilibrium rule puts 0.79 on its own
previous rate, the quarterly smoothing coefficient Clarida, Gali and
Gertler (2000) estimate for the Volcker-Greenspan Fed. Solving gives
`CALIBRATED_INERTIA_WEIGHT = 6.97`. The same weight applies to every bank;
the RBI's implied coefficient is 0.67.

`smoothing_coefficient` reads a bank's coefficient from its rule.
`scripts/check_dynamic_calibration.py` reproduces the calibration and the
checks below; it needs no data.

Seven tests: zero inertia keeps the original state; inertia adds lagged
moves to it; rules remain best responses with inertia; the calibrated
weight gives the Fed 0.79; the RBI's caused move builds up rather than
jumping; inertia makes every bank's path smoother.

## Result

**The calibration**, weight against implied smoothing:

| Weight | 0.5 | 1 | 2 | 4 | 7 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| Fed | 0.51 | 0.60 | 0.68 | 0.75 | 0.79 | 0.81 | 0.86 |
| RBI | 0.30 | 0.41 | 0.51 | 0.60 | 0.67 | 0.71 | 0.77 |

**The RBI's caused path**, quarters 0 to 7, basis points:

| | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 |
|---|---|---|---|---|---|---|---|---|
| No inertia (ADR 023) | +16.3 | +10.0 | +5.0 | +1.3 | -1.3 | -2.8 | -3.6 | -3.9 |
| Calibrated inertia | +0.4 | +0.7 | +0.9 | +1.0 | +1.0 | +0.9 | +0.7 | +0.5 |

**The check against India's measured response**, on shape (squared gaps
between paths normalised to their peaks, quarters 0 to 4):

| Measured series | No inertia | Calibrated |
|---|---|---|
| Repo 2012-2023, Indian CPI controlled | 2.987 | **0.039** |
| Repo 2012-2023, uncontrolled | 2.660 | **0.043** |
| Discount rate 2000-2022 | 1.035 | 1.043 |

Calibrated on the Fed alone, the model reproduces the shape of the RBI's
response in the current regime almost exactly, and nothing about the
Indian data was used to get there. It does not match the pre-2013 regime,
when the RBI's response peaked at three months and faded within a year,
the managed exchange rate period ADR 019 found unresponsive to the Fed.

## The remaining gap: magnitude

With inertia, both banks respond gradually to a shock that fades with it,
so both move less: the Fed's caused move peaks at about 9bp rather than
88, and the RBI's at about 1bp. The RBI's peak relative to the Fed's falls
from 0.24 to 0.12. ADR 022 found the real RBI follows the Fed by far more
than the static model's RBI; the calibrated dynamic RBI follows it by less
still.

So the dynamic game now has the right timing and the wrong size. The size
is set by how strongly the RBI responds to the Fed at all, the
exchange-rate channel and the RBI's external objective, not by inertia,
and calibrating that to the RBI's measured response relative to the Fed's
is the next step. Until then the dynamic chain's magnitude is not a
defensible headline, and step 4 waits for it.

## Consequences

- The dynamic game answers how quickly the RBI follows the Fed in a way
  consistent with India's data: it builds up over two quarters and
  persists, as the repo rate does.
- The static game stays the headline; the dynamic game supplies timing.
- `check_dynamic_game.py` and every other output are unchanged.
