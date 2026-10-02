# 27. The RBI's response to the Fed, calibrated, and the dynamic headline

Date: 2026-10-01
Status: Accepted

## Context

ADR 026 gave the dynamic game the right timing: with policy inertia
calibrated to the Fed's estimated smoothing, the RBI's caused move builds
up over two quarters and persists, matching India's measured response in
shape. It left the size wrong. The calibrated dynamic RBI moved 0.12
points per point of the Fed's move, the static one 0.15, and ADR 022 found
the real RBI follows the Fed by far more.

## What was measured

`scripts/check_rbi_fed_ratio.py` estimates both sides the same way: the US
two-year Treasury yield and the RBI repo rate, each by local projection on
the same Bauer-Swanson surprises, August 2012 to December 2023, with the
same controls (Indian CPI for the repo rate). The two-year yield stands
for expected US policy because the funds rate sat at zero for 2012-2015.

| Months | US two-year | RBI repo | RBI / US |
|---|---|---|---|
| 0 | +1.55 (t 2.5) | +1.00 (t 2.1) | 0.65 |
| 1 | +2.79 (t 2.5) | +1.33 (t 2.3) | 0.47 |
| 3 | +2.71 (t 1.8) | +3.06 (t 3.9) | 1.13 |
| 6 to 12 | not significant | +5.0 (t 2.0 to 3.3) | undefined |

The script's declared rule, horizons of three months or more, found no
usable ratio, because the surprise's effect on the US two-year yield fades
within a quarter while the RBI's response builds and persists. Where both
responses are significant the RBI moves 0.47 to 0.65 per point; every
horizon at which a ratio can be formed gives 0.47 or more. Using the early
horizons is a choice made after seeing the data, so the figure is used as
a floor, not an estimate: **the RBI moves at least 0.47 points per point
of expected US policy.**

## A ceiling in the model

`scripts/check_rbi_following.py`. The RBI's external objective penalised
its rate gap to the simple average of the other four banks, so a Fed move
counted a quarter. No external weight could make the RBI follow the Fed
by more than about 0.3:

| RBI external weight | 0.4 | 1 | 2 | 5 | 10 | 50 |
|---|---|---|---|---|---|---|
| Static, equal reference | 0.15 | 0.17 | 0.20 | 0.24 | 0.28 | 0.32 |
| Dynamic, equal reference | 0.12 | 0.15 | 0.19 | 0.23 | 0.26 | 0.29 |
| Static, dollar reference | 0.20 | 0.28 | 0.38 | 0.55 | 0.67 | 0.82 |
| Dynamic, dollar reference | 0.21 | 0.35 | 0.48 | 0.65 | 0.74 | 0.84 |

The structure, not a parameter value, made the model's RBI too passive,
which is the gap ADR 022 found.

## Decision

- `SpilloverMatrix.external_reference`, `None` by default so every
  validated result reproduces, weights each bank's external objective over
  the others; `with_reference` sets one bank's weights. Both the static and
  the dynamic solver use it.
- The RBI's reference is weighted by invoicing currency: the Fed's rate
  carries `INDIA_DOLLAR_INVOICING_SHARE = 0.86`, the share of India's
  imports invoiced in dollars against 5 percent originating in the United
  States (Gopinath 2015), and the rest is split equally.
- `CALIBRATED_RBI_EXTERNAL_WEIGHT = 1.92`: with that reference and the
  calibrated inertia, the dynamic RBI's peak caused move is 0.47 times the
  Fed's. The validated RBI default of 0.40 is unchanged.
- `dynamic_chain.py` runs a scenario through the chain on the dynamic
  game: households face the RBI's own caused rate path, held for each
  quarter's three months, with no VAR and no rescaled shock.
  `calibrated_inputs` assembles the game of ADRs 026 and 027.
- `run_pipeline.py` reports the headline on the calibrated dynamic game,
  on the capped population of ADR 025, beside every existing figure.
- Seventeen tests: equal weights reproduce the original equilibrium to
  1e-12; best replies hold under the weighted loss in both solvers; dollar
  weighting raises the RBI's response; the calibrated RBI meets the floor
  and keeps the measured shape; the validated default is untouched; the
  dynamic chain converts quarters to months correctly and runs through to
  households.

## Result

**The calibrated RBI**, caused move by quarter, basis points:

| | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 |
|---|---|---|---|---|---|---|---|---|
| Fed | +4.7 | +7.4 | +8.7 | +9.0 | +8.6 | +7.8 | +6.8 | +5.7 |
| RBI | +1.6 | +3.0 | +3.8 | +4.2 | +4.2 | +3.9 | +3.3 | +2.7 |

It follows the Fed by 0.47, and its shape against India's measured
response is 0.031, closer than with inertia alone (0.039). Size and timing
are both disciplined by data and neither was traded for the other.

**The households**, capped population, by group totals: floating-rate
borrowers -0.083 percent, net savers -0.034 percent, spread -0.049 points,
aggregate -0.043 percent, nobody consuming nothing.

**Per 100bp of Fed tightening**, the comparable measure across the two
games:

| | RBI follows the Fed by | Spread per 100bp of Fed tightening |
|---|---|---|
| Static game | 0.15 | -0.18pp |
| Calibrated dynamic game | 0.47 | **-0.55pp** |

With lags and inertia, a US inflation shock that fades moves the Fed
gradually and by less, peaking at 9bp rather than 88bp, so the dynamic
game's level is smaller. Per point of Fed tightening, the calibrated RBI
follows the Fed three times as closely as the static one, and floating-rate
borrowers bear three times as much.

## Consequences

- **The finding, end to end:** per 100bp of Fed tightening, floating-rate
  borrowers in India cut consumption by 0.18 to 0.55 points more than
  savers: 0.18 in the static game, whose RBI follows the Fed less than the
  real one does, and 0.55 in the dynamic game, whose RBI is calibrated to
  the least the data supports. Both rest on the capped population and the
  group-total measure. The original figures are reported beside them
  unchanged.
- The dynamic result is a lower bound on the dynamic model's answer, since
  it is calibrated to the floor of the measured response.
- The scenario's level, how far the Fed moves for a given US inflation
  shock, depends on how persistent the shock is and on the Fed's inertia.
  The per-100bp figure removes that dependence and is the one to quote.
