# 12. The central bank network against calendar 2022

Date: 2026-09-28
Status: Accepted

## Context

ADR 009 tested the banking layer against a cycle it was not calibrated
on. This is the second out-of-sample test, and the first of the central
bank network. Nothing in the network was fitted to 2022: the mandates are
sourced, the weights are assumed or were derived from 1986-2007 US data,
and the spillover matrix is anchored to an IMF estimate published in 2023
about US news shocks generally (ADR 008, ADR 011).

The test is the declared scenario `historical_2022`, run by
`scripts/validate_2022.py`. Each bank starts from its January 2022 policy
rate and faces its 2022 average inflation in its own target measure. The
network is solved once, simultaneously, and the solved moves are compared
with what the banks did over the year.

| Bank | Jan 2022 rate | Condition (2022 mean) | Series |
|---|---|---|---|
| Fed | 0.25% (target upper) | 6.545% | PCE price index, PCEPI |
| ECB | -0.50% (deposit facility) | 8.365% | HICP, CP0000EZ19M086NEST |
| BoE | 0.25% (Bank Rate) | 7.901% | CPI, GBRCPIALLMINMEI |
| BoJ | -0.02% (call rate) | 2.5% | CPI all items, Statistics Bureau |
| RBI | 4.00% (repo) | 6.692% | CPI Combined, RBI |

Inflation is the mean of the twelve monthly year-on-year rates, computed
from index levels, not taken from secondary summaries. Two exceptions. For
Japan, FRED's OECD series ends in June 2021, so the Statistics Bureau's
annual figure is used. For India, the warehouse holds CPI Combined as the
RBI publishes it, year on year. The OECD India index on FRED gives 5.90
percent and is a different measure, so it is not used. The script refuses
to run if the declared conditions, starting rates or observed moves
disagree with the data.

The reference game, against which "caused" is measured, puts every bank
at its inflation target with a zero output gap. The declared scenarios'
default states describe a later period and would be incoherent next to
January 2022 rates, so `historical_2022` carries its own starting state,
`JANUARY_2022_BANKS`.

## Result

| Bank | Model | Unprompted | Caused | Observed | Model / observed |
|---|---|---|---|---|---|
| Fed | +262bp | 0bp | +262bp | +425bp | 0.62 |
| ECB | +401bp | +4bp | +397bp | +250bp | 1.60 |
| BoE | +360bp | +1bp | +359bp | +325bp | 1.11 |
| BoJ | +79bp | +6bp | +74bp | -5bp | wrong sign |
| RBI | +151bp | -37bp | +188bp | +225bp | 0.67 |

**The RBI-to-Fed ratio matches, like for like.** The RBI's total move
against the Fed's total move is 58 percent in the model (151 / 262) and
53 percent observed (225 / 425).

**The correction this ADR exists partly to record.** The question was
first framed as the observed ratio of 53 percent against a modelled 15
percent, and 15 percent suggested the model imports a third of the
spillover it should. That compared two different quantities. The 15
percent (14.7 percent: 39 / 266) is the Fed's *share* of the RBI's move:
the part caused by the Fed's condition alone, divided by the Fed's own
move. The 53 percent is a ratio of *totals*, and the RBI's total includes
its response to India's own inflation. The data cannot separate the two,
so the observed figure has no counterpart to the 15 percent. The
like-for-like comparison is total against total, and that one is close.

**What the RBI's move is made of.** The network is linear while no bank
leaves its band, so the caused move splits exactly into one contribution
per bank's condition. The RBI's +188bp:

| Source | Contribution | Share |
|---|---|---|
| India's own inflation | +105.7bp | 56% |
| Fed | +39.1bp | 21% |
| ECB | +27.4bp | 15% |
| BoE | +14.0bp | 7% |
| BoJ | +1.8bp | 1% |

Against that, the unprompted move is -37bp. With no inflation problem
anywhere, the RBI's external objective pulls it towards the mean of the
other four rates, which in January 2022 were all near zero.

The Fed's share runs almost entirely through the exchange channel. The
RBI's best-response slope to the Fed's rate is +0.128, made of +0.165
from the exchange channel (inflation through `exchange[RBI,Fed]` = 0.421,
which is 1.17 times the 0.36 IMF anchor), -0.060 from the demand channel
(a Fed hike weakens Indian output, which argues for cutting) and +0.023
from the external differential, which is small because its 0.40 weight
is spread across the mean of four banks (ADR 011). The data here does not
identify any of these.

**The Fed and the RBI both under-move by about a third.** Model over
observed is 0.62 for the Fed and 0.67 for the RBI. That is why the ratio
matches: both fall short by about the same proportion. This is not a
systematic under-move across the network, and it should not be described
as one:

- **The ECB overshoots by 60 percent**, +401bp against +250bp. Its
  condition, 8.4 percent HICP, is the highest of the five, and a
  one-shot quadratic loss responds to it in full. The ECB in 2022 began
  late, from a negative rate, and was still raising in 2023.
- **The BoE is roughly right**, +360bp against +325bp, 1.11.
- **The BoJ has the wrong sign.** The model tightens by 79bp: its own
  condition, 0.5 points above target, contributes +24bp, and spillovers
  from the other four banks contribute +50bp. The BoJ held its policy
  rate at -0.10 percent throughout 2022 under yield curve control, and the
  call rate drifted down 5bp. The model has no mechanism for a bank that
  judged imported inflation transitory and chose not to respond.

**The RBI ends outside its band.** At equilibrium Indian inflation is
6.46 percent, above the six percent ceiling. The RBI's 151bp hike takes
1.2 points off inflation through the own-rate effect, but the other banks
tighten more, and the exchange channel adds most of it back. With the
other banks at their 2022 conditions, any RBI condition above **5.34
percent** leaves it outside the band at equilibrium. The imported
inflation brings that threshold below the ceiling itself. The 7.79
percent April peak is not the comparable number. The model takes one
static condition, and the annual mean of 6.69 percent already breaches.

This is the first declared scenario to leave a bank outside its band, and
ADR 011 said what follows. The reaction system omits the band penalty, so
the simultaneous solver is approximate here. Holding the other rates at
their solved values, the RBI's best reply under its true loss is 5.62
percent against a solved 5.51 percent, 10.8bp higher. That is a best
reply, not a re-solved equilibrium, so it bounds the error rather than
correcting it.

## Caveats on the test itself

**A one-shot game against a year of decisions.** The model solves one
simultaneous move from a static state. The banks made between four and
eight decisions each over 2022, against inflation that was rising for most of
it and against forecasts, not realised annual means. The under-move of
the Fed may be as much about this as about any parameter: the Fed in 2022
was catching up to inflation it had earlier called transitory, which a
static loss cannot represent.

**The condition is a choice.** The annual mean of monthly year-on-year
rates is one reasonable summary. The ratio of annual averages differs by
at most 0.02 points for the three computed series. The peak, or the
December figure, would give different conditions and different answers.
The mean was chosen before the model was run and is kept.

**The Bank of England's rates are not on FRED.** Bank Rate is taken from
the Bank's published decisions: 0.25 percent in January, 3.50 percent in
December. SONIA, which is on FRED, moved from 0.191 to 3.428 percent.

**One year, five banks.** The comparison is five numbers, one of which
has the wrong sign. It is not a statistical test.

## Decision

Report the result and tune nothing. Adjusting the weights, the spillover
matrix or the ECB's and the BoJ's preferences after seeing 2022 would
turn the network's only held-out test into an in-sample fit, which is the
same reasoning as ADR 009.

Declare `historical_2022` with its own starting state and observed moves
in `src/moirai/engine/scenarios.py`, so the test is reproducible offline.
`scripts/validate_2022.py` recomputes every condition from the data and
refuses to run on a disagreement. The design itself lives in
`scripts/historical_validation.py`, shared with ADR 013, so that a second
year cannot quietly change it.

Tests in `tests/unit/test_scenarios.py` pin two properties for every
historical scenario:
`test_a_historical_caused_move_splits_exactly_by_condition`, on which the
attribution above rests, and
`test_the_simultaneous_solver_is_approximate_when_the_band_binds`, which
measures the gap ADR 011 predicted.

## Consequences

**The network gets the relative size of the RBI's response right in one
year, and the absolute size wrong by a third.** Scenario results that
depend on the RBI's move relative to the Fed's are better supported than
before. Results that depend on the size of either move are not.

**The Fed's share of any RBI move is unvalidated.** The 21 percent Fed
share rests on the spillover matrix's allocation, which ADR 008 already
marks as not sourced, and 2022 cannot test it because the data only
records totals.

**The simultaneous solver is no longer exact for every declared
scenario.** The README's claim that no declared scenario breaches the
band is withdrawn. Fixing it would mean adding the band penalty to the
reaction system, which is a change to the solver with its own
consequences for linearity and for the exact decomposition above. It is
not done here.

**The household chain has not been run on this scenario.** Carried by
the synthetic impulse response the unit tests use, the RBI's 188bp caused
move drives the income growth path outside the plausible range at the
Layer 1 to Layer 3 seam, and the guard refuses it. The guard is not
widened. Whether the estimated US transmission would also be refused has
not been checked. The scenario's purpose is the network test, and no
household result is claimed for it.

**The next real improvement is dynamics, not weights.** Both the Fed's
under-move and the BoJ's wrong sign point at the one-shot structure: a
bank that responds to forecasts, or that can wait, is not in the model.
That would be a mechanism change with its own held-out test.
