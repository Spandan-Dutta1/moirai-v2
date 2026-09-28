# 13. The central bank network against 2013, the taper tantrum year

Date: 2026-09-28
Status: Accepted

## Context

ADR 012 tested the network against calendar 2022 with a fixed design.
This applies the same design, unchanged and untuned, to 2013, the year of
the taper tantrum. The design is in `scripts/historical_validation.py`,
shared by both years. The declared scenario is `historical_2013`, run by
`scripts/validate_2013.py`.

The design: one calendar year; the rate in force on 1 January and on 31
December; each bank's condition is the mean of its twelve monthly
year-on-year inflation rates in its own target measure; the reference game
puts every bank at target with a zero output gap; the network is solved
simultaneously with the sourced spillover matrix.

| Bank | 1 Jan 2013 rate | Condition (2013 mean) | Series |
|---|---|---|---|
| Fed | 0.25% (target upper) | 1.319% | PCE price index, PCEPI |
| ECB | 0.00% (deposit facility) | 1.354% | HICP of the 17-member euro area, CP0000EZ17M086NEST |
| BoE | 0.50% (Bank Rate) | 2.293% | CPI, GBRCPIALLMINMEI |
| BoJ | 0.083% (call rate) | 0.338% | CPI all items, JPNCPIALLMINMEI |
| RBI | 8.00% (repo) | 10.072% | CPI Combined, RBI |

Every condition except India's is computed from index levels; FRED's
Japanese CPI still covers 2013, so no figure is declared. The 19-member
HICP gives the same 1.35 percent. The OECD India index gives 10.92
percent and is a different measure. The RBI's starting rate is the
December 2012 month-end value, because the warehouse's January 2013 value
already includes the cut of 29 January.

### What the design can test

**Not the tantrum.** The tantrum, from May 2013, was a shock to expected
Fed policy and to term premia, with no move in the Fed's rate. The RBI's
main defence was a 200bp rise in its marginal standing facility rate in
July, unwound by October, not the repo rate. The design can express
neither. It conditions banks on their inflation and transmits rate moves
between them, so it tests whether the network explains 2013's rate
decisions and nothing more.

**Three anachronisms are carried, not corrected.** The model gives every
bank today's mandate. The RBI had no inflation target in 2013; the
four percent CPI target and the two to six percent band date from 2015-16,
and in 2013 it gave weight to wholesale prices. The ECB's target was
"below but close to" two percent, not the symmetric two percent of 2021.
And the model has no lower bound on rates, while four of the five banks
were at or near zero. Replacing any of these after seeing the result would
be the specification search the fixed design exists to prevent.

## Result

| Bank | Model | Unprompted | Caused | Observed |
|---|---|---|---|---|
| Fed | -39bp | 0bp | -39bp | 0bp |
| ECB | -36bp | +5bp | -41bp | 0bp (MRO -50bp) |
| BoE | +7bp | +3bp | +4bp | 0bp |
| BoJ | -67bp | +12bp | -79bp | -1.3bp |
| RBI | +153bp | -71bp | +224bp | -25bp |

**The RBI has the wrong sign, by 178bp.** The model hikes 153bp; the RBI
cut 25bp over the year. Almost all of the model's move is India's own
condition, +238bp of the +224bp caused, from 10.07 percent CPI against a
four percent target. The RBI in 2013 was not targeting that measure at
that level. The test cannot separate how much of the failure is the
anachronistic mandate and how much is the model, and the ADR does not
try to.

**The calendar window hides the tantrum response.** The RBI's -25bp nets
three 25bp cuts, in January, March and May, against two 25bp hikes in
September and October. The design fixes the window at the calendar year, so that
netting is part of the observed figure, as the design requires.

**Three banks solve below zero.** Fed -0.14 percent, ECB -0.36 percent,
BoJ -0.59 percent. Each had inflation below target and the model has no
floor. All three stayed at or near their floors in reality. The ECB did
cut its main refinancing rate by 50bp, and the Bank of Japan began
quantitative and qualitative easing in April. Easing in those directions
is consistent with the model's sign, but none of it is a policy rate move
the design can compare, and it is not counted as agreement.

**The Bank of England is close**: +7bp against no move.

**The spillover the model transmits runs the wrong way for a tantrum.**
The Fed's condition, inflation below target, makes the Fed cut in the
model, and that pushes the RBI down by 5.9bp. The tantrum's pressure on
India was a tightening. A network that transmits rate moves cannot see a
shock to expected rates, and in 2013 the rate move and the expectation
pointed in opposite directions.

**The Fed's share is a constant of the matrix, not a result.** The Fed's
share of the RBI's move is 14.7 percent in 2013, as it was in 2022. The
system is linear, so the RBI's response to the Fed's condition, divided by
the Fed's response to it, depends only on the weights and the spillover
matrix. Neither year says anything about it. The ratio of total moves is
undefined in 2013 because the Fed did not move.

**The RBI ends outside its band again.** Its inflation at equilibrium is
7.13 percent. With the other banks at their 2013 conditions, any RBI
condition above 6.75 percent breaches; the threshold is higher than in
2022 (5.34 percent) because the other banks ease rather than tighten, so
less inflation is imported. The solver's error is larger: the RBI's best
reply under its true loss is 9.80 percent against a solved 9.53 percent,
26.7bp higher, against 10.8bp in 2022.

## Caveats on the test itself

**A one-shot game against a year of decisions**, as in ADR 012.

**The unprompted move is large.** -71bp for the RBI, because it starts
eight points above the mean of the others and its external objective
pulls it towards them. That is a property of the reference state the
design fixes. It is larger in 2013 than in 2022 because the gap was
larger.

**Five numbers again, four of them zero or near zero.** Only the RBI's
move is large enough to compare as a ratio, and it has the wrong sign.

## Decision

Report the result and tune nothing. Do not replace the RBI's mandate
with a 2013 one, add a lower bound, or move the window to May-December,
after seeing the result. Each has a case, and each would need to be
declared before it is run, with its own held-out year.

Declare `historical_2013` with its starting state and observed moves.
Move the design into `scripts/historical_validation.py` so both years run
through the same code; the 2022 results were unchanged by the move.

## Consequences

**The network fails 2013 on the one move large enough to judge.** Taken
with ADR 012, the network got the relative size of the Fed's and the
RBI's moves right in one year and the direction of the RBI's move wrong
in another. That is not a validated network.

**Historical tests need a declared mandate history.** A model that
assigns today's mandates to earlier years will fail those years for
reasons that have nothing to do with spillovers. Declaring a dated
mandate for each bank, before running any further year, would make later
tests informative. That is a model change with its own ADR.

**The network cannot represent a shock to expected policy.** The tantrum,
forward guidance and balance sheet policy all move markets without moving
the policy rate. Scenarios built on this network are about rate moves
only, and should not be read as covering episodes driven by expectations.

**A lower bound is a known gap.** Three of five banks solve below zero in
2013. Scenarios that start near zero with inflation below target produce
moves no bank could make.

**The simultaneous solver's error grows with the breach.** 10.8bp in
2022, 26.7bp in 2013. Adding the band penalty to the solver is the
remedy; ADR 012 left it undone for reasons that still hold.
