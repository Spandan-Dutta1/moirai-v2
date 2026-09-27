# 10. Indian-origin scenarios carry the US transmission with inflation held and the horizon capped

Date: 2026-09-28
Status: Accepted

## Context

The household layer is calibrated to Indian data: AIDIS 2019, PLFS
2023-24, and a 75 percent floating-rate share of debt. The only credibly
identified transmission is the US VAR(12) on 1985-2007 data (ADR 007), so
the scenarios fed Indian households a Federal Reserve rate path.

The intended chain is an imported shock: the Fed tightens, the spillover
reaches India, the RBI responds, Indian banks reprice and Indian
households bear it. Every layer for that exists. An RBI-origin run takes
its magnitude from the RBI's equilibrium move in the policy game and its
shape from the US impulse response. The magnitude is Indian; the shape is
American.

### The US shape against published Indian estimates

| | US VAR(12), 1985-2007 | Indian estimates |
|---|---|---|
| output | growth dips from month 4, mostly negative months 6-24 | falls after 2-3 quarters |
| inflation | rises for 4 months; price level +0.14% at month 36 | falls after 3-5 quarters |
| persistence | cumulative rate response not halved from its month 10 peak by month 60 | effects persist 8-10 quarters |

Indian sources: Mohanty (2012), a quarterly structural VAR, finds output
falling after two quarters and inflation after three, with effects
persisting eight to ten quarters. Khundrakpam and Jain (2012) give lags of
two to three quarters for output and three to four for inflation, with
the same persistence. Kapur and Behera (2012) find a peak effect on
non-agricultural growth after two quarters and on non-food manufactured
products inflation after five. All three use pre-2016 samples and
recursive-type identification, so they are weaker benchmarks than they
look. They still agree with each other on the sign of the price response,
and the US response contradicts it.

Output timing roughly matches. Inflation has the wrong sign, and US
persistence is more than twice the Indian range. Output timing alone
cannot carry the proxy.

A note on the persistence figure. The VAR's reported half-life, 12.7
months, is the decay of the system's slowest eigenmode, not of the rate
response. A 16.9 month figure that circulated came from the 1960-2019
full-sample VAR, which the pipeline rejects for failing specification
tests. Neither describes how long the policy shock lasts.

### Alternative tried: a longer Indian sample

ADR 007 lists a longer sample as a remedy, so it was tested before
falling back on the proxy. The RBI table has industrial production and
the repo rate from December 2010, but CPI Combined only from 2012. WPI
covers the longer span. Substituting WPI for CPI and starting the sample
at the earliest common month adds 12 observations (170 to 182). The
cells were fixed before estimation:

| price series | sample from | lags | diagnostics usable | output peak |
|---|---|---|---|---|
| CPI (ADR 007 baseline) | 2012-02 | 3 | no | +1.15 at h=1 |
| CPI (ADR 007 baseline) | 2012-02 | 6 | no | +1.41 at h=1 |
| WPI | 2012-02 | 3 | no | +1.19 at h=1 |
| WPI | 2012-02 | 6 | no | +1.38 at h=1 |
| WPI | 2011-01 | 3 | no | +1.21 at h=1 |
| WPI | 2011-01 | 6 | no | +1.30 at h=1 |

Output still rises after a tightening in every cell and under all six
orderings. WPI removes the CPI autocorrelation failure but the IIP one
remains, because it comes from year-on-year construction (ADR 006) and
no sample length removes it. WPI is also not the RBI's target measure.
These are the fifth and sixth specifications tried, and they are
reported as negative results rather than searched further.

## Decision

For scenarios whose household-facing shock originates at the RBI:

1. **Hold inflation at baseline.** The inflation path is pinned at the
   RBI's current inflation for the whole horizon. This asserts no price
   response, which is weaker than asserting a response the Indian evidence
   contradicts.

2. **Cap the horizon at 30 months** rather than 36, the top of the Indian
   eight to ten quarter range. Beyond it, the path is US persistence
   without Indian support.

Both are declared on the scenario, not applied silently. `HeldChannel`
and `HorizonCap` in `src/moirai/engine/scenarios.py` each carry the
reason, the reason travels into the ledger, and `run_scenarios.py` prints
both. The shared declarations are `INFLATION_HELD_FOR_INDIA` and
`HORIZON_CAPPED_FOR_INDIA`. A test requires every declared RBI-origin
scenario to carry both.

`imported_tightening` is the headline Indian-origin scenario, reported
alongside `us_inflation_shock`. It conditions only the Fed, at 4.5 percent
inflation, so the RBI's move against the unconditioned game is wholly
imported: +12.9 basis points, 0.3 standard deviations of the US shock.
Indian households lose 0.066 percent of consumption in aggregate and
floating-rate borrowers 0.59 percent, against 0.45 and 2.94 percent when
the Fed's own path is delivered. That is the honest size of an imported
shock. It is small because the spillover is small, not because of the
restrictions this ADR imposes: holding inflation and capping the horizon
change the aggregate by less than 0.003 percentage points.

### Related changes made at the same time

These define the magnitude of an RBI-origin shock, so they are recorded
here.

**The shock is a counterfactual difference.** The game is now solved
with and without the scenario's conditions, in the same mode, and the
origin bank's move is the difference. It used to be measured from the
bank's current rate, which counted the move the bank makes with no
conditions at all. In the default network the RBI goes from 5.25 to
5.31 percent unprompted. That is 5.8 of the 18.7 basis points it moves in
the US inflation scenario, and none of it is imported. The Fed moves 58
of its 146 basis points unprompted, so every scenario's shock shrinks:

| scenario | origin | shock before | shock after | aggregate before | aggregate after |
|---|---|---|---|---|---|
| us_inflation_shock | Fed | +145.8bp | +87.7bp | -0.749% | -0.453% |
| fed_leads | Fed | +150.0bp | +87.6bp | -0.770% | -0.452% |
| twin_tightening | RBI | +113.0bp | +107.1bp | -0.592% | -0.576% |
| global_tightening | Fed | +145.1bp | +87.0bp | -0.747% | -0.450% |

The twin_tightening "after" figure also includes this ADR's declarations
and the income baseline change below. The fed_leads "after" figure also
includes the Stackelberg solver fix of the same date. The old solver
pinned the leader by overwriting its current rate, which hid the leader's
move from every spillover channel. Solved properly, Fed leadership is
almost indistinguishable from simultaneous play, because the followers'
responses barely reach the anchor economy.

**Income growth comes from the bank.** `CentralBank.current_income_growth`
now supplies the income baseline, as `current_rate` and
`current_inflation` already supplied theirs. It used to be a separate
argument defaulting to 2 percent, so an RBI-origin run gave Indian
households American income growth. The Fed's 2 percent and the RBI's 6.5
percent are carried over from their former call sites and are unsourced.
A bank without a value cannot originate a household path.

## Consequences

**What holding inflation costs.** In the model, inflation erodes real
wealth and real debt. Measured on the RBI-origin runs:

| outcome | imported_tightening, free / held | twin_tightening, free / held |
|---|---|---|
| aggregate consumption | -0.066% / -0.067% | -0.575% / -0.588% |
| floating-rate borrowers | -0.539% / -0.571% | -3.290% / -3.458% |
| fixed-rate borrowers | -0.070% / -0.107% | -0.622% / -0.851% |
| job losses | unchanged | unchanged |

The aggregate, the spread and job losses barely move. They run through
debt service and employment, and employment does not read inflation.
**Fixed-rate borrowers are the exception.** Their result moves by a third
to a half, because without a rate channel the inflation erosion of their
debt is a large share of what happens to them. The US sign relieves them;
the Indian sign would burden them further. Holding at baseline sits
between the two, so the fixed-borrower loss is probably understated
relative to what the Indian evidence implies.

**The cap truncates rather than resolves.** Households are simulated for
31 periods instead of 37. Effects in months 31 to 36 are dropped, not
returned to baseline, and job losses fall mechanically because fewer
months are counted.

**The remaining shape is still American.** Within 30 months the rate path
peaks at month 10 and the output response follows US timing. That timing
roughly matches Indian estimates, which is why the output channel is kept,
but it is not Indian evidence.

**The result stays provisional.** The declarations stop the proxy from
asserting what the Indian evidence contradicts. They do not make it an
Indian transmission. The banking layer's out-of-sample failures (ADR 009)
and the fitted-to-target parameters (ADR 008) still apply. No result from
this chain is validated.

**When to revisit.** If an Indian identification becomes credible, from
external instruments around RBI announcements or narrative identification
from MPC minutes (ADR 007), the proxy and both declarations should go.
