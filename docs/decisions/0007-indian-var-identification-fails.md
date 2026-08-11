# 7. The Indian monetary VAR is not credibly identified

Date: 2026-08-12
Status: Accepted

## Context

A three variable recursive VAR on RBI data (industrial production, CPI
inflation, policy repo rate; 2012-02 to 2026-05; 170 observations) produces
an industrial production response to a policy tightening that peaks at
about +1.2 percentage points one month after the shock.

Output rising after a rate rise is not a plausible monetary transmission.
It is the output-side analogue of the price puzzle: the RBI tightens when
it expects strength, and a VAR that cannot observe what the RBI observed
attributes the subsequent strength to the tightening.

## What was tried

The ten year G-Sec par yield was added as a fourth variable, ordered after
the policy rate. Bond yields embed market expectations of growth and
inflation, which is close to the information a policy maker acts on, and
they reprice within minutes of an announcement, so the ordering is
defensible. This is the same class of remedy as Sims' commodity price
augmentation for the price puzzle.

Four specifications were estimated, at three and six lags, with and without
the yield:

| specification | lags | obs/param | output peak |
|---|---|---|---|
| three variable | 3 | 16.7 | +1.15 |
| three variable | 6 | 8.6 | +1.41 |
| with 10y yield | 3 | 12.7 | +1.26 |
| with 10y yield | 6 | 6.5 | +1.33 |

The sign does not move. The price response is correctly negative in all
four, between -0.13 and -0.18, so this is not a general identification
failure but a specific one in the output equation.

The stability of the wrong sign across specifications is itself evidence.
A fragile artefact would move when the specification moved.

## Decision

Report the Indian monetary VAR as not credibly identified, and do not use
its impulse responses to drive a distributional simulation.

The Indian pipeline is retained and runs end to end, because it
demonstrates that the data fabric, the metadata-driven preparation logic
and the unit handling all work correctly on real Indian data. What it does
not demonstrate is a credible monetary shock.

## Why the problem is harder here

The sample is 170 monthly observations spanning demonetisation in 2016,
the GST transition in 2017, the pandemic, and a full easing and tightening
cycle. That is a large number of regime-relevant events for the number of
observations available.

India also adopted flexible inflation targeting only in 2016, so the first
four years of the sample are drawn from a different policy framework than
the rest. A constant coefficient VAR across that boundary is estimating an
average of two regimes.

## What would be needed

- A longer sample, which requires splicing across the CPI and IIP rebasings
- An identification strategy that does not rest on contemporaneous timing:
  external instruments from monetary policy surprises around RBI
  announcements, or narrative identification from MPC minutes
- Or a sign restriction approach, which is already implemented, though a
  set-identified result would not resolve the underlying information
  problem

## Consequences

**The US pipeline remains the demonstration of the full chain**, because
its shock is credibly identified and passes specification tests.

**The Indian pipeline demonstrates Layer 0 and the metadata logic**, which
is a real result: the preparation layer correctly declined to difference an
already-differenced series based only on the unit recorded at ingestion.

**A negative result reported is better than a positive result forced.** The
alternative was to keep adding variables until the sign flipped, which
would be specification searching, and the resulting model would be fitted
to a prior rather than to the data.