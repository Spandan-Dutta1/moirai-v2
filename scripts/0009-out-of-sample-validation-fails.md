# 9. The banking layer fails two of three out-of-sample predictions

Date: 2026-08-14
Status: Accepted

## Context

ADR 008 identified the banking layer's pass-through coefficients as fitted
to their target: they were adjusted until the model reproduced the RBI's
easing-cycle figures, and that agreement was then reported as a result.
Five of six group checks passed, by construction.

The RBI also publishes transmission for the tightening cycle of May 2022
to November 2024, against a cumulative 250 basis point hike. Those figures
were not used in calibration, so they are a genuine held-out test.

## Result

**1 of 3 predictions survives.**

**Passes: the lending asymmetry.** The model predicts 80 percent
pass-through when tightening against 66 percent when easing. The data
shows 76 percent in tightening. The mechanism produces roughly the right
magnitude on a cycle it never saw, and the direction is the model's
central claim.

**Fails: the public-private ordering.** The model gives private banks a
higher external benchmark share and lower stress, so it predicts they
transmit more in both directions: 88 percent against 75. The data shows
the ordering reverses under tightening, with public banks at 73 percent
and private at 71. The model has no mechanism that could produce a
reversal.

**Fails badly: deposit pass-through.** The model predicts 31 percent; the
observed figure is 97 percent. The assumption that banks are slow to raise
deposit rates held early in the cycle, when median term deposit rates rose
48 basis points against a 190 point EBLR move. Over the full cycle the
weighted average rate on fresh deposits rose 243 points against a 250
point hike. Deposits repriced sharply once surplus liquidity drained, and
the model has no liquidity state, so it cannot produce a late catch-up.

## Caveats on the test itself

The two cycles are not the same sample. Calibration used February 2025 to
May 2026; the test uses May 2022 to November 2024, so the composition of
the banking system differs between them.

The published measures also disagree with each other. Fresh against
outstanding, median against weighted average, and the window chosen all
change the number materially, and the deposit failure above is partly a
disagreement between measures rather than purely a model error.

## Decision

Report the failures rather than tuning until they pass. Adding a liquidity
state or a mechanism that reverses the group ordering, after seeing the
data those mechanisms need to reproduce, would convert a held-out test
into a second in-sample fit and destroy the only real validation the
project has.

## Consequences

**The banking layer is a mechanism, not a calibrated forecast.** It
produces the asymmetry it was designed to produce and gets its magnitude
roughly right. It does not predict group-level transmission and it does
not predict deposit rates.

**Any result depending on deposit pass-through is unreliable.** That
includes the saver side of the distributional finding, which is the
smaller of the two effects but is not nothing.

**The next real improvement is a liquidity state**, not a better fit. The
deposit failure has an identified cause, and modelling it would be a
change to the mechanism rather than to its parameters.

**A model that passed everything after being tuned on half the data would
be more suspicious than one that fails two of three.** The in-sample
result was 5 of 6; the out-of-sample result is 1 of 3. That gap is the
honest measure of how much the calibration was fitting.