# 14. Sign restrictions do not identify the Indian VAR

Date: 2026-09-28
Status: Accepted

## Context

ADR 007 found that the recursive Indian VAR has output rising after a
rate rise in all four specifications tried. It listed sign restrictions
as an untried alternative. This ADR tries them, on the pipeline's VAR as
it stands, without respecifying it.

The VAR is the one in `scripts/run_pipeline_india.py`: IIP growth, CPI
inflation and the repo rate (differenced by the preparation layer),
February 2012 to May 2026, 170 observations, six lags.

The restrictions, on one shock named "policy":

| Variable | Restriction | Horizons |
|---|---|---|
| Repo rate | rises | impact |
| IIP growth | does not rise | months 0, 1 and 2 |
| CPI inflation | unrestricted, because the price puzzle is in question | |

The other two shocks are unrestricted.

`identify_sign_restrictions` checked impact responses only, so
`SignRestriction` gained a `horizons` field. The default is impact alone,
which leaves every existing result unchanged. A response at horizon h is
Psi_h B, so the moving-average coefficients moved from `irf.py` to
`VARResult.ma_coefficients`, where identification can use them without a
circular import. The tests check each accepted draw's response path
against the closed form A^h B of a simulated VAR(1). They also check that
longer restrictions only shrink the set under a fixed seed.

Everything in `scripts/check_india_sign.py` was fixed before the first
run:

- **Draws:** 20,000 Haar-uniform rotations with seed 42, at the VAR's
  point estimates.
- **Normalisation:** each draw scaled to a 100bp impact on the repo rate.
- **Classification:** IIP growth cannot rise at months 0 to 2 in any
  accepted draw, by construction, so the classification uses the
  unrestricted horizons. It takes the cumulative IIP growth response over
  months 3 to 24 in each accepted draw. The set is mostly negative if at
  least 84 percent of draws are negative, positive if at most 16 percent
  are, and straddles zero otherwise.
- **Acceptance:** a rate near zero would be a finding in itself.

The decision rule was set in advance too. A mostly negative set would
mean Cholesky ordering was the problem, and the restrictions would be
adopted. A set that straddles zero would mean the data cannot tell, and
the range is the report. A positive set would mean something structural
is wrong.

## Result

**Acceptance is 13.3 percent** (2,664 of 20,000). The repo restriction
alone accepts 50.3 percent. Adding IIP on impact only gives 21.9 percent.
The restrictions sit comfortably with the covariance.

**The set straddles zero.** 58.0 percent of accepted draws have a
negative cumulative IIP growth response over months 3 to 24. That is well
short of 84 percent. Cholesky gives a positive response over the same
window.

**The sign changes with the horizon.** The share of draws with IIP growth
below zero:

| Month | 3 | 6 | 9 | 12 | 18 | 24 |
|---|---|---|---|---|---|---|
| Share negative | 89% | 25% | 82% | 16% | 61% | 61% |

At months 6 and 12 the set is mostly positive, despite the restriction.

**The CPI response is unrestricted and ambiguous.** The median is negative
at every horizon shown, but 54 to 65 percent of draws are negative. That
is no evidence either way on the price puzzle.

**The pre-declared normalisation misbehaved.** Scaling each draw to 100bp
of repo impact explodes where a draw barely moves the repo rate, with
tails reaching -108,893pp. Every accepted draw raises the repo rate on
impact, so the scaling is positive and changes no sign. The shares and
the classification are unaffected. The per-100bp magnitudes are not
meaningful and are not reported here.

## What the accepted shocks are

The following diagnostics were added after the first run and are labelled
as such in the script. They explain the result rather than change it.

**The accepted "policy" shocks are mostly output shocks.** Per standard
deviation, the median accepted draw moves the repo rate by 4.5bp (16 to
84 percent: 1.4 to 8.4bp) and IIP growth by -3.5pp (-4.6 to -2.0pp). A
shock that moves the policy rate a few basis points and output growth by
several points is output variance with a small rate move attached. The
restrictions admit it because they ask only for directions, and a
rotation that loads output variance onto the "policy" shock satisfies
them easily.

**The pandemic dominates the covariance.** The residual standard
deviation of IIP growth is 5.04pp against 13.5bp for the repo rate, a
ratio of about 37. March 2020 to December 2021 is 21 of 164 residual
months and carries 49 percent of the IIP residual variance. The rotations
are largely redistributing that variance.

**Year-on-year growth may contribute.** A dip in the level of IIP lowers
year-on-year growth for twelve months and raises it when the dip leaves
the base. The sign pattern above, with a rebound at month 12, is
consistent with that. It has not been tested.

## Decision

The outcome is the second of the three: the data cannot tell. Report the
range and do not adopt sign restrictions for the Indian VAR.
`run_pipeline_india.py` keeps its recursive identification and continues
to withhold its distributional result (ADR 007).

Do not add restrictions, or restrict other shocks, to sharpen the set.
Restrictions on the inflation response or on a demand shock might well
exclude the output-shock rotations. But choosing them after seeing this
set is the same specification search ADR 007 refused. The same holds for
dropping the pandemic months or changing the output measure. Each has a
case, and each is a respecification that would need declaring in advance.

Keep the horizon support in `SignRestriction`. It is a general capability
with tests, independent of this result.

## What external-instrument identification would need

**An OIS series.** The Indian high-frequency literature (Lakdawala and
Sengupta, JMCB 2025) builds surprises from MIBOR OIS rates at 1, 3, 6 and
9 months and one year, in narrow intraday windows around RBI
announcements. Those come from commercial terminal data, not a public
series.

**Announcement dates.** The MPC's June 2026 meeting was its 61st, so
there have been 62 meetings from October 2016 to August 2026. About 61
had a rate decision; the November 2022 meeting drafted the report to
government. Some were off-cycle (March and May 2020, May 2022), and the
announcement time has moved over the period, so each window needs
checking.

**FBIL.** fbil.org.in is reachable, but it is a JavaScript application,
and no data endpoint was found. api.fbil.org.in does not resolve. FBIL's
public MIBOR-OIS curve starts around April 2018 and covers 6 months to 5
years. It is an end-of-day fixing from trades reported by 5pm, so it
gives daily windows, which also take in everything else that happened
that day. It has no 1 to 3 month tenors, and it loses 2016 to 2018. A
late-2025 FIMMDA consultation on SORR-based OIS suggests the MIBOR OIS
market may itself be transitioning.

A usable instrument realistically needs intraday terminal data. The
fallback is a daily FBIL instrument from 2018, with its contamination
declared as a caveat.

## Consequences

**ADR 007 stands, and is now stronger.** The Indian VAR was not
identified by timing, and it is not identified by signs either. The
wrong-signed recursive response is not simply an artefact of the Cholesky
ordering.

**The problem is partly in the covariance.** With the pandemic carrying
half of the output residual variance, any identification that works by
rotating the residuals is working mostly with pandemic variance. That
bears on every method that rotates, not only this one.

**External instruments are the remaining route, and they need data the
project does not have.**
