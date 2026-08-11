# 6. The Indian VAR uses year-on-year series, with a stated limitation

Date: 2026-08-12
Status: Accepted

## Context

RBI's monthly indicators table publishes industrial production and CPI as
year-on-year percent changes rather than as index levels. A VAR estimated
on those series fails its residual autocorrelation tests at every lag
order tried, and adding lags does not help.

The cause is mechanical rather than a modelling error. A year-on-year rate
is a twelve month difference, so consecutive observations share eleven
months of underlying data. The overlap induces serial correlation in the
residuals by construction, in the same way overlapping return regressions
do. No lag length can absorb it because it is a property of how the series
is built.

The alternative is index levels from MoSPI's e-Sankhyiki portal. That was
investigated. It requires:

1. A second download, because e-Sankhyiki serves the 2024-base and
   2012-base series separately. The 2024-base download contains eighteen
   observations of All-India Combined CPI.
2. Chain-splicing the two bases. IIP was rebased to 2022-23 and CPI to
   2024, so both carry a level discontinuity.
3. Month-on-month log differencing.
4. Seasonal adjustment. Indian monthly CPI and IIP are strongly seasonal
   through harvests, festivals and the monsoon, which is precisely why RBI
   publishes the year-on-year rate.

## Decision

Estimate the Indian VAR on the year-on-year series from RBI, and state the
autocorrelation limitation rather than concealing it or working around it
with a specification nobody can defend.

## Consequences

**The specification gate will report these models as not usable.** That is
correct behaviour and it is left in place. A result derived from them is
marked provisional, in the same way results depending on unsourced
calibration targets are.

**The limitation is explicable.** "Year-on-year construction induces
residual autocorrelation through overlapping windows" is a statement that
can be defended and quantified. The alternative path would introduce
splicing and seasonal adjustment judgements that are also caveats, and
less well understood ones.

**The metadata layer was vindicated.** IIP and CPI arrive declared as
PERCENT_CHANGE, so the preparation layer left them undifferenced while
differencing the repo rate. Nothing in the pipeline specified that; it
followed from the unit recorded at ingestion. A pipeline without that
metadata would have differenced an already-differenced series and produced
a plausible, wrong answer.

**Not foreclosed.** The MoSPI long-format file is retained in the raw
archive. It carries state-level and division-level detail no other source
provides, and it is the right shape for a filter-based adapter when the
splicing work is done.