# 32. Historical validations read their FRED indices as of a pinned ALFRED vintage

Status: accepted, 8 November 2026

## Context

`validate_2022.py` (ADR 012) declares each bank's 2022 inflation condition.
`historical_validation.py` then recomputes every condition from the data and
refuses to run if a recomputed value differs from the declared one by more
than 0.001 points. The FRED indices were read as FRED publishes them today.

At the end of September 2026, FRED's annual revision of PCE moved the 2022
mean of year-on-year PCE inflation from 6.545% to 6.562%. From then on,
`validate_2022` refused to run (`historical_2022 disagrees with the data`)
with no change to the code. The guard worked. The design was wrong: a
validation of a past year read a moving source.

## Evidence

`probe_pce_vintages.py` (not committed) computed the 2022 mean year-on-year
PCE inflation as ALFRED held it on 17 dates:

| Vintage | 2022 mean y/y |
|---|---|
| 1 Jan 2023 | December 2022 not yet published |
| Apr to Jul 2023 | 6.266% |
| Oct 2023 to Jul 2024 | 6.464% |
| Oct 2024 to Jul 2025 | 6.555% |
| **Oct 2025 to Jul 2026** | **6.545%** (the declared value) |
| Oct 2026 to Nov 2026 | 6.562% |

Each autumn revision moved the number. Across the four revisions it moved
by 0.3 points in total, and by 0.2 points since the first one. The
declared value was correct for the vintages from October 2025 to July 2026.

I predicted a match in a 2024 or 2025 vintage. That was wrong: the match is
one year later. It is recorded as wrong.

## Decision

1. `FredIndex` takes an optional `vintage`. `fetch` and `average_inflation`
   pass it to ALFRED as a one-day real-time window (`realtime_start =
   realtime_end = vintage`). Without a vintage nothing changes.
2. `validate_2022` reads all three FRED price indices (PCE, euro-area HICP,
   UK CPI) as published on **1 July 2026**, inside the window the declared
   conditions came from. The run prints "as published on 2026-07-01" under
   each one.
3. Policy rates are not pinned. They are not revised.

## Result

- `validate_2022` runs to the end again:
  - Fed 6.54%, ECB 8.36%, BoE 7.90%, all at the vintage. The declared
    conditions match.
  - The model results are those of ADR 012: caused moves of Fed +262bp
    and RBI +188bp; RBI-to-Fed ratio 57.7% against 52.9% observed; RBI
    best-response slope +0.128.
- Tests: 5 in `test_historical_vintage.py`, written first. Before the change:
  4 failed, 1 passed. After: 5 passed. The suite is 1,453.
- The comparison run against main showed two differences, both predicted:
  - `_pytest`;
  - `validate_2022`, which exits 1 ("disagrees") on main and 0 on this
    branch.

## Consequences

- No reported number changes. What changes is that the 2022 validation can
  no longer be broken by a FRED revision.
- To move to a later vintage is now a decision: change `VINTAGE`,
  re-declare the conditions, and record both in an ADR.
- `validate_2013` still reads the current vintage. It passes today, but the
  same thing could happen to it. Pin it the same way when it is next
  touched, or at once if it ever refuses.
- The Stage 4 backtest needs this exact mechanism: the model must see the
  data only as they stood in April 2022.
- This closes open finding 10.
