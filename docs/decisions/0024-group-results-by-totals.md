# 24. Group results reported by totals, beside the original measure

Date: 2026-10-01
Status: Accepted

## Context

Every group result in the project, the change in consumption for
floating-rate borrowers, fixed-rate borrowers and net savers, has been the
average of each member household's own proportional change, with the
denominator floored at one rupee:

    change = (shocked - baseline) / np.maximum(baseline, 1.0)
    borrower_change = change[population.is_rate_exposed].mean()

The dynamic game (ADR 023) sent the household layer an easing path for the
first time; every earlier scenario was a tightening. A rate path roughly
neutral on balance produced a 0.54 percent gain for floating-rate
borrowers. Simple constant paths isolated it:

| 30 months at | Floating borrowers, average of household changes |
|---|---|
| +13bp | -0.60% |
| -13bp | +12.9% |

A 13bp cut cannot raise consumption by 13 percent, twenty times the effect
of the matching rise. The cause is two things together:

- 1,442 households in the 200,000 consume nothing in every period: their
  debt service exceeds their income and disposable income is floored at
  zero. Most are floating-rate borrowers.
- A household near that floor whose consumption rises from about one rupee
  to a few hundred posts a change of thousands of percent, and the average
  is dominated by it. A tightening leaves such households at zero, so in
  every earlier scenario the distortion stayed invisible in sign but
  inflated the size.

## Decision

The original figures are kept, unchanged, so every validated result
reproduces. Beside them, each group's change is also reported as the
change in its total consumption, which has no small denominators and is
the standard way to report a group:

- `behaviour.group_change(baseline, shocked, members)`,
  `behaviour.total_consumption(outcomes)` and
  `behaviour.never_consuming(outcomes)` are new.
- `ScenarioResult` keeps `borrower_change`, `saver_change` and `spread`
  as they were and gains `borrower_change_total`, `saver_change_total`,
  `spread_total`, `households_never_consuming` and each household's total
  baseline and shocked consumption. The ledger records both spreads.
- `run_pipeline.py` prints the new figure under the headline, a
  group-total column in the exposure table, a by-totals column in the
  evidence table, and the count of households consuming nothing.

Six tests cover the new measure, including one that pins the original
figures to their original definition so they cannot be changed by
accident, and one showing a 13bp cut and a 13bp rise have effects of
comparable size under the new measure.

## Result

On the real data, side by side:

| | Original measure | By group totals |
|---|---|---|
| Floating-rate borrowers | -0.650% | -0.235% |
| Fixed-rate borrowers | -0.139% | -0.066% |
| Net savers | -0.043% | -0.056% |
| **Spread** | **-0.61pp** | **-0.18pp** |
| Evidence range (ADR 019) | -0.20 to -0.61 | -0.05 to -0.18 |

Households consuming nothing throughout: 1,442 of 200,000.

The comparison against the main branch showed `run_pipeline` as the only
script whose output changed, every existing line identical.

## Consequences

- The finding keeps its direction under both measures: floating-rate
  borrowers lose more than savers. Its size is about a third under the
  measure that households at the zero floor cannot distort.
- The zero-consumption households are themselves a modelling problem,
  addressed in ADR 025.
