# 5. Calibration targets carry their provenance

Date: 2026-08-12
Status: Accepted

## Context

The household population has around fifteen parameters governing income,
wealth, debt and employment. Early values were chosen by judgement, and at
least one was chosen because it moved a distributional result in the
direction expected. That is fitting a parameter to a conclusion.

The consequences were not cosmetic. A wealth level parameter of 12.0
produced a hand-to-mouth share of 0.7 percent against a plausible target
near 40 percent, which flattened the marginal propensity to consume across
the whole population and muted the consumption channel. With that
calibration a monetary tightening *raised* aggregate consumption. With a
fitted value of 10.0 it lowers it, which is the correct sign. The
parameters were determining the sign of the headline result.

## Decision

Every calibration target is declared in code with a source string and a
confidence flag before any fitting takes place. Parameters are then fitted
to the declared targets by grid search rather than chosen.

Confidence takes three values:

- **SOURCED**: the value is taken directly from a named published
  statistic, whose concept and definition match what the model measures
- **DERIVED**: the value is computed by arithmetic on published statistics,
  with the derivation stated
- **UNSOURCED**: nobody has sourced this yet; provisional everywhere

The flag records where the value came from, not how good the source is.
Limitations of a published statistic, such as a working paper rather than
a journal article, a population that differs from the model's (all-India
against urban), imputed inputs or a robustness row, go in the target's
note. They do not demote a SOURCED value to DERIVED. DERIVED is reserved
for values that required arithmetic to obtain. A value becomes SOURCED only
when a named published statistic supplies it; the flag is never raised on
judgement.

The calibration report shows every target with its confidence, including
the ones that were missed.

## Current status

| Target | Value | Confidence | Source |
|---|---|---|---|
| share_indebted | 0.224 | SOURCED | AIDIS 2019, urban incidence of indebtedness |
| employment_rate | 0.494 | SOURCED | PLFS 2023-24, urban WPR age 15+ |
| mean_debt_to_mean_income | 0.50 | DERIVED | AIDIS mean urban debt over an assumed mean income |
| wealth_gini | 0.75 | UNSOURCED | not verified |
| income_gini | 0.50 | UNSOURCED | not verified |
| share_hand_to_mouth | 0.374 | SOURCED | Gupta, Pizzolon and Singh (2025), Table 5, monthly pay period |
| indebtedness_gradient | 7.0 | UNSOURCED | qualitative direction only |

The hand-to-mouth share was originally 0.40 and unsourced, on the belief
that no Indian estimate in the Kaplan, Violante and Weidner sense existed.
One does: Gupta, Pizzolon and Singh (2025, working paper) apply the KVW
method to AIDIS 2019, with income imputed from CPHS. Their monthly pay
period row uses a half-month threshold, which matches this model's
definition, and gives a total share of 0.374. The total rather than the
poor-only share is used because the model has one liquid wealth field.
The estimate covers all of India, not urban households alone. The note on
the target records this and the other limitations. `log_wealth_mean` was
deliberately not re-fitted to the new target (see ADR 008).

Three of seven targets remain unsourced. One of those may not be sourceable
as stated:

- **Indebtedness gradient by income quintile** is not reported by AIDIS,
  which tabulates by asset decile instead. The target encodes a direction,
  not a level.

The two Gini targets are probably findable but need care. Widely cited
Indian Gini figures are usually consumption based and substantially lower
than income based ones, and the two are not interchangeable.

## Consequences

**Gained: the sign of the headline result no longer depends on an
unexamined choice.** The wealth parameter now takes the value that
reproduces a declared target, and if that target is wrong the error is
visible and attributable.

**Gained: a reader can see how much of the calibration rests on real
numbers.** Four unsourced targets out of seven is a weakness, but a stated
weakness is a different thing from a hidden one.

**Gained: grid search reveals which parameters the moments identify.**
Varying the wealth level changes the loss by a factor of five. Varying
initial expected inflation does not change it at all, so its fitted value
should not be quoted as though the data had chosen it.

**Limitation: two AIDIS caveats apply to every wealth target.** AIDIS
measures total assets, dominated by land and buildings, whereas the model's
wealth field is the liquid buffer a household can spend from. And household
surveys understate the top of the wealth distribution, because wealthy
households respond less often and under-report financial assets, so any
survey-derived Gini is a lower bound.

**Any result depending on an unsourced target is provisional and should
say so.**