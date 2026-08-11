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

- **SOURCED**: taken directly from a named published statistic
- **DERIVED**: arithmetic on published statistics, with the derivation stated
- **UNSOURCED**: nobody has sourced this yet; provisional everywhere

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
| share_hand_to_mouth | 0.40 | UNSOURCED | no Indian estimate located |
| indebtedness_gradient | 7.0 | UNSOURCED | qualitative direction only |

Four of seven targets remain unsourced. Two of those may not be sourceable
as stated:

- **Hand-to-mouth share** in the Kaplan, Violante and Weidner sense
  requires separating liquid from illiquid assets. AIDIS reports total
  assets dominated by land and buildings, and may not support the
  decomposition. This is the single most consequential target for the
  aggregate consumption response and the least well grounded.

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