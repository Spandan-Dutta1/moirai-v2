# 8. Parameter provenance audit

Date: 2026-08-14
Status: Accepted

## Context

The project now spans four layers and roughly fifty parameters. Some are
estimated from data, some taken from published statistics, and some
chosen by judgement. Without an audit it is not possible to say which
results rest on evidence and which on assumption, and a reader cannot tell
either.

This document is that audit. It is deliberately unflattering.

## Sourced or estimated

| Quantity | Source |
|---|---|
| All macro series | FRED, World Bank, RBI, fetched live and content-hashed |
| VAR coefficients, impulse responses | Estimated, specification-tested |
| Household indebtedness, 22.4% | AIDIS 2019, urban incidence |
| Employment rate, 49.4% | PLFS 2023-24, urban WPR aged 15+ |
| Bank transmission by group | RBI Bulletin, Feb 2025 to May 2026 |
| Central bank mandates and targets | Statutes and published strategies |
| Fed output weight, 0.40 | Fitted to observed policy decisions |

Around fifteen quantities.

## Assumed

**Household calibration, four of seven targets unsourced:** wealth Gini,
income Gini, hand-to-mouth share, indebtedness gradient. The hand-to-mouth
share is the most consequential and the least grounded; no Indian estimate
in the Kaplan-Violante-Weidner sense was located, and AIDIS may not support
the liquid-illiquid decomposition it requires.

**Behavioural parameters, all assumed:** the two MPC endpoints, the Okun
coefficient, the job loss income gradient, the job finding rate, the
unemployment replacement rate, the expectation learning rate, and the
average debt maturity. Several are literature-plausible; none was taken
from a specific paper.

**Banking layer:** individual bank characteristics are plausible rather
than looked up. Asset sizes are order of magnitude.

**Network:** every tier multiplier, the base spillover magnitudes, and the
tier assignments. The assignments are defensible from the global financial
cycle literature; the numbers are not from it.

**Central bank preference weights:** every one except the Fed's output
weight. The RBI's external weight has a documented objective behind it but
an assumed magnitude.

Around thirty five to forty quantities.

## The circularity that matters

Two parameters were tuned until the model reproduced a target, and then
the model's agreement with that target was reported as a result. That is
fitting to the target, not validating against it.

**`log_wealth_mean`**, set to 10.0 by grid search over the household
moment set. The search is legitimate as a calibration procedure, but the
resulting hand-to-mouth share of 0.299 against a target of 0.40 is not
independent evidence, because the parameter was chosen to move it there.
The target itself is unsourced, so this is a parameter fitted to a guess.

**The banking pass-through coefficients**, the 0.75 and 0.45
discretionary bases and the stress multipliers. These were adjusted until
the group aggregates landed near the RBI's published figures. Five of six
group checks pass, and that agreement is by construction. The RBI figures
are real; the model's reproduction of them is not a test.

## What would make these validations rather than fits

**Out-of-sample.** Calibrate the banking layer on the easing cycle and
check it against a tightening cycle, which the RBI also publishes. The
asymmetry is a genuine prediction and the model currently has no chance to
fail it.

**Held-out moments.** Fit `log_wealth_mean` on a subset of household
moments and check the remainder. With seven targets and effectively one
free parameter this is feasible.

**Source the four unsourced targets**, or establish that they cannot be
sourced. Two probably can be with more searching; the hand-to-mouth share
may genuinely not exist for India, in which case saying so is the result.

## Decision

Record the audit rather than acting on it immediately. Out-of-sample
validation of the banking layer is the highest-value next step and is
scheduled; the rest is noted.

No result in this project should be described as validated. The
specification tests are real, the diagnostics are real, and the data is
real. The behavioural and preference parameters are chosen, and the two
cases above are chosen to fit.

## Consequences

**The distributional result is a mechanism demonstration, not a
measurement.** The 2.6 percentage point spread between floating-rate
borrowers and savers follows from the model's structure. Its magnitude
depends on parameters nobody has estimated.

**Layers 0 and 1 are on much firmer ground than layers 2 and 3.** That
asymmetry is inherent: macro series are published and behavioural
parameters are not. It should not be smoothed over by presenting the
whole chain at one confidence level.

**Anyone reading a result should be able to see its provenance.** The
confidence flags, the sensitivity analysis and this document exist so that
the answer to "how much of this did you make up" is available rather than
awkward.