# 40. The income share is measured on CPHS, and ADR 038 chose the wrong concept

Status: accepted, 15 November 2026. Supersedes the value set in ADR 038.

## Context

ADR 038 set the unemployed income share to 0.57, from a published abstract
reporting a 43 per cent drop in individual earnings after formal job loss in
India. It argued for the individual figure over the paper's gentler
household one on this reasoning:

> The on-impact earnings figure is used rather than the household-income
> figure. This model carries one income and one employment status per
> household, so there is no second earner for a household-level measure to
> average over.

That record also named the argument as the one place it was open to
challenge. CPHS access arrived the same day, and the challenge can now be
settled with the panel the paper itself used.

## Method

Treatment is a household with **exactly one employed member** in one wave
whose member is unemployed and looking in the next - the closest real analogue
to a Moirai household, which has one income and one employment status. Control
is a single-earner household whose member is employed at both interviews.
Comparing a job-loser against its own past would book ordinary nominal income
growth, 8 to 12 per cent a year through this period, as part of the job loss.

Shares are ratios of group totals rather than means of household ratios,
which is the convention `behaviour.group_change` already sets in this project
and for the same reason: a household whose income falls to near nothing posts
a ratio that swamps an average.

**The event runs backwards in this data, and finding that took three
attempts.** CPHS asks each interview about the preceding four months, so the
income of an unemployment spell is attributed to the months *leading up to*
the interview that detects it. The first two estimates measured the months
after and returned 0.991 and 1.011 - apparently no income loss at all. Both
were reported before being understood, and both were wrong in a way no error
message would have caught. What exposed it was plotting wage income month by
month around the event and seeing the trough sitting squarely before month
zero, with recovery after.

Windows are reported two ways, because a single figure would hide whether the
answer depends on a judgement call. It does not materially: the two differ by
about one point.

## Result

| Measure | wide (-8..-5 vs -4..-1) | deep (-6,-5 vs -2,-1) |
|---|---|---|
| **Total household income** | **0.833** | 0.822 |
| Wage income only | 0.598 | 0.572 |

1,615 job losses against 784,784 single-earner controls, 2021 to 2026,
excluding every pair touching 2020.

**The wage row is the validation.** The published study reports a 43 per cent
drop in earnings, a share of 0.57. This estimate, built from raw waves with
its own sample definition, control group and windows, and never fitted to that
figure, returns 0.572 on the narrow window and 0.598 on the wide one.
Reproducing a published result the method was not aimed at is the strongest
evidence available that the join, the windows and the controls are right - and
it is what makes the household figure beside it credible.

**Both numbers are correct, for different quantities.** 0.57 is what the
earner loses. 0.83 is what the household keeps, and the gap between them is
pension, rent, business income and transfers carrying on while wages stop.

The model scales **total household income**, so 0.83 is the figure it needs.
ADR 038's reasoning was wrong: having one earner does not make a household's
income equal to that earner's wages, and this panel measures the difference
directly in exactly the households the argument was about.

`UNEMPLOYED_EARNINGS_SHARE_INDIA` moves from 0.57 to 0.83. The consumption gap
at twelve months moves from -1.5880pp to -1.5983pp, one hundredth of a point,
for the same reason every household correction has been small: the gap is a
differential measured against a baseline run of the same population.

## What this parameter's history now shows

| | Value | Basis |
|---|---|---|
| Before ADR 038 | 0.25 / 0.35 | two bare literals that disagreed |
| ADR 038 | 0.57 | a published abstract, wrong concept |
| ADR 040 | **0.83** | measured, right concept |

Three moves, every one toward less severity. A job loss now costs a household
about a sixth of its income where the original code claimed three quarters.

That direction deserves stating rather than burying. It is the third finding
in two days pointing the same way: ADR 037 found the implied MPC decaying
faster than the published annual estimate, ADR 039 found unemployment spells
ending 6.1 times too quickly, and this record finds the income loss from
unemployment overstated by a factor of four and a half. **The household layer
was tuned pessimistically in ways nobody had checked, and every parameter that
has since met real Indian data has moved toward less severe.**

None of it changes the headline, because the headline is a differential. But a
reader is entitled to ask whether the next parameter will move too, and the
honest answer is that it might. The defence is not the numbers; it is that the
method, the sample, the window sensitivity and the three failed attempts are
all written down.

## Consequences

- `scripts/check_cphs_replacement.py` recomputes both measures over both
  windows from `MOIRAI_DATA`, outside the repository. No licensed data is
  committed; the shares, sample sizes and method are.
- `check_behaviour`, `check_households`, `check_calibration` and
  `check_mpc_sensitivity` will print different numbers. Those are the expected
  differences in the comparison run.
- ADR 038 stands as the record of the inconsistency it fixed - one object
  instead of two numbers, income derived from status rather than applied and
  reversed, shares that cannot be zero. Only its *value* is superseded, and
  its own stated doubt is what led here.
- Still unestimated, and now clearly worth doing: whether the share differs by
  whether the household carries debt. A borrower who loses work is the case
  this whole model is about, and the panel can answer it.
