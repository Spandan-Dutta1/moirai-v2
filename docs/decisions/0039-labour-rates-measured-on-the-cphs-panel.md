# 39. Labour rates are measured on the CPHS panel, and the panel stays out of the repository

Status: accepted, 15 November 2026

## Context

CMIE's Consumer Pyramids Household Survey became available through an
institutional licence on 15 November. It is the panel the household layer has
always needed and never had: 1.42 million people, surveyed three times a year
since 2014, each person identified across waves.

Nineteen waves of People of India, May 2019 to August 2025, were loaded. Three
of the model's parameters are rates this panel measures directly, and all three
were numbers with nothing behind them - `job_finding_rate` and
`out_of_labour_force_rate` are both among the 45 undeclared parameters ADR 036
reports.

## The licence comes first

`moirai-v2` is public. CPHS is licensed to an institution and a named user.
Committing any of it - a wave, an extract, a sample, a fixture with real rows -
breaches that licence, and cannot be undone: git keeps the history and GitHub
keeps the forks.

So the data lives at `MOIRAI_DATA`, outside the repository, and
`scripts/check_cphs_labour.py` refuses to run without that path rather than
defaulting to somewhere convenient. `.gitignore` was rejected as the control:
`git add -f` walks past it, and a file added before the pattern existed stays
tracked. `tests/unit/test_licensed_data_stays_out.py` is the control that
fails - it scans the working tree for CMIE filename patterns, refuses any
tracked data file over 5 MB outside the two allowed directories, and asserts
its own patterns still match real CMIE filenames so the guard cannot rot into
matching nothing.

What crosses into the repository is what this record states: rates, sample
sizes and a method. Those are findings, not licensed data.

## Method

The panel is cut to its 19 relevant columns and to responding household
members, which turns 16 GB of CSV into 51 MB. Status follows CPHS's own
definitions: the labour force is the employed plus those unemployed and
actively looking, and people unemployed and not looking sit outside it.

Waves are four months apart, so a transition between consecutive waves is a
four-month probability, converted as `1 - (1 - p)^(1/4)`. That assumes a
constant hazard inside the window; exits are usually front-loaded, so every
monthly rate here is a **lower bound**.

The four pairs touching 2020 are reported but excluded from the figures the
model uses. Accepted members fall from about 575,000 per wave in 2019 to
297,668 in May-August 2020 - a 48 per cent collapse - so lockdown behaviour
and survey attrition cannot be told apart in them. Including them raises the
job finding rate from 0.0293 to 0.0397, which is itself a measure of how much
that window distorts.

## Result

| Quantity | Model | CPHS | |
|---|---|---|---|
| Monthly job finding rate | 0.1800 | **0.0293** | 6.1x too high |
| Monthly job loss rate, steady state | not a parameter | 0.0016 | |
| Labour force participation, 15-64 | 0.650 | **0.423** | 1.5x too high |
| Unemployment rate | 0.078 | 0.071 to 0.085 | agrees |

14 usable transition pairs, 163,000 to 376,000 matched members in each.

**`job_finding_rate` moves from 0.18 to 0.0293.** The model was refilling an
unemployment spell six times faster than Indian data does, which means an
employment shock died out in the model long before it does in life. The
headline is barely touched: the consumption gap at twelve months moves from
-1.5579pp to -1.5880pp, about two per cent, because the gap is a differential
between floating-rate borrowers and debt-free households and the employment
channel enters both arms. The 0.18 to 0.56pp claim stands.

**The unemployment rate agrees and participation does not.** 7.8 per cent
against a measured 7.1 to 8.5 is a good match. 65 per cent participation
against 42.3 is not, and it is **deliberately not corrected here**. CPHS and
the PLFS disagree about Indian participation by more than twenty points; the
model currently sits on the PLFS side of that gap without ever saying so.
Choosing between two credible national surveys decides who sits in the
denominator of every distributional result this model produces, and that
deserves its own record with both measures stated, not a parameter edit
inside one about something else.

## Two silent failures this exposed

Both produced plausible wrong numbers rather than errors, which is the failure
mode this project keeps meeting.

**CPHS renamed its employment-arrangement categories in May 2022**, splitting
digital platform work out and turning `Salaried - Temporary` into
`Salaried - Temporary, not on digital platform` plus
`Digital platform based salaried temporary`. The first version of this analysis
matched the old names, so the salaried share appeared to fall from 20.5 to 11.9
per cent in one wave - a labour-market story that was entirely a broken filter.
`arrangement_group` now classifies on substring and **raises** on any category
it does not recognise, so the next rename stops the run instead of shrinking a
population.

**CPHS dropped its unprefixed weight columns in May 2023**, keeping only the
responding-member weights. A build that kept using `GE15_MEM_WGT_W` would have
silently produced unweighted figures for the last five waves. Everything uses
`R_GE15_MEM_WGT_W`, which exists in every wave and is the correct weight for a
responding subset anyway.

The schema also lost the `EMPLOYMENT_STATUS_SINCE_*` duration fields after
January-April 2023, so unemployment-duration work is limited to waves before
that.

## Consequences

- One parameter moves from undeclared to SOURCED, measured on Indian panel
  data. Two more are now measured, with one of them deliberately left alone
  and its disagreement documented.
- `check_behaviour`, `check_households`, `check_calibration` and
  `check_mpc_sensitivity` will print different numbers through
  `job_finding_rate`. Those are the expected differences in the comparison run.
- A second finding in the same direction as ADR 037. That record found the
  household layer's implied MPC decaying faster than the published annual
  estimate; this one finds unemployment spells ending six times too quickly.
  Independent checks, same conclusion: **the model under-persists shocks.**
  Neither is fatal and both are now named, but the pattern is worth holding
  on to, because a stress engine that recovers too fast understates exactly
  what a stress test is for.
- Stage 5 is no longer blocked on data access. The next thing it should do is
  estimate the income replacement share in ADR 038 directly, which needs the
  Income Pyramids waves joined to these.
