# 38. Income when nobody is working is one sourced number, used in both layers

Status: accepted, 14 November 2026

## Context

The roadmap carried this as step 1.D1: "Unemployed income: 25% at generation vs
35% in behaviour." Two layers disagreed about what a household earns once
nobody in it is working. Opening the code found two further problems behind the
one that was recorded.

**The generation values were not parameters at all.** `generate_population`
ended with three bare literals:

```python
income[employment == EmploymentStatus.RETIRED] *= 0.45
income[employment == EmploymentStatus.UNEMPLOYED] *= 0.25
income[employment == EmploymentStatus.OUT_OF_LABOUR_FORCE] *= 0.30
```

Three numbers that shape every household result, invisible to the
documentation generator because it reads the fields of parameter classes. They
were not among the 45 undeclared parameters ADR 036 reports — they were below
the floor that inventory can see. The 45 was an undercount, and this is how.

**The transition was asymmetric.** The behaviour layer did this:

```python
updated.income[lost_job] *= parameters.unemployment_income_replacement
updated.income[found_job] /= max(parameters.unemployment_income_replacement, 1e-6)
```

Multiplying on the way out of work and dividing on the way back in cancels only
if both layers use the same figure *and* the household began employed. A
household generated unemployed already had 0.25 applied to it; dividing by 0.35
when it found work left it on 0.25 / 0.35 = 0.71 of an otherwise identical
household that never lost a job — permanently, since nothing restores the
level afterwards. The recorded defect was a disagreement about a number. The
actual defect was that the disagreement leaked into income *levels* for a
sub-population, through every subsequent period.

## The source

[Coping with labor market risk: formal job loss and earnings dynamics in
India](https://www.sciencedirect.com/science/article/abs/pii/S0264999325002251),
*Economic Modelling* (2025), estimated on CMIE's Consumer Pyramids Household
Survey panel — the same panel Stage 5 is built around:

- earnings fall **43 per cent** on impact after formal job loss
- they remain more than **30 per cent** below expected levels three years later
- household income stays more than **20 points** below expected at three years

So the share of prior earnings a household retains is **0.57**. Both values
this replaces overstated the loss, 0.25 badly: it asserts that losing a job
costs three quarters of household income against a measured 43 per cent. India
has no general unemployment insurance — the ESIC scheme reaches only insured
formal workers — so there is no statutory replacement rate to use instead, and
a measured one is the right kind of number here.

The on-impact earnings figure is used rather than the household-income figure.
This model carries one income and one employment status per household, so there
is no second earner for a household-level measure to average over: the
household's income *is* its earner's earnings, and the earner's own drop is the
consistent quantity. That is a judgement, and it is the one place this record
is open to challenge. The paper's household-income path implies a gentler
share, nearer 0.8, for a model with multiple earners per household. If the
household layer ever gains a second earner, this choice has to be revisited.

Pension and out-of-labour-force replacement rates could not be sourced. The
OECD's *Pensions at a Glance* pages served navigation rather than their data
tables, and the India country note refuses retrieval. Rather than substitute a
worse-grounded number, both keep their prior values and are declared **ASSUMED**
with that reason stated in the field. That is the outcome ADR 036 was built to
force: a value with no source has to say so out loud.

## Decision

1. **One object, `NonEmploymentIncomeParameters`**, holding the three shares,
   taken by both `PopulationParameters` and `BehaviourParameters`. Generation
   and simulation cannot disagree again, and a test asserts the two defaults
   are equal.
2. **Income is derived from status, not adjusted on the way in and out of it.**

   ```python
   shares = parameters.non_employment_income.shares
   updated.income = (
       population.income * growth
       * shares(updated.employment) / shares(population.employment)
   )
   ```

   Scaling by the ratio of the two statuses' shares is correct for every
   transition, including the one the old code got wrong, and it handles
   statuses the old code did not touch at all (share / share is one).
3. **Shares are strictly positive**, enforced at construction. A share of
   exactly zero makes a household that leaves that status unrecoverable: its
   income is already zero and the ratio that would restore it is undefined.
   Refusing it on construction replaces the `max(..., 1e-6)` that was
   previously papering over the same division at runtime.
4. **The three literals become declared fields** with provenance, so the
   documentation generator can see them.

## Result

The headline does not move:

| | Before | After |
|---|---|---|
| Consumption gap, floating vs no debt, 12m | −1.5573pp | −1.5579pp |
| Consumption gap, on impact | −1.7493pp | −1.7492pp |
| Implied MPC, on impact | 0.3447 | 0.3451 |
| Implied MPC, 12 months | 0.2856 | 0.2869 |
| Lending pass-through, on impact | 0.8026 | 0.8026 |

Six ten-thousandths of a percentage point on the headline. That is not an
accident of calibration: the gap is a *differential* between floating-rate
borrowers and debt-free households, measured against a baseline run of the
same population, so a change in the income of the unemployed moves both arms
of the counterfactual almost identically and cancels in the difference. The
0.18 to 0.56pp claim stands unchanged, and now rests on a sourced number
instead of two inconsistent ones.

What does change is the level: unemployed households' mean income rises from
roughly 0.25 to 0.57 of their employed counterparts' — in a 200,000-household
draw, a mean of about 205,000 against 361,000 for the employed. Any result
*about the unemployed* changes, which is the right consequence of correcting a
number that was wrong by more than a factor of two.

16 tests. The one worth naming is
`test_a_household_generated_unemployed_reaches_the_employed_level`: a household
whose recorded income already reflects the unemployed share must land exactly on
the employed level when it finds work, which is the case that used to settle at
71 per cent of it.

## Consequences

- The declared-parameter inventory changes:
  `unemployment_income_replacement` goes (it was one of the 45 undeclared),
  three new declared shares arrive with provenance, and two nested-group
  fields are added. Expect 74 declared parameters and 44 undeclared.
- Nothing stops a caller passing a `BehaviourParameters` whose shares differ
  from the population's. That is deliberate — studying a population under a
  different replacement rate is a legitimate counterfactual — but it means the
  guarantee is "the defaults agree", not "they cannot differ". The test asserts
  the defaults.
- `check_behaviour`, `check_households`, `check_calibration` and
  `check_mpc_sensitivity` will print different numbers, all through the
  unemployed share. Those are the expected differences in the comparison run.
- The correction makes the model's households *less* fragile. It weakens the
  severity of a channel the project leads with, which is what sourcing a value
  properly is supposed to be allowed to do.
- With CPHS access now available, this share can be estimated directly on the
  same panel the published study used, rather than read off its abstract. That
  is the first thing Stage 5 should do.
