# 37. The model is held against published estimates, and the comparison found a defect

Status: accepted, 14 November 2026

## Context

Every check in this repository so far asks whether the engine computes what it
says it computes. Forty-five scripts recompute a result and print it, 1,547
tests fix the behaviour, and 36 decision records say why each choice was made.
None of that answers the question a model validator asks last and cares about
most: is the number the right *size*?

That question cannot be answered from inside the repository. It needs estimates
produced by other people, on other data, by other methods. The standard name
for the exercise is external benchmarking, and it is the one part of validation
a project cannot do for itself.

It also has to be done honestly, which means refusing the comparisons that do
not hold. The IMF's often-quoted figure for India - 51 basis points of a Fed
move passing through, 46 of them impairing monetary autonomy - is measured on
the three-to-six-month government bond yield. This model's spillover matrix
produces a *policy rate* response. A bond yield carries term premia, risk
premia and expectations of future policy; a policy rate carries none of them.
Printing 12.8 against 51 as though one failed the other would be a fabricated
test.

## Decision

`scripts/check_benchmarks.py` runs each link of the chain at a standard
hundred-basis-point shock and prints the model's answer beside the published
estimate, with the source and a verdict.

1. **Five links are compared.** The Fed-to-RBI policy response; pass-through
   into the lending rate on the floating book and on the whole book; the
   consumption response of floating-rate borrowers; and the marginal propensity
   to consume out of interest expenses.
2. **A verdict names its position against the range**, not against the
   benchmark: `inside the range`, `below the range`, `above the range`. The
   consumption benchmark is negative, and "above" on a -1.95 benchmark would
   otherwise read as a larger response when it means a smaller one.
3. **A link that is not like-for-like is declared so and given no verdict.**
   The Fed-to-RBI row carries `comparable=False` and prints
   `NOT LIKE-FOR-LIKE`. A test asserts that exactly one row is declared this
   way, so a future row cannot quietly acquire the exemption.
4. **The household figures are differentials** - floating-rate borrowers
   against households with no debt - because that is what the micro literature
   estimates. Consumption levels drift with accumulated income growth in both
   arms of the counterfactual; the gap between them is the result.
5. **The household run uses `INDIAN_BANKING_SYSTEM`.** Without a banking
   system the behaviour layer falls back to a flat spread and passes the policy
   move through at 100 per cent, which flatters the comparison. The first
   version of this probe made that mistake and reported a 1.93pp gap; with the
   banking layer in place it is 1.54pp.

## Result

| Link | Model | Published | Verdict |
|---|---|---|---|
| Fed +100bp → RBI policy rate | 12.8bp | 51bp into the 3–6m yield | not like-for-like |
| RBI +100bp → lending rate, floating, 12m | 0.80 | 0.78 on fresh loans | inside |
| RBI +100bp → lending rate, whole book, 12m | 0.51 | 0.44 on outstanding loans | inside |
| RBI +100bp → consumption, floating vs no debt | −1.56pp | −1.95% | inside |
| MPC out of interest expenses, on impact | 0.34 | 0.2 at low/medium, 0.5 at high leverage | inside |

Sources: RBI, weighted average lending rates, 2022–24 (+194bp on fresh and
+111bp on outstanding rupee loans per 250bp of repo); Flodén, Kilström,
Sigurdsson and Vestman (2017), Riksbank Working Paper 342, roughly 300,000
Swedish households from the LINDA panel, 2000–2007; IMF Working Paper 16/195,
43 countries, 2000–2015.

The chain also multiplies correctly. The RBI's best response to the Fed is
0.1279, the consumption gap per 100bp of RBI is 1.54pp, and the product is
0.20pp — inside the 0.18 to 0.56pp headline this project published from the
static and dynamic networks respectively. Three layers built separately agree
when composed, which is not guaranteed by anything in their construction.

## The inconsistency this exposed

The household layer passes 0.80 of a repo move into the lending rate **on
impact**. The loan layer reaches that same 0.80 only after about a year: 0.56
at three months, 0.80 at twelve, because a benchmark-linked loan resets
quarterly and a tenure-first reset spreads the burden further still.

The two layers agree on how far a policy move travels and disagree on how fast.
One applies the whole reset at once; the other spreads it over four quarters.
The household layer has no reset lag.

The symptom is milder than that description suggests, and the first draft of
this record overstated it. The implied MPC starts at 0.34 and still holds 0.29
by month twelve — 85 per cent of its impact value. Against Flodén et al.'s
annual 0.2 at low and medium leverage and 0.5 at high, with this population's
median DTI at 1.8, that is the right order and somewhat strong. The response
decays, but not nearly fast enough to call the model front-loaded in any way
that would mislead a user of it.

The draft figures were 0.19 decaying to 0.06, taken from a partial working copy
of the household layer rather than from the repository. They were wrong, and
the correction is recorded here rather than silently applied, because the
difference is the whole weight of the finding: a model whose cash-flow effect
vanishes within a year would be unusable for provisioning, and this one's does
not.

Two tests pin the direction — `mpc_impact > mpc_12m` and
`|gap_impact| > |gap_12m|` — so a fix has to change them deliberately rather
than drift past. They assert the ordering, not the magnitudes, which are
printed from live values for exactly the reason above.

## Consequences

- One script, 20 tests, no change to any existing output.
- The repository can now state, with sources, that four of its five testable
  links sit inside the published range, and that the fifth is not testable
  against the figure usually quoted at it.
- A new open finding: the household layer needs the loan layer's reset lag, or
  its own. The cleanest route is to let the household layer take its debt
  service from the loan tape's repricing schedule instead of recomputing it
  from a flat maturity, which would also retire
  `BehaviourParameters.debt_maturity_years` - one of the 45 undeclared
  parameters from ADR 036. It is an inconsistency to repair, not a result to
  withdraw.
- A procedural consequence. Household figures quoted in this project come from
  the repository or they are not quoted. A partial working copy produced an MPC
  five times too small and nearly the right consumption gap, which is the worst
  kind of wrong: plausible on the number being watched and badly off on the one
  that was not.
- The benchmark figures are hard-coded with their sources in the row. They are
  published estimates and do not change; if a figure is superseded, the row
  changes and this record is amended.
