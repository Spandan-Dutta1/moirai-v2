# 36. Model documentation is generated from the repository, not written beside it

Status: accepted, 13 November 2026

## Context

Stage 6 is packaging: turning a working engine into something a reader can
assess. The reader worth designing for is a model validator, and a validator
asks four questions. What does it claim? What did you decide, and why? Where
did every number come from? What evidence do you have?

Three of those four are already in the repository. The decisions are in 35
decision records. The provenance of each value is in the field descriptions,
because every parameter added since ADR 031 has declared itself SOURCED,
CALIBRATED, DERIVED or ASSUMED in its first word. The evidence is the check
scripts, each of which recomputes a result and prints it.

A document written beside the code restates all of that and then drifts from
it. The drift is not hypothetical here: on 8 November two project documents
carried a repository state that was never true, because I asserted a merge I
had not seen.

## Decision

`scripts/model_documentation.py` reads the repository and prints the document.

1. **Decisions** come from `docs/decisions/*.md`: number from the filename,
   title from the heading, status from the status line.
2. **Parameters** come from walking every importable module of the engine and
   collecting the fields of every class whose name ends in `Parameters` or
   `Profile`. Results are excluded, and so is `CsvSpec`, whose fields map a
   publisher's columns rather than set a model value. A module that cannot be
   imported is reported in the output rather than skipped silently.
3. **Provenance** is the description's **first word**, which must be one of
   SOURCED, CALIBRATED, DERIVED, CONTRACTUAL, CONVENTION, ASSUMED, SWITCH (a
   field that is not a model value) or METADATA (a label). Anywhere else the
   word is prose: "the common industry convention" explains an ASSUMED field
   rather than declaring a different kind, and the first implementation got
   that wrong by matching anywhere in the text.
4. **Undeclared fields are reported as gaps**, not passed over, and
   `test_every_parameter_of_the_credit_engine_declares_its_provenance` fails
   when a new one appears in the credit engine.

Two fields of the credit engine were relabelled to make that control pass
honestly rather than by exception: `EclParameters.apply_floors` is a SWITCH,
and `RepricingParameters.income_growth` is a CONVENTION, since a default of
zero is the standard held-constant stress test rather than a claim about
India. `BookProfile`'s fields are SOURCED per profile, with each profile's
`source` naming the figures, and its ticket is marked as period-sourced for
every profile except `april_2022`.

## Result

| | |
|---|---|
| Declared parameters found | 70 |
| With provenance | 25 — 16 sourced, 2 assumed, 1 calibrated, 1 derived, 1 contractual, 1 convention, 1 switch, 2 metadata |
| Undeclared | **45** |
| Credit engine undeclared | **0**, enforced by a test |
| Evidence scripts | 44 |

Every one of the 45 sits outside the credit engine, in the household,
behaviour, spillover and central-bank layers:

- `SpilloverParameters`: `exchange_passthrough`, `demand_spillover`,
  `domestic_inflation_effect`, `domestic_output_effect`, `confidence`, `note`
- `PopulationParameters`: 22 fields, including `income_wealth_correlation`,
  `log_income_mean`, `log_wealth_sd`, `retirement_age`,
  `share_floating_rate`, `risk_aversion_mean`
- `BehaviourParameters`: 10 fields, including `unemployment_income_replacement`,
  `okun_coefficient`, `mpc_low_wealth`, `mpc_high_wealth`,
  `debt_maturity_years`, `expectation_learning_rate`
- the remaining 7 in the central-bank and scenario layers

The document prints all 45 by name. That list is the point of the exercise. "The central banks' preference weights
are chosen rather than estimated" has been the project's standing admission
since ADR 011, stated in prose and impossible to act on. It is now 38 named
fields, and several of them are already open findings under other names:
`unemployment_income_replacement` is the 25-against-35-percent inconsistency,
`debt_maturity_years` is the maturity the loan tape supersedes,
`okun_coefficient` is the frequency question.

## Consequences

- 12 tests, one script, four relabelled descriptions. No existing output
  changes; the labels are inside descriptions, which nothing computes from.
- The document is deterministic apart from its generation date, so the
  comparison tool sees no difference between two runs of the same code.
- The next step is cheap and now well defined: declare the 45. Most will take
  a sentence naming the source already in the code comments, and the ones that
  cannot be sourced will have to say ASSUMED out loud, which is the outcome
  that matters. Until then the gap is in the document rather than in prose.
- A validator reading this document can reach every number's origin without
  reading the code, and can see what the project does not know.
