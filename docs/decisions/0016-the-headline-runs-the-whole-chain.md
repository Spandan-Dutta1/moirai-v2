# 16. The headline runs the whole chain

Date: 2026-09-30
Status: Accepted

## Context

The project exists to trace one chain: the Fed moves, the other central
banks respond strategically, the response passes through Indian commercial
banks, and it lands on particular kinds of Indian household.

The pipeline's headline did not run that chain. `run_pipeline.py` took the
Fed's own rate path from the US VAR, scaled it to two standard deviations,
and delivered it straight to Indian households:

    SHOCK_SCALE = 2.0
    path = build_shock_path(irf, "fedfunds_shock", monetary_mappings(...), scale=SHOCK_SCALE)

Two links were missing. There was no central bank game, so the size of the
move was chosen rather than solved; and there was no RBI, so Indian
households faced the Fed's rate. The project's own `policy_shock.py`
describes the two standard deviations as "chosen because it produced a
visible response. That is a hypothetical, not a scenario."

The full chain already existed. ADR 010 built it as the scenario
`imported_tightening`: US inflation conditions the Fed, `network_nash`
solves all five banks, and the shock is the RBI's move measured against the
same game without the US condition, so only the part the Fed caused reaches
households. It was reported in `run_scenarios.py` as one row of five, while
the pipeline headline, the number a reader meets first, stayed on the
shortcut.

## Decision

`scenarios.py` declares `HEADLINE_SCENARIO = IMPORTED_TIGHTENING`, with the
reason recorded next to it.

`run_pipeline.py` keeps Layers 0 and 1 unchanged and replaces everything
after them with the chain, run through `run_scenario` so the pipeline and
`run_scenarios.py` cannot disagree about it:

    LAYER 1a  the central bank game: every bank's move, unprompted and caused
    SEAM      the RBI's caused move as a multiple of the estimated shock
    LAYER 2   what Indian banks pass on
    LAYER 3   the households
    RESULT    who in India bears a US tightening

The headline runs without `allow_extreme`. A solved shock beyond ten
standard deviations would stop the pipeline rather than be extrapolated.

The former headline is kept, after the result, under "FOR COMPARISON: the
Fed's own path, delivered straight to households", labelled as an upper
bound and not a scenario. Nothing is deleted.

Six tests in `tests/unit/test_scenarios.py` pin what the headline must be:
a declared scenario, originating with the RBI, conditioning only the Fed,
declaring its departures from the US transmission, measuring the shock as
the RBI's caused move, and needing no extreme extrapolation.

## What moved

Only `run_pipeline.py`'s output. Its headline now equals the
`imported_tightening` row of `run_scenarios.py`, which is the check that the
two are wired identically. On the August 2026 conditions and the 1985-2007
estimation:

| | Former headline (Fed direct, 2 SD) | Chain |
|---|---|---|
| Shock | 2.0 SD, chosen | 0.3 SD, the RBI's caused +13bp |
| Borrowers face | the Fed's path | +10bp; savers +4bp |
| Aggregate consumption | -0.426% | -0.066% |
| Floating-rate borrowers | -2.8% | -0.59% |
| Net savers | -0.2% | -0.03% |
| Spread | -2.63pp | -0.56pp |

The finding keeps its shape: floating-rate borrowers bear most of it and
the aggregate hides a transfer. Its size is about a fifth of what was
reported, because the RBI imports only a small part of the Fed's move.

## What did not move

Every other script. Layers 0 and 1 of the pipeline print exactly what they
did. No engine behaviour changed; `HEADLINE_SCENARIO` is a name for an
existing scenario.

## Caveats the headline carries

- The shape of the path is the US impulse response, because the Indian VAR
  is not credibly identified (ADRs 007, 010). Inflation is held at baseline
  and the horizon capped at 30 months, both declared on the scenario.
- The RBI's caused move is small, +13bp, and so is the shock. A small
  headline is the honest one; the direct version remains as the upper
  bound.
- The aggregate's sign depends on deposit pass-through, which failed its
  held-out test (ADR 009). The spread is the robust number.

## Noticed, not changed here

`shock_from_equilibrium` computes an impact response from the first period
and then overwrites it with the cumulative peak before using it. The first
assignment is dead code. It changes nothing and is left for a tidy-up.
