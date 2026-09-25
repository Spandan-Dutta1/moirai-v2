# CLAUDE.md

Moirai traces a central bank rate decision through to individual households,
in four layers: data fabric (0), policy games and causal VAR (1a/1b),
commercial banks (2), and 200,000 heterogeneous households (3). See
`README.md` for the overview and `docs/decisions/` for the ADRs. Read the
ADRs before changing anything in the engine. Several constraints below
look like bugs but are deliberate.

## Commands

The system Python does not have the dev tools. Use the project venv:

```bash
.venv/Scripts/python.exe -m pytest        # full suite, ~20s
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy src/moirai
python scripts/run_pipeline.py            # US chain, end to end
python scripts/run_pipeline_india.py      # Indian chain, result withheld
python scripts/run_scenarios.py
```

`pyproject.toml` already sets `addopts = "-q"`, so adding `-q` hides the
pass/fail summary line.

## Constraints that must not be broken

**No LLM dependency in the engine (ADR 004).** Nothing under
`src/moirai/engine/` may import or call a model provider (anthropic,
openai, or any LLM client) or require an API key beyond data ingestion.
The engine must run offline and reproduce bit-for-bit. Belief dynamics are
adaptive learning rules, not persona debate. The only network code in the
engine is the FRED adapter's `httpx` client in `data_fabric/ingestion/`.

**The Indian pipeline withholds its distributional result (ADRs 006, 007).**
The Indian VAR is not credibly identified: output rises after a rate hike
in all four specifications tried. `scripts/run_pipeline_india.py` runs the
full chain but deliberately prints no household numbers. Do not make it
print them. Do not add variables or change the specification until the sign
flips, because that is specification searching. The US pipeline is the
credibly identified demonstration.

**The banking layer's out-of-sample failures are pinned (ADR 009).** It was
calibrated on the easing cycle and tested against the held-out tightening
cycle. One of three predictions survives. These tests in
`tests/unit/test_commercial_banks.py` assert the failures:

- `test_the_group_ordering_fails_out_of_sample`
- `test_deposit_pass_through_fails_out_of_sample`
- `test_out_of_sample_performance_is_worse_than_in_sample`

Do not add mechanisms or tune parameters to make them pass. Doing that after
seeing the held-out data would turn the project's only real validation
into a second in-sample fit. **If one of these tests goes green, that is a
regression, not progress.** Stop and flag it. The known remedy for the
deposit failure is a liquidity state. Adding it would be a mechanism change
that needs its own ADR and a new held-out test, not a quiet fix. The
docstring in `test_stress_lowers_pass_through_in_both_directions`
documents an intent-vs-arithmetic mismatch that is left unfixed for the
same reason.

**Fitted-to-target parameters stay caveated (ADR 008).** Two parameters
were tuned until the model reproduced their targets:

- `log_wealth_mean` (10.0, fitted to an unsourced hand-to-mouth target)
- the banking pass-through bases (0.75, 0.45) and stress multipliers

Their agreement with those targets is by construction, not evidence. Do
not remove or soften the caveats in code, reports, the README or the ADRs.
Never describe any result as "validated".

## Conventions

**Frozen, validated-at-construction models.** Pydantic models use
`ConfigDict(frozen=True, extra="forbid")`. Plain dataclasses use
`frozen=True`. Invalid states (an unsorted `TimeSeries`, a NaN in an
`Observation`, config changing mid-run) must fail at construction. Keep new
models the same way.

**Every calibration target carries its provenance (ADR 005).** Each
`Moment` in `src/moirai/engine/economy/calibration.py` has a source string
and a `Confidence` of SOURCED, DERIVED or UNSOURCED. Declare the target and
its confidence before fitting. Never upgrade a confidence flag without a
named published source. Results that depend on UNSOURCED targets are
provisional and must say so.

**No `test_` prefixes in production code.** Functions in `src/` and
`scripts/` must never start with `test_`, and no class may start with
`Test`, because pytest collects them. This includes Pydantic models and
helpers. Use names like `check_`, `evaluate_` or `validate_`.

**Guards at layer seams are load-bearing.** `_check_plausible` and
`PLAUSIBLE_RANGES` in `src/moirai/engine/economy/shock_path.py` guard the
Layer 1 to Layer 3 seam. They exist because a factor-of-twelve unit error
once produced a 2,500,000% consumption response. The specification gate
likewise refuses to build a shock path from a VAR that fails its critical
diagnostics. Do not widen ranges, weaken the checks or change defaults to
make a run pass. `check_plausibility=False` and gate overrides are explicit,
recorded opt-outs for a caller who accepts the consequences, not fixes.
`test_an_implausible_shock_is_refused` in `tests/unit/test_scenarios.py`
covers the guard.

**Tests verify correctness, not stability.** Estimators are tested by
recovering known parameters from simulated data, and impulse responses
against their closed form. Don't write snapshot tests that only lock in
current output.

**Limitations are reported, not hidden.** Stationarity tests that disagree
return INCONCLUSIVE. Bootstrap bands are pointwise and may under-cover.
Keep those caveats in the output.

**Style.** Ruff line length is 100, with rules E, F, I, UP, B and SIM.
Mypy runs in strict mode on `src/moirai`. Commit messages are prefixed by
area (`financial:`, `economy:`, `docs:`, `scenarios:`, `pipeline:`,
`chore:`). Significant decisions, especially negative results, get an ADR
in `docs/decisions/`.

## Working style

- Give complete files, ready to paste, rather than fragments or diffs.
- Work one step at a time: finish and verify one change, then stop and let
  the user review before starting the next.
