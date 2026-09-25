# Moirai

A local-first simulation that traces a central bank rate decision through
to individual households.

Five central banks solve a game to set policy. An estimated VAR supplies
the transmission. Twelve commercial banks set the rates people actually
pay. Two hundred thousand heterogeneous households bear the result.

**1,136 tests. Nine architecture decision records. No external API
dependency in the engine.**

---

## The finding

A monetary tightening looks almost neutral in aggregate and is not.

| | consumption change |
|---|---|
| floating-rate borrowers | **-4.48%** |
| fixed-rate borrowers | -1.10% |
| net savers | -0.30% |
| **aggregate** | **-0.75%** |

A representative-household model reports the aggregate and concludes
monetary policy barely moves consumption. The four-point spread is the
finding, and it is invisible without heterogeneity. It is also the
intuition that heterogeneous-agent models formalise: policy works largely
through income and redistribution rather than intertemporal substitution.

```bash
python scripts/run_pipeline.py     # the chain, end to end
python scripts/run_scenarios.py    # four named scenarios compared
```

---

## What makes it more than a VAR and some agents

**A specification gate.** If the VAR fails its residual autocorrelation
tests, the pipeline refuses to build a shock path from it. The Indian
pipeline runs and withholds its distributional result for exactly this
reason.

**Identification treated as an assumption.** Cholesky ordering is a claim
about the world, not a result. `ordering_sensitivity` re-solves under
every permutation and reports the range. The lower bound is frequently
exactly zero, which is the assumption showing rather than an estimate.

**Bitemporal storage.** Every observation carries two dates: when it was
true and when it was learned. US GDP for Q2 2020 was 17,205 in August
2020 and 19,034 by 2024. Evaluating a June 2020 forecast against the
latter credits the model with data that did not exist.

**Out-of-sample validation that fails.** The banking layer was calibrated
on the RBI's easing cycle, where five of six checks pass by construction.
The tightening cycle was held out. **One of three predictions survived.**
The failures are documented rather than fixed, because adding mechanisms
after seeing the answer would convert the test into a second fit.

**Calibration provenance.** Every target is marked SOURCED, DERIVED or
UNSOURCED. Four of seven household targets are unsourced, and the reports
say so.

---

## Architecture


### Layer 0 — Data Fabric

Bitemporal warehouse in DuckDB. Adapters for FRED, World Bank, RBI
spreadsheets and manual CSV, all behind one contract, so nothing above
Layer 0 knows where data came from. Raw responses archived with SHA-256
content hashes and provenance sidecars. Fetch and parse are separate, so a
schema change years later is fixed by replaying the archive rather than
refetching today's vintage.

### Layer 1a — Policy games

Five central banks with their published mandates: the Fed's dual mandate,
the RBI's 4% target with a legislated 2–6% band, the ECB's
price-stability-primary. The Fed is the only major central bank with a
dual mandate and the RBI the only one with a band, so a game with
symmetric objectives would misrepresent the system.

Spillovers are tiered rather than uniform. The Fed exports its stance and
absorbs almost nothing; the RBI the reverse. Nash and Stackelberg
equilibria are solved analytically, since a five-bank game on a 25bp grid
would be 39 million profiles.

### Layer 1b — Causal intelligence

Stationarity testing with ADF and KPSS reporting disagreement rather than
resolving it. Transformations chosen from series metadata and recorded.
VAR with information-criterion lag selection and stability checking.
Cholesky and sign-restriction identification. Impulse responses, variance
decomposition, bootstrap bands. Specification diagnostics separating
critical failures from advisory ones.

### Layer 2 — Commercial banks

Twelve banks across the RBI's three groups. Pass-through is derived from
each bank's external benchmark share, retail deposit dependence and
stress, not set directly, and calibrated against the RBI's published
transmission figures. Of a 100bp policy rise, borrowers absorb roughly
80bp and savers receive 31bp; the wedge is bank margin.

### Layer 3 — Households

200,000 agents as parallel NumPy arrays rather than objects. A rate change
reaches them through four channels: debt service, interest income,
employment risk and inflation eroding real income. The marginal propensity
to consume falls with liquid wealth, which is why *who* is hit matters as
much as how much. Calibrated to AIDIS 2019 and PLFS 2023-24.

---

## Getting started

Requires Python 3.11 or later.

```bash
git clone https://github.com/Spandan-Dutta1/moirai-v2.git
cd moirai-v2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

A free FRED API key is needed for the FRED adapter. The World Bank adapter
needs none, and the RBI adapter reads a spreadsheet from `data/raw/manual`.

```bash
echo "MOIRAI_FRED_API_KEY=your_key_here" > .env
pytest
```

---

## Design principles

**Every result carries its provenance.** Content hashes on raw data,
recorded transformations, stated identifying assumptions, fixed seeds.

**Illegal states are unrepresentable.** Models are frozen and validated at
construction, so a `TimeSeries` cannot be unsorted, an `Observation`
cannot hold a NaN, and a run's configuration cannot change mid-run.

**Guards at the seams.** Layer boundaries are where unit errors live: a
factor-of-twelve error once produced a consumption response of 2,500,000%.
The plausibility guard now catches that class of mistake at the boundary
rather than three layers downstream.

**Limitations reported, not hidden.** Percentile bootstrap bands can
under-cover in autoregressive models; that caveat is written into the
ledger output. Stationarity tests that disagree return INCONCLUSIVE.

**Tests verify correctness, not stability.** Estimators are tested by
recovering known parameters from simulated data; impulse responses are
checked against their closed form.

---

## Known limitations

- **Partial equilibrium.** Household responses do not feed back into the
  aggregate path. Closing that loop is the Krusell-Smith problem. The
  direction of the bias is known: general equilibrium amplifies, so these
  estimates are conservative.
- **The Indian VAR is not credibly identified.** Output rises after a rate
  hike across four specifications, which is the RBI tightening into
  expected strength. Documented in ADR 007; the result is withheld.
- **Roughly 15 sourced parameters against 35-40 assumed ones**, audited in
  ADR 008. Layers 0 and 1 rest on data; layers 2 and 3 rest substantially
  on judgement.
- **Two parameters were fitted to their targets** and their agreement is
  construction rather than evidence. Also in ADR 008.
- Behavioural rules rather than solved optimisation. Bootstrap bands are
  pointwise, not joint. No credit default, no interbank market.

---

## Decisions

Nine ADRs in `docs/decisions/`, including why the LLM belief society was
removed, why Indian data uses year-on-year series, why the Indian VAR is
not credibly identified, the parameter provenance audit, and the
out-of-sample validation results.

---

## Development

```bash
pytest              # 1,136 tests
ruff check .        # lint
mypy src/moirai     # type check
```