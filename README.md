markdown
# Moirai

A local-first economic intelligence platform. Moirai ingests economic data
with full provenance, stores it bitemporally, and estimates causal models
that report their own assumptions and refuse to vouch for themselves when
the specification does not hold.

**Status:** Phase 3 of 10. Data Fabric and Causal Intelligence complete.
661 tests passing.

---

## Why bitemporal storage

Most forecasting systems store one value per period. That makes honest
backtesting impossible.

US real GDP for Q2 2020, as it was known on three different dates:

| Known on | Value |
|---|---|
| 2020-08-01 | 17,205.8 |
| 2021-01-01 | 17,302.5 |
| 2024-01-01 | 19,034.8 |

Evaluating a forecast made in June 2020 against today's figure credits the
model with information that did not exist when it forecast. Moirai stores
both *when a fact was true* and *when it was learned*, so a backtest can
ask the only honest question: what did we know that day?

```python
from datetime import UTC, datetime
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

with Warehouse() as warehouse:
    vintage = warehouse.as_of("gdpc1", datetime(2020, 8, 1, tzinfo=UTC))
```

## Why the models argue with you

Estimation is statistics; identification is economics. A VAR fits
correlations. Turning those into causal statements needs an assumption the
data cannot test, so Moirai carries the assumption alongside every result
and measures how much the answer depends on it.

Running the canonical monetary VAR through the full pipeline:

CRITICAL: portmanteau, ljung_box[cpiaucsl], ljung_box[fedfunds]
usable : False


Six lags left serial correlation in the residuals, which means the
confidence bands computed from that model are too narrow. The pipeline
says so rather than letting the result through.

---

## Architecture

Layer 0 Data Fabric ingest, validate, version, store
Layer 1 Causal Intelligence VAR, identification, IRF, diagnostics
Layer 2 Financial System planned
Layer 3 Economy Simulation planned
Layer 4 Evidence Fusion planned
Layer 5 Validation planned


### Layer 0 — Data Fabric

- Source adapters behind one contract (FRED, World Bank), so nothing above
  Layer 0 knows where data came from
- Raw responses archived with SHA-256 content hashes and provenance
  sidecars, so any published number traces back to bytes that can be
  verified unaltered
- Fetch and parse are separate, so a schema change years later is fixed by
  replaying the archive rather than refetching today's vintage
- DuckDB warehouse with as-of queries pushed into SQL

### Layer 1 — Causal Intelligence

- Stationarity testing with ADF and KPSS, reporting disagreement rather
  than resolving it
- Transformations chosen from series metadata and recorded for the ledger
- VAR estimation with information-criterion lag selection and stability
  checking
- Structural identification: Cholesky, sign restrictions, and an ordering
  sensitivity analysis that reports how much a conclusion depends on the
  assumed causal order
- Impulse responses, forecast error variance decomposition, and residual
  bootstrap confidence bands
- Specification diagnostics separating critical failures from advisory ones

---

## Getting started

Requires Python 3.11 or later.

```bash
git clone https://github.com/Spandan-Dutta1/moirai-v2.git
cd moirai-v2
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

A free FRED API key is needed for the FRED adapter
(https://fred.stlouisfed.org/docs/api/api_key.html). The World Bank
adapter needs no key.

```bash
echo "MOIRAI_FRED_API_KEY=your_key_here" > .env
pytest
```

### A complete example

```python
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import bootstrap_bands, impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

with FredAdapter() as fred:
    prepared = [
        prepare(fred.fetch_series(code, observation_start="1960-01-01"))
        for code in ("INDPRO", "CPIAUCSL", "FEDFUNDS")
    ]

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)

var = estimate_var(data, names, n_lags=6, periods=periods)
report = diagnose(var)
if not report.is_usable:
    print("specification problems:", report.summary())

model = identify_cholesky(var, ordering=names)
responses = impulse_responses(model, horizon=36)
bands = bootstrap_bands(model, horizon=24, n_draws=500)
```

Runnable versions of each stage live in `scripts/`.

---

## Design principles

**Every result carries its provenance.** Content hashes on raw data,
recorded transformations, stated identifying assumptions, fixed random
seeds.

**Illegal states are unrepresentable.** Models are frozen and validated at
construction, so a `TimeSeries` cannot be unsorted, an `Observation` cannot
hold a NaN, and a run's configuration cannot change mid-run.

**Limitations are reported, not hidden.** Percentile bootstrap bands can
under-cover in autoregressive models (Kilian 1998); that caveat is written
into the ledger output rather than a footnote. Stationarity tests that
disagree return `INCONCLUSIVE` rather than picking the convenient answer.

**Tests verify correctness, not stability.** Estimators are tested by
recovering known parameters from simulated data; impulse responses are
checked against their closed form.

---

## Known limitations

- Indian data sources are not yet implemented; FRED's India coverage is
  shallow and lagged
- The CUSUM implementation is a simplified corridor test with low power
  against changes in slope coefficients
- Bootstrap bands are pointwise, not joint
- Sign restriction identification returns a set; bands are not computed for
  it, since the set already expresses the range

---

## Development

```bash
pytest              # 661 tests
ruff check .        # lint
mypy src/moirai     # type check
```
