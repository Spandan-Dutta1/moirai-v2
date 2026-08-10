# 2. Defer Indian data sources until a research question requires them

Date: 2026-08-11
Status: Accepted

## Context

The project's distinctive value is Indian economic analysis: G-Sec
auctions, RBI operations, SEZ policy. Three access routes were considered:

- **RBI DBIE**: deep time series, but no API. Requires scraping HTML or
  parsing downloaded Excel, and breaks whenever RBI changes a page.
- **data.gov.in**: a real REST API with keys, but its 237,000 resources are
  mostly flat tables from parliamentary questions rather than time series.
  Many stopped updating in 2023. Each resource has arbitrary columns, so a
  per-resource field mapping is needed rather than one parser.
- **FRED and World Bank**: clean APIs with some India coverage via OECD and
  IMF feeds. Shallow and lagged, but immediately usable.

## Decision

Build Layer 0 against FRED and World Bank only. Defer Indian sources until
a specific research question defines what is actually needed.

## Consequences

**Gained:** Layer 1 could be built and validated immediately against the
canonical monetary VAR, where results can be checked against published
findings. Momentum was preserved through the conceptually hardest phase.

**Lost:** the project currently demonstrates its machinery on US data,
which is not where its distinctive contribution lies.

**Cheap to reverse:** the adapter contract was validated by adding World
Bank, which has a different response shape, no authentication, and no
vintages. That change required no modification above Layer 0. A third
adapter is expected to be similarly contained.

**Note on scope:** data.gov.in will need a `ResourceSpec` design where each
resource declares its period column, value column, period format and
frequency. Indian fiscal years ("2019-20" meaning April 2019 to March 2020)
must be parsed explicitly; treating them as calendar years misaligns them
against FRED data by up to nine months.

A manual CSV adapter reading DBIE downloads from `data/raw/` may be the
faster route to real RBI data, with full provenance and no scraping.