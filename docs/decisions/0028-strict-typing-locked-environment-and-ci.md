# 28. Strict typing, a locked environment and continuous integration

Status: accepted, 4 November 2026

## Context

mypy was configured as strict but reported 52 errors in 20 files, so it
enforced nothing. Dependencies were declared as lower bounds (`numpy>=2.0`),
so a fresh install months later would pull whatever was newest. Nothing ran
the checks except a person remembering to.

## Decision

1. Python 3.13 everywhere (`requires-python`, ruff, mypy). Type stubs for
   scipy, pandas, openpyxl and pytz; the pydantic mypy plugin; statsmodels
   declared untyped.
2. mypy strict to zero. 8 errors cleared by configuration. 18 read as possible
   bugs (`manual_csv`, `central_banks`, `commercial_banks`): none was a runtime
   bug; each was logic mypy could not follow (`_fail` annotated `None` instead
   of `NoReturn`; optional bands guarded by a property; a mixed-type results
   dict). 26 mechanical (numpy returns, generic arguments, annotations).
3. `uv.lock`, pinned to the environment that passed 1,321 tests. The newest
   resolution would have moved duckdb 1.5.5 -> 2.0.0, statsmodels 0.14.6 ->
   0.15.0, python-dateutil 2.9 -> 4.0 and pandas 3.0.5 -> 3.1.0. Upgrading is
   deferred to its own change with its own comparison run.
4. GitHub Actions on every push: ruff, mypy strict, pytest, installed with
   `uv sync --locked`. Runner image pinned (`ubuntu-24.04`); actions pinned to
   commit SHAs.

## Evidence

- mypy: `Success: no issues found in 49 source files`. ruff clean.
- The lock is complete: four leftover packages (formulaic, interface-meta,
  narwhals, wrapt) were imported nowhere; removed; 1,321 tests pass on locked
  packages only.
- CI runs 1 and 2 green (47s, 53s) on a clean Linux machine: the test suite is
  hermetic; it needs no local data, network or API key.
- `compare_runs.ps1`, main against `phase-a/tooling`: both 1,321 passed, each
  importing its own code. The masked comparison flagged five files, all traced:
  `print_var_equations` (untracked in main only); `check_var` (FRED returned
  different raw bytes for INDCPIALLMINMEI minutes apart; every statistic
  identical); `check_india_var` and `check_rbi` (PowerShell wraps the longer
  trial path in an openpyxl warning at a different column); `validate_2022`
  (stdout and stderr interleaved differently; identical content). No numerical
  difference.

## Consequences

- Every push is linted, strictly typed and tested on a clean machine.
- An install reproduces the validated environment, not the newest one.
- Found while reading, not fixed here: `parse_period` ignores the second year
  of an Indian fiscal-year label, so "2019-25" reads as FY2019-20 (queued, test
  first).
- The comparison tool lived outside the repository (`C:\dev\compare_runs.ps1`).
  It moves into `scripts/` with masking for content hashes, wrapped paths and
  stream order (open item 18).
- The RBI starting-inflation fix described in the study notes as ADR 028 was
  never committed; it will be redone as ADR 029.