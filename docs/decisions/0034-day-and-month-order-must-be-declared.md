# 34. The order of day and month in a downloaded file must be declared, not guessed

Status: accepted, 12 November 2026

## Context

`ManualCsvAdapter` reads files published as downloads rather than through an
API, which is how most Indian statistical releases arrive. Its `ISO_DATE`
period format tried five patterns in turn:

    %Y-%m-%d, %Y-%m, %d/%m/%Y, %d-%m-%Y, %m/%d/%Y

Day-first came before month-first, so `03/04/2024` was always read as 3
April. A file written in the United States convention, where it means 4
March, was read with day and month swapped for every date up to the twelfth
of the month. Nothing fails on those rows. The first error appears at the
thirteenth, by which time the earlier observations are already in the
warehouse at the wrong periods, and a monthly series silently carries values
in months they do not belong to.

This contradicts the rule the module already states for itself: the mapping
is declared per file rather than inferred, because inference works for most
files and fails silently for the rest, which in a provenance system is worse
than failing loudly for all of them. Day and month order was the one place
the module inferred.

## Decision

`ISO_DATE` accepts only unambiguous forms: `2024-03-01` and `2024-03`.

Two formats are added, and a file written either way must say which:

| Format | Accepts | `03/04/2024` means |
|---|---|---|
| `DAY_MONTH_YEAR` | `%d/%m/%Y`, `%d-%m-%Y` | 3 April (Indian and European) |
| `MONTH_DAY_YEAR` | `%m/%d/%Y`, `%m-%d-%Y` | 4 March (United States) |

A slashed date passed under `ISO_DATE` is refused, and the refusal names
both formats rather than merely reporting a bad date, so the person reading
the error is told what to declare. Two-digit years are refused in both new
formats: `03/04/24` has three plausible readings, and a window rule for the
century would be another silent inference.

## Result

- 13 tests in `test_date_formats.py`, written first. Against the old module:
  11 failed, 2 passed, the two being the ISO acceptance tests, which already
  worked. After the change: 13 passed.
- The 27 existing `parse_period` tests are unaffected.
- The only caller of `ISO_DATE` is `scripts/check_gk_proxy.py`, reading the
  Federal Reserve's excess bond premium file, which uses ISO dates. Its
  output is unchanged.

## Consequences

- No existing series changes. The fix removes a way to be silently wrong in
  future, which is the whole value of it: there is no bug to point at today,
  only one that was waiting for the first US-formatted download.
- A publisher who changes their date format between releases now causes a
  refusal rather than a silent reinterpretation.
- This closes the ISO date ambiguity among the open findings.
- Numbering: this work was built before the Stage 3 credit layer and
  originally carried "ADR 033" in `manual_csv.py` and `test_date_formats.py`.
  The credit layer was finished and merged first and took 033, so the
  references were moved to 034.
