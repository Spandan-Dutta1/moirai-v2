"""Every cell the spreadsheet adapter reads as missing, and what it actually held.

`_to_value` returns None for publisher sentinels, for bands such as
"8.35/9.90", and for any text it cannot parse as a number. The last rule is
a catch-all: a figure marked provisional ("5.9 (P)"), carrying a footnote
star ("6.5*"), or written with a Unicode minus sign would be read as
missing without a warning, and `drop_missing` would then remove it before
estimation.

This script opens the real RBI workbook, walks the same rows the adapter
reads, and prints the raw content of every cell that became missing, so
each gap can be classified as genuinely unpublished or as a figure the
parser failed to read. It reads the file and writes nothing.
"""

from __future__ import annotations

import io
import warnings

import openpyxl

from moirai.core.paths import get_paths
from moirai.engine.data_fabric.ingestion.manual_excel import (
    MISSING_MARKERS,
    RBI_MONTHLY_SPECS,
    _to_period,
    _to_value,
)

PATH = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"


def classify(cell: object) -> str:
    if cell is None:
        return "empty cell"
    text = str(cell).strip()
    if text.lower() in MISSING_MARKERS:
        return "publisher sentinel"
    if "/" in text:
        return "band"
    return "UNPARSED TEXT: check whether this is a real figure"


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(
            io.BytesIO(PATH.read_bytes()), data_only=True, read_only=True
        )

    print(f"workbook: {PATH}")
    unparsed_total = 0
    for spec in RBI_MONTHLY_SPECS:
        rows = list(workbook[spec.sheet].iter_rows(values_only=True))
        last = spec.last_data_row or len(rows)
        missing: list[tuple[int, object, object]] = []
        n_periods = 0
        for index in range(spec.first_data_row - 1, min(last, len(rows))):
            row = rows[index]
            if len(row) < max(spec.date_column, spec.value_column):
                continue
            period = _to_period(
                row[spec.date_column - 1], anchor_to_start=spec.anchor_to_period_start
            )
            if period is None:
                continue
            n_periods += 1
            cell = row[spec.value_column - 1]
            if _to_value(cell) is None:
                missing.append((index + 1, period, cell))

        print()
        print("=" * 78)
        print(
            f"{spec.series_id}  (column {spec.value_column}, {n_periods} dated rows, "
            f"{len(missing)} read as missing)"
        )
        print("=" * 78)
        if not missing:
            print("  none")
            continue

        # Leading and trailing runs are coverage (a series starting later);
        # anything between them is an interior gap, which is what matters.
        periods_with_values = [
            _to_period(r[spec.date_column - 1], anchor_to_start=spec.anchor_to_period_start)
            for r in rows[spec.first_data_row - 1 : min(last, len(rows))]
            if len(r) >= max(spec.date_column, spec.value_column)
            and _to_period(r[spec.date_column - 1], anchor_to_start=spec.anchor_to_period_start)
            and _to_value(r[spec.value_column - 1]) is not None
        ]
        start, end = min(periods_with_values), max(periods_with_values)
        interior = [m for m in missing if start < m[1] < end]
        edges = len(missing) - len(interior)
        print(f"  values from {start} to {end}; {edges} missing at the edges (coverage)")
        print(f"  interior gaps: {len(interior)}")
        for row_number, period, cell in interior:
            kind = classify(cell)
            if kind.startswith("UNPARSED"):
                unparsed_total += 1
            print(f"    row {row_number:>4}  {period}  raw={cell!r:<22} {kind}")

    workbook.close()
    print()
    print(f"cells holding text the parser could not read: {unparsed_total}")
    if unparsed_total:
        print("each one above may be a real figure recorded as missing.")


if __name__ == "__main__":
    main()
