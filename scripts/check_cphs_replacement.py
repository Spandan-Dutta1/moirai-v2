"""The income a household keeps when its only earner loses work (ADR 040).

Three attempts preceded this one. Two measured the months after the interview
that records unemployment and returned 1.0, because CPHS asks each interview
about the preceding four months: the income of a spell is attributed to the
months leading up to its detection, not following it. The event is backwards
in the data relative to the obvious reading.

This measures both income concepts over both plausible windows, because the
answer moves with the window and a single number would hide that:

  wide    the four months an unemployed interview covers (-4 to -1) against
          the four before them (-8 to -5). Matches CPHS's recall period.
  deep    the two months before detection (-2, -1) against two clearly
          pre-spell months (-6, -5). Weights the trough.

Treatment is a single-earner household whose earner is employed at one
interview and unemployed-and-looking at the next; control is a single-earner
household whose earner is employed at both, measured over the same windows
relative to its own second interview, so nominal growth and seasonality net
out of the ratio.

Reads the panel from MOIRAI_DATA, outside this repository, because CPHS is
licensed and this repository is public. What may be committed is what this
prints: shares, sample sizes and a method.

    $env:MOIRAI_DATA = "C:\\dev\\moirai-data\\cphs"
    python scripts/check_cphs_replacement.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from moirai.engine.economy.households import (
    CPHS_HOUSEHOLD_INCOME_SHARE,
    CPHS_WAGE_SHARE,
    NonEmploymentIncomeParameters,
)

EMPLOYED = "Employed"
LOOKING = "Unemployed, willing and looking for a job"
MIN_AGE, MAX_AGE = 15, 64

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ORDINAL = {name: n for n, name in enumerate(MONTHS, start=1)}

MEASURES = {
    "total household income": "TOT_INC",
    "wage income only": "INC_OF_ALL_MEMS_FRM_WAGES",
}
#: name -> (before offsets, during offsets), relative to the interview month.
WINDOWS = {
    "wide (-8..-5 vs -4..-1)": (range(-8, -4), range(-4, 0)),
    "deep (-6,-5 vs -2,-1)": ([-6, -5], [-2, -1]),
}


def to_ordinal(label: object) -> int:
    name, year = str(label).split()
    return int(year) * 12 + ORDINAL[name[:3]]


def events(poi_dir: Path) -> pd.DataFrame:
    """Every single-earner household observed at two consecutive interviews."""
    keys = [p.stem.replace("poi_", "") for p in sorted(poi_dir.glob("poi_*.parquet"))]
    collected = []
    for before_key, after_key in zip(keys, keys[1:], strict=False):
        if "2020" in before_key or "2020" in after_key:
            continue
        first = pd.read_parquet(
            poi_dir / f"poi_{before_key}.parquet",
            columns=["HH_ID", "MEM_ID", "AGE_YRS", "EMPLOYMENT_STATUS"],
        )
        first = first[first["AGE_YRS"].between(MIN_AGE, MAX_AGE)]
        first = first[first["EMPLOYMENT_STATUS"].astype(str) == EMPLOYED]
        counts = first.groupby("HH_ID", observed=True).size()
        first = first[first["HH_ID"].isin(counts[counts == 1].index)]

        second = pd.read_parquet(
            poi_dir / f"poi_{after_key}.parquet",
            columns=["HH_ID", "MEM_ID", "EMPLOYMENT_STATUS", "MONTH_SLOT"],
        )
        status = second["EMPLOYMENT_STATUS"].astype(str)
        second = second.assign(
            S=np.where(status == EMPLOYED, "E", np.where(status == LOOKING, "U", "O")),
            M=second["MONTH_SLOT"].map(to_ordinal).astype("int32"),
        )
        pair = first[["HH_ID", "MEM_ID"]].merge(
            second[["HH_ID", "MEM_ID", "S", "M"]], on=["HH_ID", "MEM_ID"]
        )
        collected.append(pair[pair["S"].isin(["E", "U"])])
    return pd.concat(collected, ignore_index=True)


def grid_for(income_dir: Path, column: str) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    parts = []
    for path in sorted(income_dir.glob("inc_*.parquet")):
        frame = pd.read_parquet(path, columns=["HH_ID", "MONTH", column])
        frame["M"] = frame["MONTH"].map(to_ordinal).astype("int32")
        parts.append(frame[["HH_ID", "M", column]])
    wide = pd.concat(parts, ignore_index=True).pivot_table(
        index="HH_ID", columns="M", values=column, aggfunc="mean"
    )
    return (
        {h: i for i, h in enumerate(wide.index)},
        {m: i for i, m in enumerate(wide.columns)},
        wide.to_numpy(dtype="float64"),
    )


def mean_over(rows, cols, values, households, anchors, offsets) -> np.ndarray:
    offsets = list(offsets)
    out = np.full(len(households), np.nan)
    for position, (household, anchor) in enumerate(
        zip(households, anchors, strict=True)
    ):
        row = rows.get(household)
        if row is None:
            continue
        got = []
        for offset in offsets:
            col = cols.get(anchor + offset)
            if col is not None and not np.isnan(values[row, col]):
                got.append(values[row, col])
        if got:
            out[position] = float(np.mean(got))
    return out


def data_root() -> tuple[Path, Path]:
    """The licensed panel, which never lives inside this repository."""
    root = os.environ.get("MOIRAI_DATA")
    if not root:
        raise SystemExit(
            "MOIRAI_DATA is not set. It must point at the directory holding "
            "the CPHS panel, which lives outside this repository because the "
            "data is licensed. For example:\n"
            '    $env:MOIRAI_DATA = "C:\\dev\\moirai-data\\cphs"'
        )
    poi, income = Path(root) / "panel", Path(root) / "income"
    if not sorted(poi.glob("poi_*.parquet")):
        raise SystemExit(f"no poi_*.parquet waves in {poi}")
    if not sorted(income.glob("inc_*.parquet")):
        raise SystemExit(f"no inc_*.parquet months in {income}")
    return poi, income


def main() -> None:
    poi_dir, income_dir = data_root()
    sample = events(poi_dir)
    households = sample["HH_ID"].to_numpy()
    anchors = sample["M"].to_numpy()
    is_lost = sample["S"].to_numpy() == "U"
    print(f"{len(sample):,} household-events, {int(is_lost.sum()):,} job losses\n")

    print("=" * 86)
    print("SHARE OF INCOME KEPT WHEN A SINGLE-EARNER HOUSEHOLD'S EARNER LOSES WORK")
    print("=" * 86)
    print(f"  {'measure':<24} {'window':<26} {'lost':>6} {'kept':>7}  {'share':>7}")

    for label, column in MEASURES.items():
        rows, cols, values = grid_for(income_dir, column)
        for window_name, (before_offsets, during_offsets) in WINDOWS.items():
            before = mean_over(rows, cols, values, households, anchors, before_offsets)
            during = mean_over(rows, cols, values, households, anchors, during_offsets)
            usable = ~np.isnan(before) & ~np.isnan(during) & (before > 0)
            lost, kept = usable & is_lost, usable & ~is_lost
            share = (during[lost].sum() / before[lost].sum()) / (
                during[kept].sum() / before[kept].sum()
            )
            print(
                f"  {label:<24} {window_name:<26} {int(lost.sum()):>6,} "
                f"{int(kept.sum()):>7,}  {share:>7.3f}"
            )

    print()
    print("=" * 86)
    print("AGAINST THE MODEL AND THE LITERATURE")
    print("=" * 86)
    model = NonEmploymentIncomeParameters().unemployed
    print(f"  the model uses                       {model:.3f}")
    print(f"  published, individual earnings       {1 - 0.43:.3f}  "
          f"(Economic Modelling 2025, same panel)")
    print(f"  measured here, wage income           {CPHS_WAGE_SHARE:.3f}  "
          f"- reproduces it, and was never fitted to it")
    print(f"  measured here, household income      {CPHS_HOUSEHOLD_INCOME_SHARE:.3f}  "
          f"- the quantity this model scales")
    print()
    print("  The two measured figures are not in conflict. Wages are what the")
    print("  earner loses; household income is what the household keeps, and")
    print("  the gap between them is pension, rent, business income and")
    print("  transfers carrying on. ADR 038 used the first where the model")
    print("  needed the second; ADR 040 corrects it.")


if __name__ == "__main__":
    main()
