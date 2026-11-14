"""Labour-market rates measured on the CPHS panel, not assumed (ADR 039).

Three of this model's household parameters are rates that a panel of Indian
households measures directly, and all three were numbers with nothing behind
them: the monthly chance an unemployed household finds work, the share of
working-age people outside the labour force, and the rate at which employed
people lose work.

This script reads CMIE's People of India waves and prints those rates per
wave and per transition. **It reads them from outside the repository.** CPHS
is licensed microdata and `moirai-v2` is public, so no wave, extract or sample
may be committed - see `test_licensed_data_stays_out.py`, which fails if one
appears. Set the directory holding the compact panel:

    $env:MOIRAI_DATA = "C:\\dev\\moirai-data\\cphs"
    python scripts/check_cphs_labour.py

What may be committed is what this prints: rates, sample sizes and a method.
Those are findings, not licensed data.

CPHS surveys the same people three times a year, so consecutive waves are four
months apart. A monthly rate is backed out as 1 - (1 - p)^(1/4), which assumes
a constant hazard inside the window. Exits are usually front-loaded, so that
understates the monthly rate: the figures here are lower bounds.

Status follows CPHS's own definitions. The labour force is the employed plus
those unemployed and actively looking; people unemployed and not looking are
outside it. That definition is why CPHS participation sits twenty points below
the PLFS measure, which is a live disagreement between two credible surveys
rather than an error in either.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from moirai.engine.economy.behaviour import BehaviourParameters

EMPLOYED = "Employed"
LOOKING = "Unemployed, willing and looking for a job"

MIN_AGE, MAX_AGE = 15, 64

#: The weight present in every wave. CPHS dropped the unprefixed weights in
#: May 2023, keeping only the responding-member ones, which are the correct
#: weights here in any case since this keeps accepted responses only.
WEIGHT = "R_GE15_MEM_WGT_W"

#: What the model used before this panel was read, kept so the comparison
#: still shows the size of the correction after the default has moved.
PRIOR_JOB_FINDING_RATE = 0.18

#: Participation the model implies: everyone of working age who is not out of
#: the labour force. Still unsourced, deliberately - see the closing note.
MOIRAI_PARTICIPATION = 0.65


def arrangement_group(value: object) -> str:
    """Classify an employment arrangement, refusing anything unrecognised.

    CPHS split these categories in May 2022 to separate digital platform work,
    renaming "Salaried - Temporary" as it did. Matching the old names alone
    dropped a third of the salaried from that wave onward, and the result
    looked exactly like a labour-market shift rather than a broken filter.
    Substring matching survives the rename; the raise is what stops the next
    one passing silently.
    """
    low = str(value).lower()
    if "salaried" in low:
        return "salaried"
    if "self employed" in low or "self-employed" in low:
        return "self"
    if "daily wage" in low or "casual" in low:
        return "casual"
    if low in ("not applicable", "data not available", "nan", "none", ""):
        return "none"
    raise ValueError(f"unrecognised employment arrangement: {value!r}")


def monthly(four_month: float) -> float:
    """A four-month transition probability expressed as a monthly one."""
    return 1.0 - (1.0 - four_month) ** 0.25


def panel_directory() -> Path:
    """Where the licensed panel lives, which is never inside this repository."""
    root = os.environ.get("MOIRAI_DATA")
    if not root:
        raise SystemExit(
            "MOIRAI_DATA is not set. It must point at the directory holding "
            "the CPHS panel, which lives outside this repository because the "
            "data is licensed. For example:\n"
            '    $env:MOIRAI_DATA = "C:\\dev\\moirai-data\\cphs"'
        )
    directory = Path(root) / "panel"
    waves = sorted(directory.glob("poi_*.parquet"))
    if not waves:
        raise SystemExit(f"no poi_*.parquet waves found in {directory}")
    return directory


def load(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except ImportError as err:  # pandas needs an engine to read parquet
        raise SystemExit(
            "reading the panel needs a parquet engine, which this environment "
            "does not have. Install one:\n"
            "    uv add --dev pyarrow\n"
            f"({err})"
        ) from err
    frame = frame[frame["AGE_YRS"].between(MIN_AGE, MAX_AGE)].copy()
    raw = frame["EMPLOYMENT_STATUS"].astype(str)
    frame["S"] = np.where(raw == EMPLOYED, "E", np.where(raw == LOOKING, "U", "O"))
    frame["W"] = frame[WEIGHT].fillna(0.0)
    frame["GROUP"] = frame["EMPLOYMENT_ARRANGEMENT"].map(arrangement_group)
    return frame


def main() -> None:
    directory = panel_directory()
    waves = sorted(directory.glob("poi_*.parquet"))

    print("=" * 94)
    print(f"CPHS LABOUR RATES, {len(waves)} WAVES, AGED {MIN_AGE}-{MAX_AGE}, WEIGHTED")
    print("=" * 94)
    print(f"  {'wave':<20} {'members':>9}  {'participation':>13}  {'unemployed':>10}")

    frames: dict[str, pd.DataFrame] = {}
    participation: list[float] = []
    for path in waves:
        key = path.stem.replace("poi_", "")
        frame = load(path)
        frames[key] = frame
        w, s = frame["W"], frame["S"]
        force = w[np.isin(s, ["E", "U"])].sum()
        share = force / w.sum()
        participation.append(share)
        print(
            f"  {key:<20} {len(frame):>9,}  {100 * share:>12.1f}%  "
            f"{100 * w[s == 'U'].sum() / force:>9.1f}%"
        )

    print()
    print("=" * 94)
    print("TRANSITIONS BETWEEN CONSECUTIVE WAVES (four months apart)")
    print("=" * 94)
    print(f"  {'pair':<16} {'matched':>9}  {'E->U':>6} {'U->E':>6}  {'E->U/mo':>8} {'U->E/mo':>8}")

    keys = sorted(frames)
    rows: list[tuple[str, float, float]] = []
    for before, after in zip(keys, keys[1:], strict=False):
        joined = frames[before].merge(
            frames[after][["HH_ID", "MEM_ID", "S"]],
            on=["HH_ID", "MEM_ID"],
            suffixes=("_0", "_1"),
            how="inner",
        )
        if joined.empty:
            continue
        w = joined["W"]
        was_employed, was_unemployed = joined["S_0"] == "E", joined["S_0"] == "U"
        lost = was_employed & (joined["S_1"] == "U")
        found = was_unemployed & (joined["S_1"] == "E")
        e_to_u = w[lost].sum() / max(w[was_employed].sum(), 1e-9)
        u_to_e = w[found].sum() / max(w[was_unemployed].sum(), 1e-9)
        label = f"{before[:6]}-{after[:6]}"
        rows.append((label, e_to_u, u_to_e))
        print(
            f"  {label:<16} {len(joined):>9,}  {100 * e_to_u:>5.1f}% {100 * u_to_e:>5.1f}%  "
            f"{100 * monthly(e_to_u):>7.2f}% {100 * monthly(u_to_e):>7.2f}%"
        )

    steady = [r for r in rows if "2020" not in r[0]]
    finding = float(np.mean([monthly(r[2]) for r in steady]))
    losing = float(np.mean([monthly(r[1]) for r in steady]))
    all_finding = float(np.mean([monthly(r[2]) for r in rows]))

    print()
    print("=" * 94)
    print("WHAT THE MODEL USES, AND WHAT THE PANEL SAYS")
    print("=" * 94)
    print(f"  pairs used: {len(steady)} of {len(rows)}; the rest touch 2020, when")
    print("  lockdown and a 48 per cent collapse in survey response are inseparable")
    print(f"  (including them raises job finding to {all_finding:.4f}).")
    print()
    model_finding = BehaviourParameters().job_finding_rate
    print(f"  {'quantity':<34} {'model':>9} {'CPHS':>9}  {'ratio':>7}")
    print(
        f"  {'monthly job finding rate':<34} {model_finding:>9.4f} "
        f"{finding:>9.4f}  {model_finding / finding:>6.1f}x"
    )
    print(
        f"  {'  (before this panel was read)':<34} {PRIOR_JOB_FINDING_RATE:>9.4f} "
        f"{finding:>9.4f}  {PRIOR_JOB_FINDING_RATE / finding:>6.1f}x"
    )
    print(f"  {'monthly job loss rate, steady state':<34} {'-':>9} {losing:>9.4f}")
    print(
        f"  {'labour force participation':<34} {MOIRAI_PARTICIPATION:>9.3f} "
        f"{float(np.mean(participation)):>9.3f}  "
        f"{MOIRAI_PARTICIPATION / float(np.mean(participation)):>6.1f}x"
    )
    print()
    print("  The job finding rate was the one to act on: before this panel was")
    print("  read, the model refilled an unemployment spell six times faster than")
    print("  Indian data does, so an employment shock died out too quickly.")
    print("  Participation is not a correction to make unilaterally - CPHS and the")
    print("  PLFS disagree by more than twenty points, and which one is right")
    print("  decides who sits in the denominator of every distributional result.")


if __name__ == "__main__":
    sys.exit(main())
