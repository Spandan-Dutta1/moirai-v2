"""How far the rupee moves when the Fed surprises the market (F13, step 2).

The network's most influential coefficient says a one point Fed tightening
raises Indian inflation by 0.42 points through a weaker rupee. It is an
emerging market bond yield estimate relabelled as an inflation effect
(ADR 008). This script measures the first half of the chain it stands for:
how much the rupee depreciates against the dollar per unit of Fed surprise.

Identification is high frequency. Bauer and Swanson (2023) measure each
surprise from futures rates in a 30-minute window around the announcement,
so nothing else that day can explain it. The surprise is scaled so that its
effect on expected US rates about a year ahead is one percentage point, so
the coefficient reads as the rupee's move per point of expected tightening.

Timing. FRED's daily rupee rate (DEXINUS) is the noon New York rate, and
most announcements come at 2pm. For an announcement at or after noon, the
first rate that can reflect it is the next trading day's; before noon, it
is the same day's. The window therefore runs from the last noon fixing
before the announcement to the first one after it.

Reads data and prints estimates. Writes nothing except FRED's usual
archive of the raw response.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime

import numpy as np
import openpyxl
import statsmodels.api as sm

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

warnings.filterwarnings("ignore")
configure_logging("WARNING")

SURPRISES = get_paths().raw / "manual" / "frbsf_monetary_policy_surprises.xlsx"
SHEET = "FOMC (update 2023)"


def announcement_hour(text: str) -> float:
    """'2:00pm' -> 14.0, '11:30am' -> 11.5."""
    text = text.strip().lower()
    clock, half = text[:-2], text[-2:]
    hours, minutes = (int(x) for x in clock.split(":"))
    if half == "pm" and hours != 12:
        hours += 12
    if half == "am" and hours == 12:
        hours = 0
    return hours + minutes / 60


def load_surprises() -> list[dict]:
    workbook = openpyxl.load_workbook(SURPRISES, read_only=True, data_only=True)
    rows = [r for r in workbook[SHEET].iter_rows(values_only=True) if any(r)]
    header = list(rows[0])
    column = {
        name: header.index(name) for name in ("Date", "Time", "Unscheduled", "MPS", "MPS_ORTH")
    }
    events = []
    for row in rows[1:]:
        day = row[column["Date"]]
        if not isinstance(day, datetime):
            continue

        def number(name: str, row: tuple = row) -> float | None:
            value = row[column[name]]
            return float(value) if isinstance(value, (int, float)) else None

        events.append(
            {
                "date": day.date(),
                "hour": announcement_hour(str(row[column["Time"]])),
                "unscheduled": bool(row[column["Unscheduled"]]),
                "mps": number("MPS"),
                "mps_orth": number("MPS_ORTH"),
            }
        )
    workbook.close()
    return events


def load_rupee() -> tuple[list[date], np.ndarray]:
    with FredAdapter() as fred:
        series = fred.fetch_series("DEXINUS", observation_start="1995-01-01")
    points = [(o.period, o.value) for o in series.observations if o.value is not None]
    return [p for p, _ in points], np.log(np.array([v for _, v in points], dtype=float))


def window(days: list[date], log_rate: np.ndarray, event: dict, *, shift: int = 0) -> float | None:
    """Percent change in rupees per dollar across the announcement.

    `shift` moves the window back by that many trading days, which gives
    a placebo: a window that closes before the announcement should show
    nothing.
    """
    after = next((i for i, d in enumerate(days) if d > event["date"]), None)
    same = next((i for i, d in enumerate(days) if d == event["date"]), None)
    if event["hour"] >= 12.0:
        # the announcement day's noon rate is the last one before it
        if same is None or after is None:
            return None
        start, end = same, after
    else:
        # the same day's noon rate is the first one after it
        if same is None or same == 0:
            return None
        start, end = same - 1, same
    start, end = start - shift, end - shift
    if start < 0:
        return None
    return float((log_rate[end] - log_rate[start]) * 100.0)


def estimate(
    label: str, sample: list[dict], days, log_rate, *, measure: str = "mps_orth", shift: int = 0
) -> None:
    pairs = [
        (e[measure], window(days, log_rate, e, shift=shift))
        for e in sample
        if e[measure] is not None
    ]
    pairs = [(x, y) for x, y in pairs if y is not None]
    if len(pairs) < 10:
        print(f"  {label:<44} too few events ({len(pairs)})")
        return
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")
    beta, se = fit.params[1], fit.bse[1]
    print(
        f"  {label:<44} n={len(pairs):>3}  beta={beta:>+6.2f}%  se={se:>5.2f}  "
        f"t={beta / se:>+5.2f}  R2={fit.rsquared:.3f}"
    )


def main() -> None:
    events = [e for e in load_surprises() if e["date"] >= date(1995, 1, 4)]
    days, log_rate = load_rupee()
    print(f"surprises: {len(events)} announcements, {events[0]['date']} to {events[-1]['date']}")
    print(f"rupee    : {len(days)} daily noon fixings, {days[0]} to {days[-1]}")
    print("\nrupee depreciation (%, rupees per dollar up) per 1pp Fed surprise")
    print("beta > 0 means a tightening surprise weakens the rupee\n")

    scheduled = [e for e in events if not e["unscheduled"]]
    estimate("all announcements, MPS_ORTH (main)", events, days, log_rate)
    estimate("scheduled only, MPS_ORTH", scheduled, days, log_rate)
    estimate("all announcements, MPS (raw)", events, days, log_rate, measure="mps")
    estimate("1995-2007, MPS_ORTH", [e for e in events if e["date"].year <= 2007], days, log_rate)
    estimate("2008-2023, MPS_ORTH", [e for e in events if e["date"].year >= 2008], days, log_rate)
    estimate(
        "2013-2023, MPS_ORTH (after the taper tantrum)",
        [e for e in events if e["date"].year >= 2013],
        days,
        log_rate,
    )
    print()
    estimate("PLACEBO: window one day earlier", events, days, log_rate, shift=1)
    estimate("PLACEBO: window two days earlier", events, days, log_rate, shift=2)
    recent = [e for e in events if e["date"].year >= 2013]
    estimate("PLACEBO 2013-2023: one day earlier", recent, days, log_rate, shift=1)
    estimate("PLACEBO 2013-2023: two days earlier", recent, days, log_rate, shift=2)

    sizes = np.array([abs(e["mps_orth"]) for e in events if e["mps_orth"] is not None])
    print(
        f"\ntypical surprise: median |MPS_ORTH| = {np.median(sizes):.3f}pp, "
        f"90th percentile {np.percentile(sizes, 90):.3f}pp"
    )


if __name__ == "__main__":
    main()
