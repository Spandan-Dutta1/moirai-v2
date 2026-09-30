"""How Fed surprises reach India over months, not minutes (F13, step 4).

The event study (check_rupee_fed_surprise.py) shows the rupee weakens on
the day of a Fed tightening surprise. The network's coefficients describe
sustained effects, so two things still need measuring:

  1. Persistence. Does the rupee's depreciation last, fade or build over
     the following year? That turns a one-day move into the sustained
     depreciation the exchange coefficient needs.
  2. The demand channel. How does Indian industrial output respond? The
     network's demand coefficient (0.36 points of output gap per point of
     Fed tightening) comes from the same emerging market bond yield proxy
     as the exchange coefficient, and it pulls the RBI the other way, so
     replacing one without the other would be lopsided.

Method: Jorda (2005) local projections on the monthly Bauer-Swanson
surprise, the sum of each month's announcement surprises, orthogonalised to
public information. For each horizon h:

    y[t+h] - y[t-1] = a + b[h] * surprise[t] + controls + e

with six lags of the outcome's change and of the surprise as controls, and
Newey-West standard errors, since overlapping horizons make the errors
serially correlated. b[h] is the response h months after a 1pp surprise.

Reads data and prints estimates. Writes nothing except the adapters' usual
archives.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import openpyxl
import statsmodels.api as sm

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.ingestion.manual_excel import RBI_IIP, ManualExcelAdapter

warnings.filterwarnings("ignore")
configure_logging("WARNING")

SURPRISES = get_paths().raw / "manual" / "frbsf_monetary_policy_surprises.xlsx"
RBI_WORKBOOK = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"
HORIZONS = (0, 1, 3, 6, 9, 12)
LAGS = 6
PANDEMIC = (date(2020, 3, 1), date(2021, 12, 1))


def monthly_surprises() -> dict[date, float]:
    workbook = openpyxl.load_workbook(SURPRISES, read_only=True, data_only=True)
    rows = [r for r in workbook["Monthly (update 2023)"].iter_rows(values_only=True) if any(r)]
    header = list(rows[0])
    y, m, s = header.index("Year"), header.index("Month"), header.index("MPS_ORTH")
    out = {}
    for row in rows[1:]:
        if isinstance(row[y], (int, float)) and isinstance(row[s], (int, float)):
            out[date(int(row[y]), int(row[m]), 1)] = float(row[s])
    workbook.close()
    return out


def fred_monthly(code: str, start: str) -> dict[date, float] | None:
    try:
        with FredAdapter() as fred:
            series = fred.fetch_series(code, observation_start=start)
    except Exception as error:
        print(f"  {code}: not available ({type(error).__name__})")
        return None
    return {o.period: o.value for o in series.observations if o.value is not None}


def local_projection(
    outcome: dict[date, float],
    shock: dict[date, float],
    *,
    start: date,
    end: date,
    exclude: tuple[date, date] | None = None,
) -> list[tuple[int, float, float, int]]:
    """Responses at each horizon: (h, beta, standard error, n)."""
    months = sorted(d for d in outcome if start <= d <= end)
    index = {d: i for i, d in enumerate(months)}
    values = np.array([outcome[d] for d in months])
    shocks = np.array([shock.get(d, np.nan) for d in months])
    diffs = np.diff(values, prepend=np.nan)

    results = []
    for h in HORIZONS:
        ys, xs = [], []
        for d in months:
            t = index[d]
            if t - 1 - LAGS < 0 or t + h >= len(months):
                continue
            if exclude and (
                exclude[0] <= d <= exclude[1] or exclude[0] <= months[t + h] <= exclude[1]
            ):
                continue
            window = np.concatenate(
                ([shocks[t]], diffs[t - LAGS : t][::-1], shocks[t - LAGS : t][::-1])
            )
            target = values[t + h] - values[t - 1]
            if np.isnan(target) or np.isnan(window).any():
                continue
            ys.append(target)
            xs.append(window)
        if len(ys) < 3 * (2 * LAGS + 2):
            results.append((h, np.nan, np.nan, len(ys)))
            continue
        fit = sm.OLS(np.array(ys), sm.add_constant(np.array(xs))).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1}
        )
        results.append((h, float(fit.params[1]), float(fit.bse[1]), len(ys)))
    return results


def show(title: str, unit: str, rows: list[tuple[int, float, float, int]]) -> None:
    print(f"\n{title}")
    print(f"  {'h':>3}  {'response':>10}  {'se':>7}  {'t':>6}  {'n':>4}")
    for h, beta, se, n in rows:
        if np.isnan(beta):
            print(f"  {h:>3}  {'too few observations':>30}  {n:>4}")
            continue
        print(f"  {h:>3}  {beta:>+9.2f}{unit}  {se:>7.2f}  {beta / se:>+6.2f}  {n:>4}")


def main() -> None:
    shock = monthly_surprises()
    first, last = min(shock), max(shock)
    print(f"monthly surprises: {first} to {last}")

    # ---- 1. persistence of the rupee response -------------------------------
    rupee = fred_monthly("EXINUS", "1994-01-01")
    if rupee:
        log_rupee = {d: float(np.log(v)) * 100 for d, v in rupee.items()}
        for label, start in (("1995-2023", date(1995, 1, 1)), ("2013-2023", date(2013, 1, 1))):
            show(
                f"RUPEE, % depreciation against the dollar after a 1pp surprise ({label})",
                "%",
                local_projection(log_rupee, shock, start=start, end=last),
            )

    # ---- 2. the demand channel: Indian industrial output ---------------------
    iip = ManualExcelAdapter(RBI_IIP).fetch(RBI_IIP.series_id, path=RBI_WORKBOOK, archive=False)
    iip_yoy = {o.period: o.value for o in iip.observations if o.value is not None}
    print(
        f"\nRBI IIP year-on-year growth: {min(iip_yoy)} to {max(iip_yoy)}, "
        f"{len(iip_yoy)} months (April 2021 and August 2024 missing)"
    )
    show(
        "INDIAN IIP, change in year-on-year growth (pp) after a 1pp surprise, pandemic excluded",
        "pp",
        local_projection(iip_yoy, shock, start=date(2011, 1, 1), end=last, exclude=PANDEMIC),
    )

    oecd = fred_monthly("INDPROINDMISMEI", "1994-01-01")
    if oecd and len(oecd) > 120:
        log_ip = {d: float(np.log(v)) * 100 for d, v in oecd.items()}
        print(f"\nOECD India industrial production index: {min(oecd)} to {max(oecd)}")
        show(
            "INDIAN INDUSTRIAL PRODUCTION (OECD index), % change after a 1pp surprise, "
            "pandemic excluded",
            "%",
            local_projection(log_ip, shock, start=date(1995, 1, 1), end=last, exclude=PANDEMIC),
        )

    print("\nReading the output")
    print("  The rupee rows turn the one-day event study into a sustained depreciation.")
    print("  The output rows are industrial production, which is more cyclical than GDP;")
    print("  the network's demand coefficient is an output gap, so a scaling step follows.")


if __name__ == "__main__":
    main()
