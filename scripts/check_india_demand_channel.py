"""How a Fed tightening reaches Indian output (the demand channel, step 2).

The network's demand coefficient says a 1pp Fed tightening lowers India's
output gap by 0.36 points. It is built from the same emerging market bond
yield proxy the exchange coefficient came from (ADR 019), and it pulls the
RBI towards easing, against the exchange channel. Industrial production
proved too noisy to measure it, so this uses three other outcomes:

  1. Real GDP, quarterly (FRED NGDPRNSAXDCINQ, not seasonally adjusted,
     2004-2026). The concept the coefficient is about. Few observations.
  2. India's merchandise exports in dollars, monthly, seasonally adjusted
     (OECD, XTEXVA01INM667S, 1990-2026). The direct trade channel.
  3. US imports from India, monthly (Census, IMP5330). The bilateral
     channel, not seasonally adjusted.

Seasonality is removed by working in year-on-year log changes where a
series is not adjusted: four quarters for GDP, twelve months for US imports.

Method as in check_india_fed_spillover.py: Jorda local projections on the
Bauer-Swanson surprise orthogonalised to public information, summed within
each month or quarter, with lags of the outcome's change and of the
surprise as controls and Newey-West errors. The pandemic is excluded:
output collapsed and rebounded for reasons unrelated to the Fed.

Reads data and prints. Changes nothing.
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

warnings.filterwarnings("ignore")
configure_logging("ERROR")

SURPRISES = get_paths().raw / "manual" / "frbsf_monetary_policy_surprises.xlsx"
PANDEMIC = (date(2020, 1, 1), date(2021, 12, 31))
DEMAND_COEFFICIENT = 0.36


def monthly_surprises() -> dict[date, float]:
    workbook = openpyxl.load_workbook(SURPRISES, read_only=True, data_only=True)
    rows = [r for r in workbook["Monthly (update 2023)"].iter_rows(values_only=True) if any(r)]
    header = list(rows[0])
    y, m, s = header.index("Year"), header.index("Month"), header.index("MPS_ORTH")
    surprises = {
        date(int(r[y]), int(r[m]), 1): float(r[s])
        for r in rows[1:]
        if isinstance(r[y], (int, float)) and isinstance(r[s], (int, float))
    }
    workbook.close()
    return surprises


def quarterly(monthly: dict[date, float]) -> dict[date, float]:
    """Sum each quarter's monthly surprises, dated to the quarter's first month."""
    out: dict[date, float] = {}
    for d, v in monthly.items():
        q = date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)
        out[q] = out.get(q, 0.0) + v
    return out


def fetch(code: str, start: str = "1990-01-01") -> dict[date, float]:
    with FredAdapter() as fred:
        series = fred.fetch_series(code, observation_start=start)
    return {o.period: o.value for o in series.observations if o.value is not None and o.value > 0}


def log_level(series: dict[date, float]) -> dict[date, float]:
    return {d: float(np.log(v)) * 100 for d, v in series.items()}


def year_on_year(series: dict[date, float], periods_per_year: int) -> dict[date, float]:
    ordered = sorted(series)
    logs = {d: float(np.log(series[d])) * 100 for d in ordered}
    return {
        ordered[i]: logs[ordered[i]] - logs[ordered[i - periods_per_year]]
        for i in range(periods_per_year, len(ordered))
    }


def local_projection(outcome, shock, *, horizons, lags, start, end, exclude=PANDEMIC):
    periods = sorted(d for d in outcome if start <= d <= end)
    values = np.array([outcome[d] for d in periods])
    shocks = np.array([shock.get(d, np.nan) for d in periods])
    diffs = np.diff(values, prepend=np.nan)
    rows = []
    for h in horizons:
        ys, xs = [], []
        for t in range(lags + 1, len(periods) - h):
            if exclude and any(exclude[0] <= periods[k] <= exclude[1] for k in (t, t + h)):
                continue
            regressors = np.concatenate(
                ([shocks[t]], diffs[t - lags : t][::-1], shocks[t - lags : t][::-1])
            )
            target = values[t + h] - values[t - 1]
            if np.isnan(target) or np.isnan(regressors).any():
                continue
            ys.append(target)
            xs.append(regressors)
        if len(ys) < 2 * (2 * lags + 2):
            rows.append((h, np.nan, np.nan, len(ys)))
            continue
        fit = sm.OLS(np.array(ys), sm.add_constant(np.array(xs))).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1}
        )
        rows.append((h, float(fit.params[1]), float(fit.bse[1]), len(ys)))
    return rows


def show(title: str, unit: str, period: str, rows) -> None:
    print(f"\n{title}")
    print(f"  {period:>3}  {'response':>10}  {'se':>7}  {'t':>6}  {'n':>4}")
    for h, beta, se, n in rows:
        if np.isnan(beta):
            print(f"  {h:>3}  {'too few observations':>30}  {n:>4}")
        else:
            print(f"  {h:>3}  {beta:>+9.2f}{unit}  {se:>7.2f}  {beta / se:>+6.2f}  {n:>4}")


def main() -> None:
    monthly = monthly_surprises()
    last = max(monthly)
    print(f"surprises: {min(monthly)} to {last}; pandemic excluded {PANDEMIC[0]} to {PANDEMIC[1]}")

    gdp = fetch("NGDPRNSAXDCINQ", "2003-01-01")
    print(f"real GDP: {min(gdp)} to {max(gdp)}, {len(gdp)} quarters")
    show(
        "INDIAN REAL GDP, change in year-on-year growth (pp) after a 1pp surprise",
        "pp",
        "q",
        local_projection(
            year_on_year(gdp, 4),
            quarterly(monthly),
            horizons=(0, 1, 2, 4, 6, 8),
            lags=2,
            start=date(2005, 1, 1),
            end=last,
        ),
    )

    exports = fetch("XTEXVA01INM667S")
    show(
        "INDIA'S EXPORTS in dollars, % change after a 1pp surprise (1995-2023)",
        "%",
        "m",
        local_projection(
            log_level(exports),
            monthly,
            horizons=(0, 1, 3, 6, 9, 12),
            lags=6,
            start=date(1995, 1, 1),
            end=last,
        ),
    )
    show(
        "INDIA'S EXPORTS in dollars, % change after a 1pp surprise (2013-2023)",
        "%",
        "m",
        local_projection(
            log_level(exports),
            monthly,
            horizons=(0, 1, 3, 6, 9, 12),
            lags=6,
            start=date(2013, 1, 1),
            end=last,
        ),
    )

    us_imports = fetch("IMP5330")
    show(
        "US IMPORTS FROM INDIA, change in year-on-year growth (pp) after a 1pp surprise",
        "pp",
        "m",
        local_projection(
            year_on_year(us_imports, 12),
            monthly,
            horizons=(0, 1, 3, 6, 9, 12),
            lags=6,
            start=date(1995, 1, 1),
            end=last,
        ),
    )

    print(f"\nThe network's demand coefficient is {DEMAND_COEFFICIENT}: a 1pp Fed tightening")
    print("lowers India's output gap by that many points. The GDP rows are the direct")
    print("test. Exports are about a fifth of GDP, so an export response of x percent")
    print("implies an output effect of roughly x/5 through trade alone.")


if __name__ == "__main__":
    main()
