"""Does the RBI follow the Fed? The balance of the two channels, measured.

ADR 021 left the headline conditional. A Fed tightening pushes the RBI to
tighten through the rupee (ADR 019) and to ease through slower Indian
demand, and Indian data confirm both channels without saying which is
larger. The network's answer, the RBI's caused move, is the balance of the
two, and it can be tested without measuring either: estimate how India's
policy rate itself responds to a Fed surprise.

Method as in the other evidence scripts: Jorda local projections of the
policy rate's level on the Bauer-Swanson monthly surprise orthogonalised
to public information, six lags of the rate's change and of the surprise
as controls, Newey-West errors. 2020-2021 are excluded: the RBI cut for
the pandemic, not for the Fed.

Two policy rate series cover different periods:
  - FRED INTDSRINM193N, IMF discount rate for India, 2000-2022
  - RBI policy repo rate from the project's workbook, 2010-2026

The network's counterpart, for comparison: under the headline scenario
the Fed moves +88bp and the RBI's caused move is +12.9bp at the default
exchange coefficient and +4.0bp at its evidence floor, so 0.15 and 0.05
points of RBI move per point of Fed move.

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
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)

warnings.filterwarnings("ignore")
configure_logging("ERROR")

SURPRISES = get_paths().raw / "manual" / "frbsf_monetary_policy_surprises.xlsx"
RBI_WORKBOOK = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"
PANDEMIC = (date(2020, 1, 1), date(2021, 12, 31))
HORIZONS = (0, 1, 3, 6, 9, 12)
LAGS = 6
MODEL_RATIO_DEFAULT = 12.9 / 87.7
MODEL_RATIO_FLOOR = 4.0 / 87.7


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


def local_projection(outcome, shock, *, start, end, control=None):
    """With `control`, lags of an Indian inflation series join the controls,
    so the response is not credited to the Fed when the RBI was reacting to
    its own inflation in the same months."""
    periods = sorted(d for d in outcome if start <= d <= end)
    values = np.array([outcome[d] for d in periods])
    shocks = np.array([shock.get(d, np.nan) for d in periods])
    extra = np.array([control.get(d, np.nan) for d in periods]) if control else None
    diffs = np.diff(values, prepend=np.nan)
    rows = []
    for h in HORIZONS:
        ys, xs = [], []
        for t in range(LAGS + 1, len(periods) - h):
            if any(PANDEMIC[0] <= periods[k] <= PANDEMIC[1] for k in (t, t + h)):
                continue
            regressors = np.concatenate(
                ([shocks[t]], diffs[t - LAGS : t][::-1], shocks[t - LAGS : t][::-1])
            )
            if extra is not None:
                regressors = np.concatenate((regressors, extra[t - LAGS : t + 1][::-1]))
            target = values[t + h] - values[t - 1]
            if np.isnan(target) or np.isnan(regressors).any():
                continue
            ys.append(target)
            xs.append(regressors)
        if len(ys) < 2 * (3 * LAGS + 3):
            rows.append((h, np.nan, np.nan, len(ys)))
            continue
        fit = sm.OLS(np.array(ys), sm.add_constant(np.array(xs))).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1}
        )
        rows.append((h, float(fit.params[1]), float(fit.bse[1]), len(ys)))
    return rows


def show(title: str, rows) -> None:
    print(f"\n{title}")
    print(f"  {'m':>3}  {'response':>10}  {'se':>6}  {'t':>6}  {'n':>4}")
    for h, beta, se, n in rows:
        if np.isnan(beta):
            print(f"  {h:>3}  {'too few observations':>28}  {n:>4}")
        else:
            print(f"  {h:>3}  {beta:>+8.2f}pp  {se:>6.2f}  {beta / se:>+6.2f}  {n:>4}")


def main() -> None:
    shock = monthly_surprises()
    last = max(shock)

    with FredAdapter() as fred:
        series = fred.fetch_series("INTDSRINM193N", observation_start="1999-01-01")
    discount = {o.period: o.value for o in series.observations if o.value is not None}
    print(f"IMF discount rate for India: {min(discount)} to {max(discount)}")
    show(
        "INDIA'S DISCOUNT RATE, change (pp) after a 1pp Fed surprise, 2000-2022",
        local_projection(discount, shock, start=date(2000, 1, 1), end=last),
    )

    repo = ManualExcelAdapter(RBI_REPO_RATE).fetch(
        RBI_REPO_RATE.series_id, path=RBI_WORKBOOK, archive=False
    )
    repo_rate = {o.period: o.value for o in repo.observations if o.value is not None}
    print(f"\nRBI policy repo rate: {min(repo_rate)} to {max(repo_rate)}")
    show(
        "RBI REPO RATE, change (pp) after a 1pp Fed surprise, 2011-2023",
        local_projection(repo_rate, shock, start=date(2011, 1, 1), end=last),
    )

    cpi = ManualExcelAdapter(RBI_CPI_INFLATION).fetch(
        RBI_CPI_INFLATION.series_id, path=RBI_WORKBOOK, archive=False
    )
    inflation = {o.period: o.value for o in cpi.observations if o.value is not None}
    show(
        "RBI REPO RATE, controlling for Indian CPI inflation (current and six lags), 2012-2023",
        local_projection(repo_rate, shock, start=date(2012, 8, 1), end=last, control=inflation),
    )
    show(
        "RBI REPO RATE without the inflation control, same 2012-2023 sample",
        local_projection(repo_rate, shock, start=date(2012, 8, 1), end=last),
    )

    print("\nTHE MODEL'S COUNTERPART, RBI move per point of Fed move under the headline:")
    print(f"  default exchange coefficient 0.42 : {MODEL_RATIO_DEFAULT:+.2f}")
    print(f"  exchange coefficient floor 0.17   : {MODEL_RATIO_FLOOR:+.2f}")
    print("\nA positive, significant response means the RBI follows the Fed: the exchange")
    print("channel outweighs the demand channel, and the headline's sign holds. A negative")
    print("one would mean the demand channel wins and the transfer reverses.")
    print("Note: the surprise is scaled to a 1pp move in expected US rates a year ahead,")
    print("the Fed's own move over that year is of that order, so the response per 1pp")
    print("surprise is comparable to the model's ratio, not identical to it.")


if __name__ == "__main__":
    main()
