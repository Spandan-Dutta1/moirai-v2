"""An Indian monetary VAR from RBI data.

Note what the preparation layer does with these series. IIP and CPI arrive
as year on year growth rates, so their declared unit is PERCENT_CHANGE and
they are left alone. The repo rate is a level and is differenced. Nothing
here specifies that: it follows from the metadata declared at ingestion.
"""

from datetime import date

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import InformationCriterion, estimate_var, select_lag_order
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_IIP,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)
from moirai.engine.data_fabric.series.temporal import drop_missing, restrict

configure_logging("WARNING")

path = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"

# CPI Combined begins in 2012, so the common sample starts there.
START = date(2012, 1, 1)

specs = [RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE]
prepared = []
for spec in specs:
    series = ManualExcelAdapter(spec).ingest(path)
    series = drop_missing(restrict(series, start=START))
    result = prepare(series)
    prepared.append(result)
    print(
        f"{result.series_id:<20} {result.record.steps[0].value:<16} "
        f"n={len(result):<4} {result.record.reason}"
    )

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
print(f"\naligned: {periods[0]} to {periods[-1]}  ({data.shape[0]} obs)\n")

selection = select_lag_order(data, max_lags=12, variables=names)
print("lag order:", {c.value: selection.best(c) for c in InformationCriterion})

for n_lags in (selection.best(InformationCriterion.BIC), 6, 12):
    try:
        var = estimate_var(data, names, n_lags=n_lags, periods=periods)
        report = diagnose(var, portmanteau_lags=n_lags + 12)
    except Exception as error:  # noqa: BLE001
        print(f"  {n_lags:>2} lags: {error}")
        continue
    print(
        f"  {n_lags:>2} lags: usable={report.is_usable}  "
        f"modulus={var.stability.max_modulus:.3f}  "
        f"obs/param={var.n_observations / var.n_parameters:.1f}"
    )
    if report.critical_failures:
        print(f"           critical: {[o.name for o in report.critical_failures]}")