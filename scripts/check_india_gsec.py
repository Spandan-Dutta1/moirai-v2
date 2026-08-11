"""Does adding the ten year yield fix the output response?

The three variable VAR shows industrial production rising after a rate
hike, which is the output-side analogue of the price puzzle: the RBI
tightens when it expects strength, and a VAR that cannot see what the RBI
saw attributes the subsequent strength to the tightening.

The ten year G-Sec par yield is a candidate remedy. It embeds market
expectations of growth and inflation, which is roughly the information the
policy maker is acting on. Ordered after the policy rate, on the grounds
that bond yields reprice within minutes of a policy announcement while the
converse is not true within a month.

The sample is 170 observations. A four variable VAR at six lags estimates
25 parameters per equation, which is thin. If this does not resolve the
sign, that is the answer rather than a reason to keep adding variables.
"""

import warnings
from datetime import date

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_GSEC_10Y,
    RBI_IIP,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)
from moirai.engine.data_fabric.series.temporal import drop_missing, restrict

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
configure_logging("ERROR")

PATH = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"
START = date(2012, 1, 1)

SPECIFICATIONS = {
    "three variable": [RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE],
    "with 10y yield": [RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE, RBI_GSEC_10Y],
}

for label, specs in SPECIFICATIONS.items():
    prepared = []
    for spec in specs:
        series = ManualExcelAdapter(spec).ingest(PATH)
        prepared.append(prepare(drop_missing(restrict(series, start=START))))

    periods, data = align(prepared)
    names = tuple(item.series_id for item in prepared)

    print(f"\n{'=' * 72}")
    print(f"{label}   {[n.replace('in_', '') for n in names]}")
    print("=" * 72)

    for n_lags in (3, 6):
        try:
            var = estimate_var(data, names, n_lags=n_lags, periods=periods)
            report = diagnose(var, portmanteau_lags=n_lags + 12)
            model = identify_cholesky(var, ordering=names)
            irf = impulse_responses(model, horizon=24)
        except Exception as error:  # noqa: BLE001
            print(f"  {n_lags} lags: {error}")
            continue

        rate_index = names.index("in_repo_rate")
        output_index = names.index("in_iip_yoy")
        price_index = names.index("in_cpi_inflation")
        shock = "in_repo_rate_shock"

        output_h, output_peak = irf.peak("in_iip_yoy", shock)
        price_h, price_peak = irf.peak("in_cpi_inflation", shock)

        print(
            f"\n  {n_lags} lags  obs/param {var.n_observations / var.n_parameters:.1f}"
            f"  usable {report.is_usable}"
            f"  modulus {var.stability.max_modulus:.3f}"
        )
        print(
            f"    output response peak : {output_peak:+.4f} at h={output_h}  "
            f"[{'WRONG SIGN' if output_peak > 0 else 'correct'}]"
        )
        print(
            f"    price response peak  : {price_peak:+.4f} at h={price_h}  "
            f"[{'puzzle' if price_peak > 0 else 'correct'}]"
        )
        print(f"    {'h':>4}  {'output':>10}  {'prices':>10}")
        for h in (3, 6, 12, 24):
            print(
                f"    {h:>4}  {irf.responses[h, output_index, rate_index]:>10.4f}"
                f"  {irf.responses[h, price_index, rate_index]:>10.4f}"
            )