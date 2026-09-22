"""What does adding variables cost?

The current specification is three variables and twelve lags on 269
observations, which is 37 parameters per equation and about 7 observations
each. Adding a variable adds twelve parameters per equation. The question
is whether the extra information is worth the extra estimation burden, and
the honest way to find out is to try it and read the diagnostics.

Two candidates, both defensible on economic grounds:

  credit        connects the policy rate to household debt service, which
                is the project's actual research question
  exchange rate matters for an open economy through imported inflation and
                capital flows, and its absence is a stated limitation
"""

import warnings

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

warnings.filterwarnings("ignore")
configure_logging("ERROR")

START, END = "1985-01-01", "2007-06-01"

SPECIFICATIONS = {
    "3-var baseline": ["INDPRO", "CPIAUCSL", "FEDFUNDS"],
    "4-var + credit": ["INDPRO", "CPIAUCSL", "TOTALSL", "FEDFUNDS"],
    "4-var + dollar": ["INDPRO", "CPIAUCSL", "FEDFUNDS", "TWEXBMTH"],
    "5-var both": ["INDPRO", "CPIAUCSL", "TOTALSL", "FEDFUNDS", "TWEXBMTH"],
}
for label, codes in SPECIFICATIONS.items():
    print(f"\n{'=' * 72}")
    print(f"{label}   {codes}")
    print("=" * 72)

    try:
        with FredAdapter() as fred:
            prepared = [
                prepare(
                    fred.fetch_series(
                        code, observation_start=START, observation_end=END
                    )
                )
                for code in codes
            ]
    except Exception as error:  # noqa: BLE001
        print(f"  could not fetch: {error}")
        continue

    periods, data = align(prepared)
    names = tuple(item.series_id for item in prepared)
    print(f"  sample: {periods[0]} to {periods[-1]}  ({data.shape[0]} obs)")

    for n_lags in (6, 12):
        try:
            var = estimate_var(data, names, n_lags=n_lags, periods=periods)
            report = diagnose(var, portmanteau_lags=n_lags + 12)
        except Exception as error:  # noqa: BLE001
            print(f"  {n_lags:>2} lags: {str(error)[:90]}")
            continue

        ratio = var.n_observations / var.n_parameters
        print(
            f"  {n_lags:>2} lags: {var.n_parameters:>2} params/eqn, "
            f"{ratio:>4.1f} obs/param, "
            f"usable {str(report.is_usable):<5} "
            f"modulus {var.stability.max_modulus:.3f}"
        )
        if report.critical_failures:
            print(f"           critical: {[o.name for o in report.critical_failures]}")

        # Does the output response keep its sign and rough magnitude?
        try:
            model = identify_cholesky(var, ordering=names)
            irf = impulse_responses(model, horizon=36).cumulate()
            rate_index = names.index("fedfunds")
            output_index = names.index("indpro")
            shock = "fedfunds_shock"
            h, peak = irf.peak("indpro", shock)
            print(f"           output response peak {peak * 100:+.3f}% at h={h}")
        except Exception as error:  # noqa: BLE001
            print(f"           IRF failed: {str(error)[:70]}")