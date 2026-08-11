"""Does the specification failure come from sample heterogeneity?

The full 1960-2019 sample spans the Great Inflation, the Volcker reserve
targeting experiment, the Great Moderation, the financial crisis and the
zero lower bound. A single linear VAR across all of that is one model
standing in for several. This compares samples rather than lag orders,
because adding lags did not help.
"""

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("ERROR")  # quiet: the comparison is the output

CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]

SAMPLES = {
    "full 1960-2019": ("1960-01-01", "2019-12-01"),
    "great moderation 1985-2007": ("1985-01-01", "2007-06-01"),
    "pre-Volcker 1960-1979": ("1960-01-01", "1979-06-01"),
    "post-crisis 2010-2019": ("2010-01-01", "2019-12-01"),
}

for label, (start, end) in SAMPLES.items():
    with FredAdapter() as fred:
        prepared = [
            prepare(
                fred.fetch_series(code, observation_start=start, observation_end=end)
            )
            for code in CODES
        ]

    periods, data = align(prepared)
    names = tuple(item.series_id for item in prepared)

    print(f"\n{'=' * 66}")
    print(f"{label}   ({data.shape[0]} observations)")
    print("=" * 66)

    for n_lags in (6, 12):
        try:
            var = estimate_var(data, names, n_lags=n_lags, periods=periods)
            report = diagnose(var, portmanteau_lags=n_lags + 12)
        except Exception as error:  # noqa: BLE001 - sample may be too short
            print(f"  {n_lags:>2} lags: could not estimate ({error})")
            continue

        critical = [o.name for o in report.critical_failures]
        verdict = "USABLE" if report.is_usable else "not usable"
        print(
            f"  {n_lags:>2} lags: {verdict:>10s}   "
            f"max modulus {var.stability.max_modulus:.3f}   "
            f"half life {var.stability.half_life:.1f}m"
        )
        if critical:
            print(f"           critical failures: {', '.join(critical)}")