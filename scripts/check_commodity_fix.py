"""Does adding commodity prices resolve the price puzzle?

The Fed watches commodity prices as a leading indicator of inflation. A VAR
that omits them cannot see why the Fed tightened, so it attributes the
inflation that prompted the tightening to the tightening itself. Sims
(1992) proposed including a commodity index to give the VAR the same
forward-looking information the policymaker had.

Ordering places commodities first: they clear in global auction markets
within minutes, so under a recursive assumption they can affect everything
contemporaneously while nothing affects them within the month.
"""

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("ERROR")

START, END = "1985-01-01", "2007-06-01"
N_LAGS = 12
HORIZON = 36

SPECIFICATIONS = {
    "three variable": ["INDPRO", "CPIAUCSL", "FEDFUNDS"],
    "with commodities": ["PPIACO", "INDPRO", "CPIAUCSL", "FEDFUNDS"],
}

for label, codes in SPECIFICATIONS.items():
    with FredAdapter() as fred:
        prepared = [
            prepare(
                fred.fetch_series(code, observation_start=START, observation_end=END)
            )
            for code in codes
        ]

    periods, data = align(prepared)
    names = tuple(item.series_id for item in prepared)

    var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)
    report = diagnose(var, portmanteau_lags=N_LAGS + 12)
    model = identify_cholesky(var, ordering=names)
    irf = impulse_responses(model, horizon=HORIZON)

    print(f"\n{'=' * 70}")
    print(f"{label}   {list(names)}")
    print(f"{'=' * 70}")
    print(f"  observations : {var.n_observations}")
    print(f"  usable       : {report.is_usable}")
    print(f"  max modulus  : {var.stability.max_modulus:.3f}")
    print()

    cumulative = irf.cumulate()
    print("  cumulative response to a monetary tightening (level, percent)")
    print(f"  {'h':>4}  " + "  ".join(f"{n[:9]:>10}" for n in names))
    for h in (6, 12, 24, 36):
        row = "  ".join(
            f"{cumulative.responses[h, i, len(names) - 1] * 100:>10.3f}"
            for i in range(len(names))
        )
        print(f"  {h:>4}  {row}")
    print()

    price_index = names.index("cpiaucsl")
    output_index = names.index("indpro")
    shock = f"{names[-1]}_shock"

    price_peak_h, price_peak = irf.cumulate().peak("cpiaucsl", shock)
    output_peak_h, output_peak = irf.cumulate().peak("indpro", shock)

    puzzle = "PRESENT" if price_peak > 0 else "resolved"
    sign = "wrong" if output_peak > 0 else "correct"

    print(f"  price puzzle    : {puzzle}  (peak {price_peak * 100:+.3f}% at h={price_peak_h})")
    print(f"  output response : {sign}  (peak {output_peak * 100:+.3f}% at h={output_peak_h})")