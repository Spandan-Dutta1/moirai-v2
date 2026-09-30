"""Identify the US policy shock with Fed surprises instead of an ordering (F10).

The headline's path takes its shape from the US VAR's response to a policy
shock, and that shock has been identified by a Cholesky ordering: output
and prices cannot respond to the funds rate within the month, and the Fed
responds to both. The ordering is untestable, and on this sample it
produces the price puzzle, prices rising after a tightening.

A proxy SVAR (Stock and Watson 2012, 2018; Mertens and Ravn 2013) uses an
external instrument instead: the Bauer-Swanson monthly surprise, which
moves with the policy shock and, by construction of the 30-minute window,
with nothing else. No ordering is assumed.

This script estimates the same VAR the pipeline uses, identifies the shock
both ways, and reports:

  1. the first stage, whether the instrument is strong enough to trust;
  2. the responses under each identification, normalised to the same
     peak funds rate move, since the chain rescales the shock to the move
     the game solves and only the shape carries through;
  3. the headline under each, run through the same chain.

Reads data and prints. Changes nothing.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import openpyxl

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky, identify_proxy
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import HEADLINE_SCENARIO, run_scenario

warnings.filterwarnings("ignore")
configure_logging("WARNING")

SURPRISES = get_paths().raw / "manual" / "frbsf_monetary_policy_surprises.xlsx"
START, END = "1985-01-01", "2007-06-01"
CODES = ("INDPRO", "CPIAUCSL", "FEDFUNDS")
N_LAGS = 12
HORIZON = 36


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


def normalised_path(irf, variable: str, shock: str) -> np.ndarray:
    """Cumulative response per 1pp of peak cumulative funds rate response."""
    cumulative = np.cumsum(irf.path(variable, shock))
    rate = np.cumsum(irf.path("fedfunds", shock))
    return cumulative / abs(rate[int(np.argmax(np.abs(rate)))])


def main() -> None:
    with FredAdapter() as fred:
        prepared = [
            prepare(fred.fetch_series(c, observation_start=START, observation_end=END))
            for c in CODES
        ]
    periods, data = align(prepared)
    names = tuple(p.series_id for p in prepared)
    var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)
    report = diagnose(var, portmanteau_lags=N_LAGS + 12)

    cholesky = identify_cholesky(var, ordering=names)
    proxy, first = identify_proxy(
        var, monthly_surprises(), "fedfunds", instrument_name="Bauer-Swanson MPS_ORTH"
    )

    print(f"VAR({var.n_lags}) on {', '.join(names)}, {periods[0]} to {periods[-1]}")
    print(f"diagnostics usable: {report.is_usable}")
    print("\nFIRST STAGE  funds rate residual on the monthly Fed surprise")
    print(f"  overlap       : {first.start} to {first.end}, {first.n_overlap} months")
    print(f"  coefficient   : {first.coefficient:+.3f}")
    print(f"  correlation   : {first.correlation:+.3f}")
    print(
        f"  robust F      : {first.f_statistic:.1f}  "
        f"({'WEAK, below 10' if first.is_weak else 'strong enough, above 10'})"
    )

    irfs = {
        "cholesky": impulse_responses(cholesky, horizon=HORIZON),
        "proxy": impulse_responses(proxy, horizon=HORIZON),
    }
    print("\nRESPONSES per 1pp peak funds rate move, cumulative, percent of level")
    print(f"  {'h':>3}  {'output':>18}  {'prices':>18}  {'funds rate':>18}")
    print(f"  {'':>3}  {'cholesky  proxy':>18}  {'cholesky  proxy':>18}  {'cholesky  proxy':>18}")
    paths = {
        (label, variable): normalised_path(irf, variable, "fedfunds_shock")
        for label, irf in irfs.items()
        for variable in names
    }
    for h in (0, 3, 6, 12, 24, 36):
        cells = []
        for variable in names:
            unit = 1.0 if variable == "fedfunds" else 100.0
            c = paths[("cholesky", variable)][h] * unit
            p = paths[("proxy", variable)][h] * unit
            cells.append(f"{c:>+8.2f} {p:>+8.2f}")
        print(f"  {h:>3}  " + "  ".join(cells))
    prices_c = paths[("cholesky", "cpiaucsl")][12]
    prices_p = paths[("proxy", "cpiaucsl")][12]
    print(
        f"\n  price puzzle at 12 months: cholesky {'yes' if prices_c > 0 else 'no'}, "
        f"proxy {'yes' if prices_p > 0 else 'no'}"
    )

    population = generate_population(PopulationParameters(n_households=200_000, seed=1))
    banks = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    print("\nTHE HEADLINE under each identification")
    print(
        f"  {'':<10} {'shock SD':>9} {'aggregate':>10} {'floating':>10} {'savers':>9} {'spread':>8}"
    )
    for label, irf in irfs.items():
        result = run_scenario(
            HEADLINE_SCENARIO,
            banks,
            spillovers,
            irf,
            shock_name="fedfunds_shock",
            rate_variable="fedfunds",
            price_variable="cpiaucsl",
            output_variable="indpro",
            population=population,
            banking_system=INDIAN_BANKING_SYSTEM,
            diagnostics=report,
        )
        print(
            f"  {label:<10} {result.shock.scale:>9.2f} "
            f"{result.aggregate_consumption_change * 100:>+9.3f}% "
            f"{result.borrower_change * 100:>+9.3f}% {result.saver_change * 100:>+8.3f}% "
            f"{result.spread * 100:>7.2f}pp"
        )
    print("\n  The RBI's caused move is the same under both: the game does not use the")
    print("  VAR. Only the shape of the path households face changes.")


if __name__ == "__main__":
    main()
