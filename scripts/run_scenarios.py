"""Four named scenarios through the full chain."""

import warnings

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.economy.calibration import evaluate
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import DEFAULT_SCENARIOS, compare, run_scenario

warnings.filterwarnings("ignore")
configure_logging("ERROR")

# ---- estimate once, shared by every scenario -----------------------------
with FredAdapter() as fred:
    prepared = [
        prepare(
            fred.fetch_series(
                code, observation_start="1985-01-01", observation_end="2007-06-01"
            )
        )
        for code in ("INDPRO", "CPIAUCSL", "FEDFUNDS")
    ]

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=12, periods=periods)
report = diagnose(var, portmanteau_lags=24)
irf = impulse_responses(identify_cholesky(var, ordering=names), horizon=36)

banks = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
population = generate_population(PopulationParameters(n_households=200_000, seed=1))
calibration = evaluate(population)

print(f"shared estimation: VAR({var.n_lags}), diagnostics usable {report.is_usable}")
print(f"network: {len(banks)} central banks, 12 commercial banks")
print(f"households: {len(population):,}, calibration loss {calibration.loss():.3f}")
print()

results = []
for scenario in DEFAULT_SCENARIOS:
    print("=" * 76)
    print(f"{scenario.name.upper().replace('_', ' ')}")
    print("=" * 76)
    print(f"  {scenario.description}")
    print(f"  {scenario.rationale}")
    print()

    result = run_scenario(
        scenario,
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
        calibration_loss=calibration.loss(),
        allow_extreme=True,
    )
    results.append(result)

    print(f"  {'bank':<24} {'rate':>9} {'move':>9}")
    banks_by_name = {b.name: b for b in scenario.apply_to(banks)}
    for name, rate in result.equilibrium.rates.items():
        move = (rate - banks_by_name[name].current_rate) * 10_000
        print(f"  {name:<24} {rate:>8.3%} {move:>+8.0f}bp")

    print()
    print(f"  shock: {result.shock.scale:.1f} standard deviations from "
          f"{result.scenario.shock_origin}")
    print(f"  banks pass {result.lending_rate_change_bp:+.0f}bp to borrowers, "
          f"{result.deposit_rate_change_bp:+.0f}bp to savers "
          f"(wedge {result.bank_wedge_bp:.0f}bp)")
    print()
    print(f"  aggregate consumption : {result.aggregate_consumption_change * 100:+.3f}%")
    print(f"  floating borrowers    : {result.borrower_change * 100:+.3f}%")
    print(f"  net savers            : {result.saver_change * 100:+.3f}%")
    print(f"  spread                : {result.spread * 100:.2f}pp")
    print(f"  extra job losses      : {result.extra_job_losses:,}")
    print()

print("=" * 76)
print("COMPARISON")
print("=" * 76)
summary = compare(tuple(results))
print(f"  {'scenario':<22} {'aggregate':>10} {'spread':>9} {'scale':>7} {'wedge':>8}")
for name, values in summary["by_scenario"].items():
    print(
        f"  {name:<22} {values['aggregate']:>9.3f}% {values['spread']:>8.2f}pp "
        f"{values['shock_scale']:>7.1f} {values['bank_wedge_bp']:>7.0f}bp"
    )
print()
print(f"  widest spread    : {summary['widest_spread']}")
print(f"  largest aggregate: {summary['largest_aggregate']}")