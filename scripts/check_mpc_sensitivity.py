"""How much the headline depends on the MPC endpoints (ADR 017).

The high-liquidity MPC is sourced from Fagereng, Holm and Natvik (2021) at
0.45; the hand-to-mouth MPC is an assumption at 0.70, below the source's
estimate of one. This runs the headline chain across both endpoints on the
same data, population and path, so the answer can be reported with the
range its least certain behavioural parameters imply.

The cell at (0.70, 0.45) is the headline and must match run_pipeline.py.
"""

import warnings

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose
from moirai.engine.causal.identification import identify_cholesky
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.economy.behaviour import BehaviourParameters
from moirai.engine.economy.households import PopulationParameters, generate_population
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.commercial_banks import INDIAN_BANKING_SYSTEM
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import HEADLINE_SCENARIO, run_scenario

warnings.filterwarnings("ignore")
configure_logging("WARNING")

LOW = (0.70, 0.85, 1.00)
HIGH = (0.10, 0.30, 0.45)

with FredAdapter() as fred:
    prepared = [
        prepare(
            fred.fetch_series(code, observation_start="1985-01-01", observation_end="2007-06-01")
        )
        for code in ("INDPRO", "CPIAUCSL", "FEDFUNDS")
    ]
periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=12, periods=periods)
report = diagnose(var, portmanteau_lags=24)
irf = impulse_responses(identify_cholesky(var, ordering=names), horizon=36)

population = generate_population(PopulationParameters(n_households=200_000, seed=1))
banks = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)

spreads: dict[tuple[float, float], float] = {}
aggregates: dict[tuple[float, float], float] = {}
for low in LOW:
    for high in HIGH:
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
            behaviour=BehaviourParameters(mpc_low_wealth=low, mpc_high_wealth=high),
        )
        spreads[(low, high)] = result.spread * 100
        aggregates[(low, high)] = result.aggregate_consumption_change * 100


def table(title: str, values: dict[tuple[float, float], float], unit: str) -> None:
    print(f"\n{title}")
    print("  hand-to-mouth MPC  " + "".join(f"  high={h:.2f}" for h in HIGH))
    for low in LOW:
        row = "".join(f"{values[(low, h)]:>10.3f}{unit}" for h in HIGH)
        marker = "   <- default row" if low == 0.70 else ""
        print(f"  {low:>17.2f}  {row}{marker}")


print(f"headline scenario: {HEADLINE_SCENARIO.name}")
print("default: hand-to-mouth 0.70 (assumed), high-liquidity 0.45 (sourced)")
table("SPREAD, floating borrowers minus savers (pp)", spreads, "")
table("AGGREGATE CONSUMPTION (%)", aggregates, "%")

sourced = [spreads[(low, 0.45)] for low in LOW]
print(
    f"\nwith the sourced high-liquidity MPC, the spread lies in "
    f"[{min(sourced):.2f}, {max(sourced):.2f}] pp across the hand-to-mouth range"
)
everything = list(spreads.values())
print(f"across the whole grid it lies in [{min(everything):.2f}, {max(everything):.2f}] pp")
