"""The network against calendar 2022, with nothing tuned (ADR 012).

Declares where every number in `historical_2022` comes from and runs the
fixed design in historical_validation.py. Two sources are not computed:

    Japan   FRED's OECD CPI stops in mid-2021, so the 2022 average
            (2.5 percent, Statistics Bureau, all items) is declared.
    UK      Bank Rate is not on FRED for 2022, so the published rates
            are declared and SONIA is printed as a cross-check.
"""

from datetime import date

from historical_validation import (
    DeclaredInflation,
    DeclaredRate,
    FredIndex,
    FredRate,
    HistoricalValidation,
    WarehouseRate,
    WarehouseYearOnYear,
    fetch,
    run,
)

from moirai.engine.financial.central_banks import (
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    ECB,
    FED,
    RBI,
)
from moirai.engine.scenarios import (
    HISTORICAL_2022,
    JANUARY_2022_BANKS,
    OBSERVED_2022_MOVES_BP,
)

START, END = date(2022, 1, 1), date(2022, 12, 31)

sonia = (
    fetch("IUDSOIA", "2021-12-01", START.isoformat())[-1][1],
    fetch("IUDSOIA", "2022-12-01", END.isoformat())[-1][1],
)

run(
    HistoricalValidation(
        year=2022,
        scenario=HISTORICAL_2022,
        banks=JANUARY_2022_BANKS,
        observed_bp=OBSERVED_2022_MOVES_BP,
        inflation={
            FED.name: FredIndex("PCEPI"),
            ECB.name: FredIndex("CP0000EZ19M086NEST"),
            BANK_OF_ENGLAND.name: FredIndex("GBRCPIALLMINMEI"),
            BANK_OF_JAPAN.name: DeclaredInflation(
                "Statistics Bureau of Japan, CPI all items; FRED's JPNCPIALLMINMEI "
                "ends 2021-06"
            ),
            RBI.name: WarehouseYearOnYear(
                "in_cpi_inflation",
                "CPI Combined as the RBI publishes it, year on year (no index held)",
            ),
        },
        start={
            FED.name: FredRate("DFEDTARU", START),
            ECB.name: FredRate("ECBDFR", START),
            BANK_OF_ENGLAND.name: DeclaredRate(0.0025, "Bank Rate, Bank of England"),
            BANK_OF_JAPAN.name: FredRate("IRSTCI01JPM156N", START),
            RBI.name: WarehouseRate("in_repo_rate", date(2021, 12, 1)),
        },
        end={
            FED.name: FredRate("DFEDTARU", END),
            ECB.name: FredRate("ECBDFR", END),
            BANK_OF_ENGLAND.name: DeclaredRate(0.0350, "Bank Rate, Bank of England"),
            BANK_OF_JAPAN.name: FredRate("IRSTCI01JPM156N", END),
            RBI.name: WarehouseRate("in_repo_rate", date(2022, 12, 1)),
        },
        cross_checks=(("OECD India CPI, a different measure", "INDCPIALLMINMEI"),),
        notes=(
            f"SONIA cross-check for Bank Rate: {sonia[0]:.3f}% to {sonia[-1]:.3f}%",
            "BoJ: the call rate drifted; its policy balance rate stayed at -0.10%",
        ),
    )
)
