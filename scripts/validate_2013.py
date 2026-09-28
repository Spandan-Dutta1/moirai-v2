"""The network against calendar 2013, the taper tantrum year (ADR 013).

The same fixed design as 2022, in historical_validation.py, with nothing
tuned. Every condition is computed from index levels except India's,
which is CPI Combined as the RBI publishes it, year on year.

What the design can and cannot test. It conditions each bank on its
inflation. The tantrum, from May 2013, was a shock to expected Fed policy
and term premia with no move in the Fed's rate, and the RBI's main
defence was a 200bp rise in its marginal standing facility rate, not the
repo rate. Neither is expressible here. So this tests whether the network
explains 2013's rate decisions, not whether it reproduces the tantrum.

Three anachronisms are carried as declared, not corrected: the RBI had no
inflation target until 2015-16, the ECB's target was "below but close to"
two percent, and the model has no lower bound.
"""

from datetime import date

from historical_validation import (
    FredIndex,
    FredRate,
    HistoricalValidation,
    WarehouseRate,
    WarehouseYearOnYear,
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
    HISTORICAL_2013,
    JANUARY_2013_BANKS,
    OBSERVED_2013_MOVES_BP,
)

START, END = date(2013, 1, 1), date(2013, 12, 31)

run(
    HistoricalValidation(
        year=2013,
        scenario=HISTORICAL_2013,
        banks=JANUARY_2013_BANKS,
        observed_bp=OBSERVED_2013_MOVES_BP,
        inflation={
            FED.name: FredIndex("PCEPI"),
            ECB.name: FredIndex("CP0000EZ17M086NEST"),
            BANK_OF_ENGLAND.name: FredIndex("GBRCPIALLMINMEI"),
            BANK_OF_JAPAN.name: FredIndex("JPNCPIALLMINMEI"),
            RBI.name: WarehouseYearOnYear(
                "in_cpi_inflation",
                "CPI Combined as the RBI publishes it; the RBI had no CPI target in 2013",
            ),
        },
        start={
            FED.name: FredRate("DFEDTARU", START),
            ECB.name: FredRate("ECBDFR", START),
            BANK_OF_ENGLAND.name: FredRate("BOERUKM", START),
            BANK_OF_JAPAN.name: FredRate("IRSTCI01JPM156N", START),
            RBI.name: WarehouseRate("in_repo_rate", date(2012, 12, 1)),
        },
        end={
            FED.name: FredRate("DFEDTARU", END),
            ECB.name: FredRate("ECBDFR", END),
            BANK_OF_ENGLAND.name: FredRate("BOERUKM", END),
            BANK_OF_JAPAN.name: FredRate("IRSTCI01JPM156N", END),
            RBI.name: WarehouseRate("in_repo_rate", date(2013, 12, 1)),
        },
        cross_checks=(
            ("HICP, 19-member euro area", "CP0000EZ19M086NEST"),
            ("OECD India CPI, a different measure", "INDCPIALLMINMEI"),
        ),
        notes=(
            "ECB: the main refinancing rate fell 50bp, 0.75% to 0.25%; the deposit "
            "rate stayed at zero",
            "RBI: -25bp nets three 25bp cuts, January to May, against two 25bp hikes "
            "in September and October; the MSF rate rose 200bp in July and was "
            "unwound by October",
            "Fed: no rate move; the tantrum was a shock to expected policy",
        ),
    )
)
