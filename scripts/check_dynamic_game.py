"""The central bank game with lags, against the static game (F12, ADR 023).

The static network game makes each bank choose one rate, with transmission
in the same period. The dynamic game in financial/dynamic_game.py makes
each choose a rule, with rates reaching output after a quarter and
inflation mostly through output after that, and the static game as its
long-run limit. This compares the two under the headline scenario: the
move the US inflation condition causes, for the Fed and the RBI, quarter
by quarter, at both ends of the exchange coefficient's evidence range
(ADR 019), and across the persistence parameters the dynamic game assumes.

Needs no data: the game layer runs on declared parameters.
"""

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.dynamic_game import DynamicParameters, dynamic_nash
from moirai.engine.financial.network import (
    DEFAULT_TIERS,
    FED_TO_INDIA_EXCHANGE_FLOOR,
    SpilloverMatrix,
    network_nash,
)
from moirai.engine.scenarios import HEADLINE_SCENARIO

configure_logging("WARNING")

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
SHORT = {FED.name: "Fed", RBI.name: "RBI"}
QUARTERS = 8


def static_caused(matrix: SpilloverMatrix) -> dict[str, float]:
    shocked = network_nash(HEADLINE_SCENARIO.apply_to(BANKS), matrix).rates
    reference = network_nash(BANKS, matrix).rates
    return {name: (shocked[name] - reference[name]) * 10_000 for name in shocked}


def dynamic_caused(matrix: SpilloverMatrix, parameters=None) -> dict[str, np.ndarray]:
    shocked = dynamic_nash(HEADLINE_SCENARIO.apply_to(BANKS), matrix, parameters=parameters)
    reference = dynamic_nash(BANKS, matrix, parameters=parameters)
    moves = (np.asarray(shocked.moves) - np.asarray(reference.moves)) * 10_000
    return {name: moves[:, shocked.index_of(name)] for name in shocked.names}


def summary(path: np.ndarray) -> str:
    year = path[:4].mean()
    turns = next((q for q, v in enumerate(path) if v < 0), None)
    turn = f"eases from quarter {turns}" if turns is not None else "never eases"
    return f"first {path[0]:+6.1f}bp  first-year average {year:+6.1f}bp  {turn}"


def main() -> None:
    default = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    floor = default.with_exchange(RBI.name, FED.name, FED_TO_INDIA_EXCHANGE_FLOOR)

    for label, matrix in (("exchange 0.42 (default)", default), ("exchange 0.17 (floor)", floor)):
        static = static_caused(matrix)
        dynamic = dynamic_caused(matrix)
        print(f"\n=== {label}: the move the US inflation condition causes")
        print(f"  {'':<12} {'static':>8}   dynamic, by quarter")
        print(f"  {'':<12} {'':>8}   " + " ".join(f"{q:>6}" for q in range(QUARTERS)))
        for bank in (FED, RBI):
            path = dynamic[bank.name]
            print(
                f"  {SHORT[bank.name]:<12} {static[bank.name]:>+7.1f}bp  "
                + " ".join(f"{v:>+6.1f}" for v in path[:QUARTERS])
            )
        print(f"  RBI: {summary(dynamic[RBI.name])}")

    print("\n=== sensitivity of the RBI's caused path to the assumed persistence (default matrix)")
    print(f"  {'output':>7} {'inflation':>10}   RBI")
    for a_y in (0.80, 0.90, 0.95):
        for a_pi in (0.70, 0.80, 0.90):
            params = DynamicParameters(output_persistence=a_y, inflation_persistence=a_pi)
            path = dynamic_caused(default, params)[RBI.name]
            print(f"  {a_y:>7.2f} {a_pi:>10.2f}   {summary(path)}")

    print("\nIn the static game the RBI's caused move is one number. In the dynamic game")
    print("it is a path: the RBI tightens first, as the rupee weakens and imported")
    print("inflation arrives within a quarter, and eases later, as slower demand reaches")
    print("Indian output and then inflation. The two channels of ADR 021 play out in")
    print("sequence rather than netting to one number. ADR 022 found the RBI's repo rate")
    print("rising for months after a Fed tightening surprise.")


if __name__ == "__main__":
    main()
