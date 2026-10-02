"""How strongly the model's RBI follows the Fed, and the calibration (ADR 027).

check_rbi_fed_ratio.py measures the RBI moving at least 0.47 points per
point of expected US policy. This shows the model cannot reach that with
the RBI's external objective tied to the simple average of the other four
banks, that weighting it by invoicing currency removes the ceiling, and the
external weight at which the calibrated dynamic RBI meets the floor.

Needs no data beyond the declared parameters.
"""

import numpy as np
from scipy.optimize import brentq

from moirai.core.logging import configure_logging
from moirai.engine.dynamic_chain import calibrated_inputs
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.dynamic_game import (
    CALIBRATED_INERTIA_WEIGHT,
    CALIBRATED_RBI_EXTERNAL_WEIGHT,
    RBI_FOLLOWING_FLOOR,
    DynamicParameters,
    dynamic_nash,
)
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix, network_nash
from moirai.engine.scenarios import HEADLINE_SCENARIO

configure_logging("WARNING")

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
EQUAL = SpilloverMatrix.from_literature(DEFAULT_TIERS)
DOLLAR = calibrated_inputs(BANKS, EQUAL)[1]
INERTIA = DynamicParameters(inertia_weight=CALIBRATED_INERTIA_WEIGHT)
MEASURED_SHAPE = np.array([1.33, 3.06, 4.99, 4.97, 5.06])


def with_weight(weight: float) -> tuple:
    return tuple(
        b.model_copy(update={"external_weight": weight}) if b.name == RBI.name else b for b in BANKS
    )


def ratios(weight: float, matrix: SpilloverMatrix) -> tuple[float, float, np.ndarray]:
    banks = with_weight(weight)
    s = network_nash(HEADLINE_SCENARIO.apply_to(banks), matrix).rates
    r = network_nash(banks, matrix).rates
    static = (s[RBI.name] - r[RBI.name]) / (s[FED.name] - r[FED.name])
    shocked = dynamic_nash(HEADLINE_SCENARIO.apply_to(banks), matrix, parameters=INERTIA)
    reference = dynamic_nash(banks, matrix, parameters=INERTIA)
    caused = np.asarray(shocked.moves) - np.asarray(reference.moves)
    fed = caused[:, shocked.index_of(FED.name)]
    rbi = caused[:, shocked.index_of(RBI.name)]
    return static, float(rbi.max() / fed.max()), rbi * 10_000


def main() -> None:
    print(f"measured floor: the RBI moves at least {RBI_FOLLOWING_FLOOR} per point of US policy\n")
    print("RBI move per point of Fed move, by the RBI's external weight")
    print(f"  {'weight':>7}  {'equal reference':>18}  {'dollar reference':>18}")
    print(f"  {'':>7}  {'static':>8} {'dynamic':>9}  {'static':>8} {'dynamic':>9}")
    for weight in (0.4, 1.0, 2.0, 5.0, 10.0, 50.0):
        es, ed, _ = ratios(weight, EQUAL)
        ds, dd, _ = ratios(weight, DOLLAR)
        print(f"  {weight:>7.1f}  {es:>8.2f} {ed:>9.2f}  {ds:>8.2f} {dd:>9.2f}")
    print("  With the others weighted equally the RBI cannot follow the Fed by more")
    print("  than about 0.3 at any weight: a Fed move counts a quarter in the average.")

    solved = brentq(lambda w: ratios(w, DOLLAR)[1] - RBI_FOLLOWING_FLOOR, 0.4, 50.0, xtol=1e-6)
    _, dynamic, path = ratios(CALIBRATED_RBI_EXTERNAL_WEIGHT, DOLLAR)
    shape = path[:5] / np.abs(path[:5]).max()
    distance = float(((shape - MEASURED_SHAPE / MEASURED_SHAPE.max()) ** 2).sum())
    print(
        f"\nweight meeting the floor with the dollar reference: {solved:.2f} "
        f"(declared constant {CALIBRATED_RBI_EXTERNAL_WEIGHT})"
    )
    print(f"  dynamic RBI follows the Fed by {dynamic:.2f}")
    print("  RBI caused path, quarters 0-7 (bp): " + " ".join(f"{v:+.1f}" for v in path[:8]))
    print(f"  shape against the measured response: {distance:.3f} (ADR 026 without it: 0.039)")


if __name__ == "__main__":
    main()
