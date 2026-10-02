"""Policy inertia in the dynamic game, calibrated and checked (ADR 026).

The dynamic game of ADR 023 makes the RBI's caused move largest in the
first quarter and then fade, the opposite of what India's repo rate does
after a Fed surprise: it rises for about two quarters and stays up
(ADR 022). Nothing in that game makes a bank move gradually. Policy
inertia does: a cost on changing the rate from one quarter to the next.

Calibration uses a published estimate, not the Indian data: the weight is
set so the Fed's equilibrium rule puts 0.79 on its own previous rate,
Clarida, Gali and Gertler's (2000) estimate. The RBI's path is then
compared with its measured response as a check, on shape only, since
ADR 022 explains why the measured magnitudes cannot be taken at face value.

Needs no data beyond the declared parameters.
"""

import numpy as np
from scipy.optimize import brentq

from moirai.core.logging import configure_logging
from moirai.engine.financial.central_banks import BANK_OF_ENGLAND, BANK_OF_JAPAN, ECB, FED, RBI
from moirai.engine.financial.dynamic_game import (
    CALIBRATED_INERTIA_WEIGHT,
    FED_SMOOTHING_TARGET,
    DynamicParameters,
    dynamic_nash,
    smoothing_coefficient,
)
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix
from moirai.engine.scenarios import HEADLINE_SCENARIO

configure_logging("WARNING")

BANKS = (FED, ECB, BANK_OF_JAPAN, BANK_OF_ENGLAND, RBI)
MATRIX = SpilloverMatrix.from_literature(DEFAULT_TIERS)

# RBI policy rate response to a Fed surprise at 1, 3, 6, 9 and 12 months,
# read as quarters 0 to 4 (scripts/check_rbi_response_to_fed.py, ADR 022).
MEASURED = {
    "repo 2012-2023, Indian CPI controlled": np.array([1.33, 3.06, 4.99, 4.97, 5.06]),
    "repo 2012-2023, uncontrolled": np.array([0.96, 2.33, 3.45, 3.33, 2.94]),
    "discount rate 2000-2022": np.array([0.39, 0.97, 0.75, 0.59, 0.09]),
}


def caused(weight: float, horizon: int = 12) -> tuple[np.ndarray, np.ndarray, float, float]:
    params = DynamicParameters(inertia_weight=weight)
    shocked = dynamic_nash(
        HEADLINE_SCENARIO.apply_to(BANKS), MATRIX, parameters=params, horizon=horizon
    )
    reference = dynamic_nash(BANKS, MATRIX, parameters=params, horizon=horizon)
    moves = (np.asarray(shocked.moves) - np.asarray(reference.moves)) * 10_000
    return (
        moves[:, shocked.index_of(FED.name)],
        moves[:, shocked.index_of(RBI.name)],
        smoothing_coefficient(reference, FED.name),
        smoothing_coefficient(reference, RBI.name),
    )


def shape_distance(path: np.ndarray, measured: np.ndarray) -> float:
    model = path[:5] / np.abs(path[:5]).max()
    target = measured / np.abs(measured).max()
    return float(((model - target) ** 2).sum())


def main() -> None:
    print("INERTIA WEIGHT AND THE SMOOTHING IT IMPLIES")
    print(f"  {'weight':>7} {'Fed rho':>8} {'RBI rho':>8}")
    for weight in (0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0):
        _, _, rho_fed, rho_rbi = caused(weight, horizon=4)
        print(f"  {weight:>7.1f} {rho_fed:>8.2f} {rho_rbi:>8.2f}")

    solved = brentq(lambda w: caused(w, horizon=4)[2] - FED_SMOOTHING_TARGET, 0.5, 30.0, xtol=1e-6)
    print(
        f"\n  weight giving the Fed rho = {FED_SMOOTHING_TARGET}: {solved:.2f} "
        f"(declared constant {CALIBRATED_INERTIA_WEIGHT})"
    )

    print("\nTHE RBI'S CAUSED PATH, quarters 0 to 7 (bp)")
    for label, weight in (
        ("no inertia (ADR 023)", 0.0),
        ("calibrated inertia", CALIBRATED_INERTIA_WEIGHT),
    ):
        fed, rbi, _, _ = caused(weight)
        print(f"  {label:<22} RBI " + " ".join(f"{v:+5.1f}" for v in rbi[:8]))
        print(f"  {'':<22} Fed " + " ".join(f"{v:+5.0f}" for v in fed[:8]))

    print("\nSHAPE AGAINST THE MEASURED RESPONSE (sum of squared gaps, normalised)")
    print(f"  {'measured series':<40} {'no inertia':>11} {'calibrated':>11}")
    plain, calibrated = caused(0.0)[1], caused(CALIBRATED_INERTIA_WEIGHT)[1]
    for label, measured in MEASURED.items():
        print(
            f"  {label:<40} {shape_distance(plain, measured):>11.3f} "
            f"{shape_distance(calibrated, measured):>11.3f}"
        )

    fed, rbi, _, _ = caused(CALIBRATED_INERTIA_WEIGHT)
    print("\nMAGNITUDE")
    print(
        f"  peak RBI move / peak Fed move: {rbi.max() / fed.max():.2f} with inertia, "
        f"{caused(0.0)[1].max() / caused(0.0)[0].max():.2f} without"
    )
    print("  With realistic inertia both banks respond gradually to a shock that fades,")
    print("  so both move less. The shape now matches the RBI's measured response; the")
    print("  size does not, and ADR 022 found the real RBI follows the Fed by more than")
    print("  the model's. ADR 026 records this as the remaining gap.")


if __name__ == "__main__":
    main()
