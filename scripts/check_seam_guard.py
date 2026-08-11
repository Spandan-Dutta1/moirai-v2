"""Confirm the guard catches the unit error that got through."""

import numpy as np

from moirai.core.exceptions import EngineError
from moirai.core.logging import configure_logging
from moirai.engine.causal.irf import ImpulseResponse
from moirai.engine.causal.preparation import Transformation
from moirai.engine.economy.shock_path import (
    MacroVariable,
    VariableMapping,
    build_shock_path,
)

configure_logging("ERROR")

# An impulse response in percentage points, as RBI series produce.
responses = np.zeros((13, 1, 1))
responses[:, 0, 0] = np.linspace(-0.16, 0.0, 13)
irf = ImpulseResponse(
    variables=("inflation",),
    shock_names=("rate_shock",),
    horizon=12,
    responses=responses,
)

wrong = VariableMapping(
    var_variable="inflation",
    macro_variable=MacroVariable.INFLATION,
    transformation=Transformation.NONE,
    scale=12.0,          # the bug: annualising an already annual rate
    baseline=0.044,
)
right = wrong.model_copy(update={"scale": 0.01})

print("with the wrong scale:")
try:
    build_shock_path(irf, "rate_shock", (wrong,))
    print("  NOT CAUGHT")
except EngineError as error:
    print(f"  caught: {str(error)[:110]}...")

print("\nwith the correct scale:")
path = build_shock_path(irf, "rate_shock", (right,))
print(f"  inflation path: {path.get(MacroVariable.INFLATION)[0]:.4f} "
      f"to {path.get(MacroVariable.INFLATION)[-1]:.4f}")

print("\noverride still available:")
path = build_shock_path(irf, "rate_shock", (wrong,), check_plausibility=False)
print(f"  built anyway, peak {np.abs(path.get(MacroVariable.INFLATION)).max():.2f}")