"""Does the banking layer survive data it was not calibrated on?

The pass-through functions were tuned until the model reproduced the RBI's
easing-cycle figures. That agreement is construction rather than evidence.
The tightening cycle was held out, so these are predictions the model can
fail, and it does.
"""

from moirai.core.logging import configure_logging
from moirai.engine.financial.commercial_banks import (
    EASING_TARGETS,
    INDIAN_BANKING_SYSTEM,
    evaluate_transmission,
    validate_out_of_sample,
)

configure_logging("ERROR")

system = INDIAN_BANKING_SYSTEM

print("=" * 74)
print("IN SAMPLE  the easing cycle the model was calibrated on")
print("=" * 74)
in_sample = evaluate_transmission(system)
print(f"  {in_sample['n_passed']}/{in_sample['n_checks']} within tolerance, "
      f"loss {in_sample['loss']:.4f}")
print()
print("  This agreement is by construction. The discretionary pass-through")
print("  coefficients were adjusted until the group aggregates landed near")
print("  these figures, so the model was never able to fail them.")
print()

print("=" * 74)
print("OUT OF SAMPLE  the tightening cycle, held out")
print("=" * 74)
print(f"  {EASING_TARGETS[0].source}")
print("  calibrated on the above; tested against May 2022 to November 2024")
print()

result = validate_out_of_sample(system)

for check in result["checks"]:
    mark = "PASS" if check["passed"] else "FAIL"
    print(f"  [{mark}] {check['name']}")
    print(f"         model    : {check['predicted_by_model']}")
    print(f"         observed : {check['observed']}")
    print(f"         {check['detail']}")
    print()

print(f"  {result['n_passed']}/{result['n_checks']} predictions survived")
print()

print("=" * 74)
print("WHAT THIS MEANS")
print("=" * 74)
print("  The asymmetry direction holds: banks do raise rates faster than")
print("  they cut them, and the model produces that without being told to.")
print()
print("  Two predictions fail. The public-private ordering reverses under")
print("  tightening and the model has no mechanism for it. Deposit")
print("  pass-through is off by a wide margin, because the model has no")
print("  liquidity state and cannot produce the late repricing that")
print("  happened once surplus liquidity drained.")
print()
print("  A model that passed everything after being tuned on half the data")
print("  would be more suspicious than one that fails two of three.")