# 17. The MPC endpoints are sourced or declared

Date: 2026-09-30
Status: Accepted

## Context

Household consumption responds to income through a marginal propensity to
consume that falls smoothly with liquid wealth, from `mpc_low_wealth` for a
household with no buffer to `mpc_high_wealth` for one with ample savings.
The two endpoints set how hard a rate change bites on each kind of
household.

The high endpoint was 0.10. The class docstring attributed it to "the
empirical MPC literature", with high-wealth estimates "around 0.05 to
0.15", and named no source. The project's session record went further: it
said the value had been moved to 0.30 on the authority of Fagereng, Holm
and Natvik (2021), with the citation stored in the code. Neither change was
ever committed. The code still read 0.10 and carried no citation.

The record's 0.30 did not match its own source either. The published
abstract of Fagereng, Holm and Natvik (2021) reports that low-liquidity
winners of the smallest prizes spend all of it within the year, and that
high-liquidity winners of large prizes spend slightly below one-half. The
record set 0.70 and 0.30.

## Decision

- `mpc_high_wealth` defaults to `MPC_HIGH_LIQUIDITY_FHN = 0.45`, the reading
  of "slightly below one-half", with the full citation in `FHN_2021`.
- `mpc_low_wealth` stays at 0.70, renamed `MPC_LOW_LIQUIDITY_ASSUMED` and
  described as an assumption below the source's estimate of one. The source
  measures spending out of the smallest one-off windfalls; the model applies
  this endpoint to every shock a hand-to-mouth household receives.
- Both field descriptions say where the value comes from, and the class
  docstring no longer cites a range nobody sourced.
- `scripts/check_mpc_sensitivity.py` runs the headline across hand-to-mouth
  MPCs of 0.70, 0.85 and 1.00 and high-liquidity MPCs of 0.10, 0.30 and
  0.45.
- Four tests pin the default to the sourced value, the low endpoint to its
  declared assumption, the gradient's direction, and the field descriptions.

## Two reasons 0.45 is a floor rather than a point

Both are stated where the constant is defined.

- The source measures spending out of a one-off windfall. The shocks in this
  model are persistent changes in debt service and interest income, and
  consumption responds more to persistent income changes than to transitory
  ones.
- The source is Norwegian. Indian households have thinner access to credit,
  which raises MPCs.

No Indian estimate of MPC by liquid wealth was found. The value is the best
available anchor for the gradient, not an Indian measurement.

## What moved

Every script that simulates households with default parameters:
`run_pipeline.py`, `run_pipeline_india.py`, `run_scenarios.py`,
`check_behaviour.py` and `check_layer_connection.py`.

The headline, measured on the real data before the change:

| High-liquidity MPC | Aggregate | Floating borrowers | Net savers | Spread |
|---|---|---|---|---|
| 0.10 (before) | -0.066% | -0.586% | -0.026% | -0.56pp |
| 0.30 (the record's claim) | -0.078% | -0.633% | -0.037% | -0.60pp |
| 0.45 (now) | -0.085% | -0.650% | -0.043% | -0.61pp |

The finding is robust to this parameter. Moving the endpoint more than
fourfold moves the spread by 0.05 points. The spread is carried by what
floating-rate borrowers pay, which does not depend on how liquid savers
spend.

## What did not move

The game, the network, the VAR, the banks, calibration, and every
historical validation. None of them simulate household consumption.

## Lesson

A record that says a change was made is not evidence that it was made.
The value in the code and the value in the citation are now pinned by a
test, so they cannot drift apart again without a failure.
