# 15. The pairwise Nash solver flipped the foreign bank's exchange rate sign

Date: 2026-09-29
Status: Accepted

## Context

`central_banks.analytic_nash` solves a two-bank game exactly, by
substituting the linear transmission equations into each bank's quadratic
loss and solving the stacked first order conditions. `_loss_coefficients`
builds each bank's coefficients, and it carried a flip:

    sign = 1.0 if is_home else -1.0

The pair's orientation is already inside `differential` in `_losses_at`
and `build_policy_game`, where the home bank's inflation falls with
`home_move - foreign_move` and the foreign bank's rises with it. Written
from each bank's own side, both say the same thing: a bank's inflation
falls when it tightens by more than the other, because its currency
appreciates. The flip counted the orientation twice. The foreign bank was
solved against a reaction function in which relative tightening raised
its own inflation.

It only bites when the foreign bank has a positive external weight, since
that weight also switches the exchange rate channel on. In
`check_analytic_nash.py` the foreign bank is the RBI, at 0.40.

## How it showed

Three checks, each independent of the others:

- **The answer depended on the label.** `analytic_nash(FED, RBI)` gave the
  RBI 5.158 percent; `analytic_nash(RBI, FED)` gave 5.383. A game's
  equilibrium cannot depend on which player is called home.
- **The solved rate was not a best reply.** Holding the Fed at its solved
  rate, the RBI's loss under `_losses_at` is minimised at 5.387 percent,
  23bp from the solved 5.158.
- **The network solver disagreed.** `network_nash` on the same two-bank
  economy gives 4.824 and 5.383 percent. Its `_reaction_system` builds the
  exchange coefficients pair by pair with no side-dependent flip, and its
  best-reply test (ADR 011) holds to 0.0bp.

The output itself was the first clue. Under the US inflation shock the
Fed hiked 147bp and the RBI solved 6bp below its current rate, while the
script's closing lines say a Fed move forces an RBI response.

## Why the tests did not catch it

`test_the_analytic_and_grid_solutions_roughly_agree` promised agreement
within one grid increment and used a tolerance of 0.005, which is 50bp,
two increments. The pairwise-against-network test in `test_network.py`
used the same 50bp. The real gaps, 9bp against the grid and 22.5bp against
the network, passed both. Every pairwise test called the solver in the
same order, so the label dependence never ran.

ADR 011 drew the right lesson for the network solver, a test that every
solved rate is a best reply under the stated loss. It was not applied to
the pairwise solver.

## Decision

The exchange rate sign is 1.0 for both banks. `is_home` is kept in the
signature so callers are unchanged.

Four tests are added to `tests/unit/test_central_banks.py`:

- `test_each_solved_rate_is_a_best_reply`, across five pairs in both
  orders, to 1e-6.
- `test_the_equilibrium_does_not_depend_on_which_bank_is_home`, a
  metamorphic check: relabelling the players cannot change the answer.
- `test_the_pairwise_and_network_solvers_agree_exactly`, to 1e-6 rather
  than 50bp.
- `test_a_fed_tightening_moves_the_rbi_up_from_its_current_rate`.

On the old code nine of the ten new cases fail. On the new code all 1,193
tests pass, against 1,183 before.

The two 50bp tests are left as they are. They test something different,
closeness to a grid, and the new tests now cover what they could not.

## What moved

Only results computed by `analytic_nash`, and everything built on it:
`coordination_value`, `weight_sensitivity`, `sensitivity_report`, and the
scripts `check_analytic_nash.py`, `check_weight_sensitivity.py` and
`check_layer_connection.py`. The last was missed when this record was first
drafted: its Layer 1a calls `analytic_nash` directly. The side-by-side run
of every script on the same data caught it.
`analytic_cooperative` minimises the true loss and was already right; only
the Nash baseline it is compared against moved.

| Scenario | Nash before (Fed / RBI) | Nash after | RBI move before -> after | Gain from coordinating before -> after |
|---|---|---|---|---|
| Current conditions | 4.844 / 5.158 | 4.824 / 5.383 | -9bp -> +13bp | 0.000011 -> 0.0000003 |
| US inflation shock | 5.719 / 5.193 | 5.687 / 5.560 | -6bp -> +31bp | 0.000034 -> 0.000007 |
| Both above target | 5.658 / 5.891 | 5.602 / 6.530 | +64bp -> +128bp | 0.000135 -> 0.000014 |

- **The RBI's direction flips in two of three scenarios.** It now moves the
  way the Fed moves, consistent with `network_nash` and with ADR 012.
- **The coordination null result is stronger.** The gains fall by five to
  forty times. Someone still loses from coordinating in every case; in
  current conditions it is now the RBI, by a negligible amount.
- **The RBI external-weight sensitivity is no longer fragile.** Across
  0.10 to 0.80 the RBI's move was -17 to +4bp with a sign flip; it is now
  +29 to +33bp with none. The sensitivity report goes from 6 of 10 checks
  robust to 8 of 10.
- **`check_layer_connection.py` moves slightly.** The Fed's equilibrium
  rate goes from 5.719 to 5.687 percent, so the game's move is 144bp
  rather than 147bp, 3.6 standard deviations either way. Downstream,
  aggregate consumption goes from +0.335 to +0.327 percent, floating-rate
  borrowers from -1.654 to -1.755 percent, and the spread from -2.62 to
  -2.70 points. A smaller shock producing a larger loss for floating-rate
  borrowers is not a contradiction of the fix: job losses are drawn from
  a fixed random seed, and a small change in the hazard changes which
  households lose their jobs. The 0.1 point movement is the granularity of
  the household simulation, and is itself worth knowing.
- **The Fed output-weight finding survives**: 145bp before, 147bp after.
  That is the figure quoted in the Fed's `weight_note` and in
  `preferences.py`.

"Both above target" starts the RBI outside its band, so the analytic
solver is approximate there before and after, since it omits the band
penalty (ADR 011).

## What did not move

Every scenario, both pipelines, both historical validations, and ADRs 010
to 014. Confirmed by running all 37 scripts in the untouched repository
and in a copy with only this change, on the same data and the same library
versions: apart from the three scripts above, every difference was a
timestamp, a file path, or a network failure during one of the two runs.
They solve through `network.py`, which never had the flip.

## Noticed, not changed here

The sensitivity table shows the RBI's move jumping from +3bp at an
external weight of 0.00 to +29bp at 0.10. That is the external weight
switching the exchange rate channel on, a separate issue in the
transmission equations of both solvers, and it needs its own decision.
