"""Verify the solution concepts against a game with a known answer."""

from moirai.core.logging import configure_logging
from moirai.engine.financial.game import (
    Game,
    Player,
    cooperation_gain,
    cooperative_optimum,
    nash_equilibria,
    stackelberg_equilibrium,
    sustainable_by_repetition,
)

configure_logging("ERROR")

# Prisoner's dilemma: 0 is cooperate, 1 is defect.
# Payoffs (row, column): CC 3,3  CD 0,5  DC 5,0  DD 1,1
TABLE = {(0.0, 0.0): (3, 3), (0.0, 1.0): (0, 5), (1.0, 0.0): (5, 0), (1.0, 1.0): (1, 1)}

game = Game(
    title="prisoner's dilemma",
    players=(
        Player(name="A", actions=(0.0, 1.0), payoff=lambda p: TABLE[p][0]),
        Player(name="B", actions=(0.0, 1.0), payoff=lambda p: TABLE[p][1]),
    ),
)

print("NASH")
for outcome in nash_equilibria(game):
    print(f"  {outcome.actions}  payoffs {outcome.payoffs}  unique={outcome.is_unique}")
print("  expected: both defect (1, 1), payoffs 1 and 1\n")

cooperative = cooperative_optimum(game)
print("COOPERATIVE")
print(f"  {cooperative.actions}  total {cooperative.total_payoff}")
print("  expected: both cooperate, total 6\n")

gain = cooperation_gain(game)
print(f"COST OF NON-COOPERATION: {gain['gain']}  (6 - 2 = 4)\n")

print("SUSTAINABLE BY REPETITION")
for discount in (0.0, 0.4, 0.5, 0.9):
    result = sustainable_by_repetition(game, cooperative, discount)
    print(f"  discount {discount}: {result['sustainable']}")
detail = sustainable_by_repetition(game, cooperative, 0.9)["by_player"]["A"]
print(f"  minimum discount factor for A: {detail['minimum_discount_factor']}")
print("  expected 0.5: gain from defecting is 2, loss per period is 2\n")

print("STACKELBERG")
outcome = stackelberg_equilibrium(game, "A")
print(f"  {outcome.actions}  payoffs {outcome.payoffs}")
print("  expected: still both defect. Moving first does not help when")
print("  defecting is dominant, which is the point of the dilemma.")