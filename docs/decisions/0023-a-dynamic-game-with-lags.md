# 23. A dynamic game with lags

Date: 2026-09-30
Status: Accepted

## Context

The network game (ADRs 008, 010, 011) is static. Each central bank
chooses one rate, and the rate lowers inflation and output in the period
it is chosen. Monetary policy works with lags: the RBI's own research
cited in ADR 010 puts India's inflation response at three to five
quarters, and central banks set rules for responding to the economy
rather than one number. A static game cannot say how quickly the RBI
follows the Fed, only how far, and ADR 021 showed its two channels, the
rupee pushing the RBI to tighten and slower demand pushing it to ease,
net to a single number that hides their timing.

## Decision

`financial/dynamic_game.py` adds a dynamic linear-quadratic game beside
the static one. Nothing that exists is changed: the static game stays the
default for the scenarios and the pipeline.

**Transmission**, quarterly, following the backward-looking model of
Rudebusch and Svensson (1999): a rate move reaches output after a
quarter, inflation through output after that, and the exchange rate
reaches inflation after a quarter.

    y_i[t+1]  = a_y y_i[t] - s u_i[t] - sum_j d_ij u_j[t]
    pi_i[t+1] = a_pi pi_i[t] + k y_i[t] - sum_j e_ij (u_i[t] - u_j[t])

**Anchored to the static model.** s, k and the spillover terms are
scaled so that a permanent rate move has, in the long run, the static
game's own output, own inflation, exchange and demand effects. With
a_y = 0.9 and a_pi = 0.8, k = 0.133 and s = 0.12 per quarter, close to
Rudebusch and Svensson's 0.14 and 0.10 for the United States. The static
game is the dynamic game's long run, except in one respect where the
static game is internally inconsistent: there a foreign rate move lowers
home output without lowering home inflation; here inflation responds to
the output gap whatever caused it.

**Losses** are the static game's, discounted at 0.99 a quarter, with the
external term in rate moves (ADR 011 showed levels and moves differ by a
constant that cancels from every caused move).

**Solution**: feedback Nash. Each bank chooses a linear rule mapping every
economy's inflation and output gaps to its rate, taking the others' rules
as given, found by iterating each bank's discounted Riccati equation, the
method of QuantEcon's `nnash` extended to five players. The external
term, which depends on the other banks' moves, enters exactly through
their rules as a state cost and a cross term.

Thirteen tests check that a move reaches output after a quarter and own
inflation only through the exchange rate in the first quarter; that a
permanent move reproduces the static own, exchange and demand effects;
that every bank's rule is a best response (random perturbations of one
rule, the others held, never lower that bank's discounted loss); that the
answer does not depend on the order the banks are given; that nobody
moves when there are no gaps; and that under the headline scenario the
Fed tightens and unwinds and the RBI follows first.

`scripts/check_dynamic_game.py` compares the two games. It needs no data.

## Result

The move the US inflation condition causes, by quarter:

| | Static | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 |
|---|---|---|---|---|---|---|---|---|---|
| Fed | +87.7 | +68.6 | +44.8 | +27.8 | +15.9 | +7.7 | +2.3 | -1.1 | -3.2 |
| RBI, exchange 0.42 | +12.9 | +16.3 | +10.0 | +5.0 | +1.3 | -1.3 | -2.8 | -3.6 | -3.9 |
| RBI, exchange 0.17 | +4.0 | +12.7 | +6.8 | +2.4 | -0.6 | -2.4 | -3.4 | -3.8 | -3.7 |

Across the assumed persistence (output 0.80 to 0.95, inflation 0.70 to
0.90), the RBI's first move is +12.8 to +18.5bp, its first-year average
+2.9 to +13.8bp, and it turns to easing between the second and eighth
quarter.

- **The two channels play out in sequence.** The RBI tightens first,
  because the rupee weakens and imported inflation arrives within a
  quarter; it eases later, when slower demand reaches Indian output and
  then Indian inflation. The static game nets these into one number;
  the dynamic game shows them in order.
- **The RBI responds more, and more robustly, than in the static game.**
  Its first move is positive at every persistence tested and at both ends
  of the exchange range, and at the exchange floor it is three times the
  static move (+12.7 against +4.0bp), because a bank setting a rule
  responds to the inflation it expects the Fed's rule to import before
  the demand effect arrives.
- **This narrows the gap ADR 022 found.** India's repo rate rises for
  months after a Fed tightening surprise; the static game's RBI moved far
  less than the real one. The dynamic game's RBI follows the Fed first and
  by more, which is the pattern in the data, though still less than the
  data's magnitude.

## Consequences

- The model can now answer how quickly the RBI follows the Fed, and why
  it later reverses: a claim the static game could not make.
- The headline still runs on the static game. Using the dynamic game in
  the chain means feeding households the RBI's caused rate path directly,
  quarter by quarter, instead of scaling the US impulse response to one
  equilibrium move. That changes `shock_path` and the scenarios and is a
  separate decision.
- The persistence parameters are assumed, anchored to Rudebusch and
  Svensson's US estimates. Estimating them for India, or calibrating them
  so the dynamic RBI reproduces its measured response to Fed surprises
  (ADR 022), is the natural next step.
- The static game's inconsistency, a demand spillover that lowers output
  without lowering inflation, is recorded here and left in place, since
  changing it would move every historical validation.
