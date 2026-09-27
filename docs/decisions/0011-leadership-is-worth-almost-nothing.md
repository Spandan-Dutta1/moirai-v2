# 11. Leadership in the central bank network is worth almost nothing

Date: 2026-09-28
Status: Accepted

## Context

The `fed_leads` scenario asked whether commitment changes the answer: the
Fed moves first and the other four banks respond simultaneously. It used
to report a Fed move of +100 basis points against +88 in simultaneous
play, and followers that barely moved.

Both numbers came from a bug. `network_stackelberg` pinned the leader by
overwriting its current rate, and every spillover in the network is
driven by a bank's move away from its current rate, so the followers
never saw the leader move. The old 25 basis point search grid added
quantisation on top. Once the leader's problem is solved by backward
induction over the followers' exact reaction function, the difference
disappears.

## Finding

**The Fed gains nothing by leading.** Under the US inflation shock the
Fed's caused move is +87.6 basis points leading against +87.7
simultaneous, and every follower's rate agrees to within a basis point.

The reason is structural. Moving first pays only if the followers'
replies feed back into the leader's own inflation and output. The Fed is
the anchor of the spillover hierarchy, and its inward sensitivity is the
smallest in the matrix: in the literature-anchored matrix the largest
effect of any other bank's move on US output is 0.016, against 0.36 in
the other direction. Whatever the followers do barely reaches the US, so
there is nothing for commitment to exploit.

**The same null holds for every other bank.** Each bank was tested as
leader under the US inflation shock with the literature-anchored matrix.
Since the external term was made consistent (below), the difference
between a leader's Stackelberg rate and its simultaneous rate is its
commitment value and nothing else:

| leader | external weight | commitment value |
|---|---|---|
| Federal Reserve | 0.00 | -0.22bp |
| European Central Bank | 0.05 | -0.34bp |
| Bank of Japan | 0.15 | -1.02bp |
| Bank of England | 0.05 | -0.29bp |
| Reserve Bank of India | 0.40 | -0.01bp |

No leader moves any follower by more than 0.1 basis points. The reason
differs down the hierarchy. For the Fed, the followers' replies do not
feed back. For the RBI, the replies never start: its outward strength is
the smallest in the matrix, so its moves barely reach anyone, with a
largest effect of 0.015 on any other economy's output. The two major
banks and the Bank of Japan sit between, with both links weak.

The expectation that leadership would matter for a non-anchor bank is
therefore not borne out by this network. The mirror test was not added,
because asserting that it matters would be false and asserting that it
does not was not the question asked.

## The external term: gap to the mean rate

### The problem

The first version of the table above showed the RBI's Stackelberg rate
9.4 basis points from its simultaneous rate, with a commitment value of
0.06. The remainder was an inconsistency between the solver and the loss,
not leadership.

`_reaction_system` in `src/moirai/engine/financial/network.py` treated
the external term as the squared gap between a bank's rate and the mean
of the others'. `CentralBank.loss`, evaluated through `_losses_at`, was
given the mean of the absolute gaps to each other rate. The two agree
only when a bank's rate lies above or below every other rate. Under the
US inflation shock the Fed, at 5.71 percent, sits above the RBI at 5.44
and the other three sit below it. The RBI's true best reply to the
others' simultaneous rates was 9.35 basis points from the rate
`network_nash` assigned it. That is the same order as its whole imported
move. The pairwise consistency test could not see this, because with two
banks the two definitions coincide.

### Decision

The external gap is the difference between a bank's rate and the mean of
the other banks' rates. The loss was changed to match the reaction
system, not the reverse:

- **It is what the RBI's objective is about.** The external weight rests
  on the RBI's documented concern with capital flows. Capital responds
  to a differential against a global rate, not to an average distance
  from five individual rates.
- **It is the natural reference for an exchange rate channel.** A
  currency moves against a basket, and the mean is the unweighted form
  of one.
- **It keeps the loss linear-quadratic,** so the reaction system is its
  exact first order condition and the simultaneous solver stays exact
  rather than needing a numerical fix.

The definition now lives in one place, `CentralBank.external_gap`, and
the network loss, the pairwise game and the pairwise loss all use it.
For two banks it gives the same squared gap as before, so every pairwise
result is unchanged.

### What moved

Nothing in the scenarios. `network_nash` always solved the linear
system, so its rates were already the mean-gap equilibrium. What was
wrong was the loss they were judged by, and that now agrees. Every
scenario reproduces its previous output exactly, including
`imported_tightening` at +12.9 basis points for the RBI and -0.066
percent aggregate consumption. The 9.35 basis point figure was the
distance to a different equilibrium, the one the absolute-gap loss
implied. It was never an error in the published rates.

What did move is everything computed from the true loss. The Stackelberg
leader optimises it, so commitment values shifted by at most 0.07 basis
points, which the table above reflects. The reported losses changed for
every bank with an external weight.

`test_the_simultaneous_solution_is_a_best_reply_under_the_stated_loss`
checks, in a network where the RBI sits between the others, that every
solved rate is a best reply under `_losses_at`. It fails on the old
definition.

### One nonlinearity remains

The RBI's tolerance band adds a penalty once inflation leaves the two to
six percent band, and the reaction system does not model it. The loss is
linear-quadratic only while every bank ends inside its band. In every
declared scenario every bank does, and the best-reply test above
confirms the solved rates are exact. A scenario that leaves the RBI
outside its band at equilibrium would make the simultaneous solver
approximate again, by an amount that should be measured, not assumed.

## Decision

Rename `fed_leads` to `commitment_value` and keep it as a declared
scenario whose point is the null result. A null that follows from the
structure of the network is a finding about that structure, and worth
keeping.

`test_leadership_is_worth_nothing_to_the_anchor` asserts that leading and
simultaneous play agree on every rate to within a basis point when the
Fed leads.

Measure the external objective as the gap to the mean of the other rates,
for the reasons above.

## Consequences

**Solution mode does not matter in the current network.** Scenario
results are insensitive to whether the Fed leads. That is a property of
the tiered spillover matrix, whose allocation below the anchor-to-recipient
cell is not sourced (ADR 008). A matrix with stronger feedback into the
anchor could make leadership matter; this one does not.

**The simultaneous solver is exact while no bank breaches its band.** A
scenario that pushes a bank outside its band at equilibrium should check
its solved rates against the true loss, as the best-reply test does.
