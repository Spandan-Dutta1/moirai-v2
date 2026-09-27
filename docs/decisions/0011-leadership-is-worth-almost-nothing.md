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
Commitment value is measured as the leader's Stackelberg rate minus its
best reply, under its true loss, to the other banks' simultaneous rates:

| leader | external weight | commitment value |
|---|---|---|
| Federal Reserve | 0.00 | -0.22bp |
| European Central Bank | 0.05 | -0.27bp |
| Bank of Japan | 0.15 | -1.02bp |
| Bank of England | 0.05 | -0.33bp |
| Reserve Bank of India | 0.40 | +0.06bp |

No leader moves any follower by more than 0.2 basis points. The reason
differs down the hierarchy. For the Fed, the followers' replies do not
feed back. For the RBI, the replies never start: its outward strength is
the smallest in the matrix, so its moves barely reach anyone, with a
largest effect of 0.015 on any other economy's output. The two major
banks and the Bank of Japan sit between, with both links weak.

The expectation that leadership would matter for a non-anchor bank is
therefore not borne out by this network. The mirror test was not added,
because asserting that it matters would be false and asserting that it
does not was not the question asked.

## A second problem found on the way

A leader's Stackelberg rate should differ from its simultaneous rate
only by its commitment value. For the RBI under the US inflation shock
it differs by 9.4 basis points while the commitment value is 0.06. The
remainder is a pre-existing inconsistency in the simultaneous solver,
not leadership.

`_reaction_system` in `src/moirai/engine/financial/network.py` treats the
external term as the squared gap between a bank's rate and the mean of
the others'. `CentralBank.loss`, evaluated through `_losses_at`, uses the
squared mean of the absolute gaps. The two agree only when a bank's rate
lies above or below every other rate. With the Fed at 5.71 percent,
above the RBI at 5.44 and the others below it, they disagree. The RBI's
true best reply to the others' simultaneous rates is then 9.35 basis
points from the rate `network_nash` assigns it. The pairwise consistency
test cannot catch this, because with two banks the two forms coincide.

This affects every bank with an external weight whose rate sits between
others', and so it affects the RBI's move in `imported_tightening`
(+12.9 basis points) by an amount of the same order. It is recorded here
and not fixed, because resolving it means deciding which of the two
external terms is the intended one, and that is a mechanism decision.

## Decision

Rename `fed_leads` to `commitment_value` and keep it as a declared
scenario whose point is the null result. A null that follows from the
structure of the network is a finding about that structure, and worth
keeping.

`test_leadership_is_worth_nothing_to_the_anchor` asserts that leading and
simultaneous play agree on every rate to within a basis point when the
Fed leads. The Fed carries no external weight, so that comparison is
free of the solver inconsistency above. The same comparison for any
other bank would measure the inconsistency rather than commitment.

## Consequences

**Solution mode does not matter in the current network.** Scenario
results are insensitive to whether the Fed leads. That is a property of
the tiered spillover matrix, whose allocation below the anchor-to-recipient
cell is not sourced (ADR 008). A matrix with stronger feedback into the
anchor could make leadership matter; this one does not.

**The simultaneous solver is not the equilibrium of the stated loss** for
banks with an external weight. Until that is resolved, rates for the ECB,
Bank of England, Bank of Japan and RBI carry an error of up to about ten
basis points whenever their rate lies between others'.
