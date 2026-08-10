# 4. The engine has no external API dependencies

Date: 2026-08-11
Status: Accepted
Supersedes: parts of ADR 3

## Context

The architecture describes two distinct AI components, which had been
conflated:

1. **The MoriFish Belief Society**: LLM personas that read news, debate in
   natural language, and form opinions. Language-mediated belief formation
   genuinely requires a language model.

2. **The three-layer economic simulation**: central banks reacting
   strategically to each other, financial institutions trading, and
   heterogeneous households responding to policy. This is game theory and
   agent-based simulation. It requires no language at all.

ADR 3 proposed building the Belief Society as the next vertical slice.
Reconsidering, the second component is both closer to the project's
purpose and free to run.

## Decision

The Moirai engine computes without calling any external API. Layers 0
through 5 depend on no model provider, no API key, and no network access
beyond data ingestion.

Belief dynamics in Layer 3 are implemented as adaptive learning rules:
agents update expectations from observed forecast errors, in the tradition
of Evans and Honkapohja. Not persona debate.

## Consequences

**Gained: Class A reproducibility becomes achievable for the whole engine.**
An LLM in the loop makes bit-reproducibility impossible, which would have
forced the simulation into the "artifact reproducible, replayed rather than
regenerated" category. Removing it means a simulation run can be verified
by re-running it.

**Gained: the local-first claim becomes literal.** The engine runs offline.

**Gained: results are testable.** A Nash equilibrium can be verified. A
persona's stated opinion cannot.

**Gained: zero marginal cost.** Sweeping thousands of parameter
combinations is a compute question, not a budget question.

**Lost: language-mediated belief formation.** Adaptive learning is a
weaker model of how people form expectations than reading news and
arguing about it. This is a real loss and is not disguised.

**Not foreclosed:** the Belief Society remains in the architecture as an
optional Layer 3 component. If it is ever built, it produces expectations
in the same form the learning rules do, so it substitutes cleanly. The
reproducibility class of any run using it would be Class B.