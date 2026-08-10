# 3. Build a vertical slice to the Belief Society before Layers 2 and 3

Date: 2026-08-11
Status: Accepted

## Context

The roadmap runs sequentially: Financial System (Phase 4), Economy
Simulation (Phase 5), then the MoriFish Belief Society (Phase 6). At
roughly two to four steps per working session, MoriFish is thirty or more
steps away on that path.

The Belief Society is the architecturally distinctive component. A VAR
pipeline is competent engineering that many people have built. An LLM
persona population whose formed expectations are validated against real
survey expectations data is a research contribution.

Its pipeline needs data (Layer 0, complete) and somewhere for expectations
to go. That destination could be Phase 5's household model, or it could
initially be a comparison against published expectation surveys.

## Decision

Build a minimal Belief Society directly on Layer 0 and Layer 1, validated
against survey expectations data, before completing Layers 2 and 3.

## Consequences

**Gained:** the riskiest idea in the project gets tested early. If
persona-formed expectations correlate poorly with survey data, that is
worth knowing now rather than after two more layers are built on the
assumption that they will.

**Lost:** architectural cleanliness. The Belief Society will initially lack
the numerical economy it is designed to feed, so its output is validated
rather than consumed.

**Consistent with stated philosophy:** the architecture document specifies
thin vertical slices. This applies that principle to the component where
it matters most.

**Falsifiable:** the validation target is explicit. Comparing formed
inflation expectations against a published survey series gives a real
number, and a negative result is still a result.