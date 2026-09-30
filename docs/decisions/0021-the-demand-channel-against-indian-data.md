# 21. The demand channel against Indian data

Date: 2026-09-30
Status: Accepted

## Context

ADR 019 bounded the Fed-to-India exchange coefficient, the channel that
makes the RBI tighten after a Fed hike. The demand coefficient,
`demand[RBI, Fed] = 0.36`, is the channel that pulls the other way: a Fed
tightening slows Indian output, which gives the RBI a reason to ease. It
comes from the same emerging market bond yield proxy (ADR 008), and a
one-sided correction of the exchange cell alone would have biased the
RBI's response downwards. Industrial production proved too noisy to
measure it (ADR 019).

## What was measured

`scripts/check_india_demand_channel.py`: Jorda local projections on the
Bauer-Swanson surprise orthogonalised to public information, with lags of
the outcome's change and of the surprise, Newey-West errors, and the
pandemic (2020-2021) excluded. Series not seasonally adjusted are used in
year-on-year log changes.

A method check on synthetic seasonal GDP with a known effect recovered it
under low noise; under realistic noise single quarters were insignificant.

**Real GDP, quarterly (FRED NGDPRNSAXDCINQ, 2005-2023, about 60 usable
quarters)**, change in year-on-year growth after a 1pp surprise:

| Quarters | 0 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Response (pp) | -5.5 | -5.4 | -3.4 | -2.7 | +8.2 | +9.9 |
| t | -1.46 | -1.37 | -0.81 | -0.32 | +0.91 | +1.12 |

Negative for a year, the expected sign, and insignificant throughout. The
95 percent interval at impact runs from about -13 to +2, which contains
zero and contains 0.36.

**India's merchandise exports in dollars (OECD, 1995-2023):** -10 to -37
percent over twelve months per 1pp surprise, t near -2 at one and nine
months. **US imports from India (Census):** -37 to -48 points of
year-on-year growth at six to twelve months, t between -1.98 and -2.31.

## Result

The demand channel is present in Indian data with the expected sign, and
its trade component is statistically detectable. Its size is not pinned
down: the GDP estimates cannot distinguish the assumed 0.36 from zero or
from values several times larger. Taken at face value, the trade
responses (exports are about a fifth of GDP) imply an output effect
larger than 0.36, but they are per 1pp surprise, an extrapolation from
surprises whose median is 0.03pp.

## What the size of the demand channel does to the RBI's response

The RBI's caused move under the headline scenario, solved on the network:

| Demand coefficient | Exchange 0.42 (default) | Exchange 0.17 (floor, ADR 019) |
|---|---|---|
| 0.18 | +15.5bp | +7.2bp |
| 0.36 (default) | +12.9bp | +4.0bp |
| 0.72 | +7.6bp | -2.4bp |
| 1.08 | +2.3bp | -8.9bp |

The RBI's response is the balance of two channels: the exchange channel
makes it tighten, the demand channel makes it ease. With the exchange
coefficient at its evidence floor and the demand coefficient twice the
default, the RBI eases after a Fed hike, and the headline changes sign:
floating-rate borrowers would gain.

## Decision

- The demand coefficient stays at 0.36. The data confirm its direction
  and cannot bound its size, so there is no evidence-based value to put
  in its place, and no evidence-based bound to report beside it.
- The script is kept as the evidence.
- The headline range of ADR 019, 0.20 to 0.61 points, is reported as
  conditional on the demand channel being no larger than assumed. It is
  not unconditional.

## Consequences

- **What the model can claim.** When the Fed tightens, whether the RBI
  follows depends on whether the rupee's inflation effect outweighs the
  slowdown in Indian demand. Indian data confirm both channels exist and
  cannot yet say which is larger. If the exchange channel dominates, the
  RBI follows and floating-rate borrowers bear 0.2 to 0.6 points more
  than savers. If demand dominates, the RBI eases and the transfer
  reverses.
- **What the historical record adds.** In both episodes the project
  validates against, 2013 and 2022, the RBI tightened while the Fed did
  or was expected to (ADRs 012, 013), which is what an exchange channel
  at least as strong as the demand channel predicts. That is evidence on
  the balance, not on either coefficient, and two episodes are few.
- **What would settle it.** A direct estimate of the RBI's own policy
  response to Fed surprises, a local projection of the repo rate on the
  surprise series, measures the balance of the two channels without
  measuring either. It is the natural next test, and the data for it are
  already in the project.
