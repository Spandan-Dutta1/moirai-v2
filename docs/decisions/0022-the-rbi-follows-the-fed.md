# 22. The RBI follows the Fed, by more than the model says

Date: 2026-09-30
Status: Accepted

## Context

ADR 021 left the headline conditional. A Fed tightening pushes the RBI to
tighten through the rupee and to ease through slower Indian demand; Indian
data confirm both channels without saying which is larger, and at the
exchange floor with a doubled demand coefficient the RBI eases and the
headline changes sign. The network's answer, the RBI's caused move, is
the balance of the two channels, so the balance can be tested directly
without measuring either: estimate how India's policy rate responds to a
Fed surprise.

## What was measured

`scripts/check_rbi_response_to_fed.py`: Jorda local projections of the
policy rate's level on the Bauer-Swanson monthly surprise orthogonalised
to public information, six lags of the rate's change and of the surprise,
Newey-West errors, 2020-2021 excluded. A method check on synthetic data
with a planted response of +0.5pp recovered +0.47pp at three months,
t = 3.0.

Response of India's policy rate per 1pp Fed surprise:

| Months | 0 | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|
| IMF discount rate, 2000-2022 | +0.29 (1.7) | +0.39 (1.2) | +0.97 (1.7) | +0.75 (0.8) | +0.59 (0.6) | +0.09 (0.1) |
| RBI repo, 2011-2023 | +0.82 (1.8) | +1.18 (2.0) | +2.86 (3.0) | +4.12 (2.0) | +3.35 (1.4) | +2.91 (1.1) |
| RBI repo, 2012-2023, Indian CPI controlled | +1.00 (2.1) | +1.33 (2.3) | +3.06 (3.9) | +4.99 (3.3) | +4.97 (3.3) | +5.06 (2.0) |
| RBI repo, 2012-2023, uncontrolled | +0.85 (1.8) | +0.96 (1.5) | +2.33 (2.5) | +3.45 (1.7) | +3.33 (1.4) | +2.94 (1.0) |

t statistics in brackets.

## Result

**The sign is settled.** India's policy rate rises after a Fed tightening
surprise, significantly in the current regime and at the margin of
significance over 2000-2022. Controlling for Indian CPI inflation makes
the response larger and more precise, not smaller: the RBI's following of
the Fed is not its reaction to its own inflation in the same months. The
exchange channel, together with whatever else leads the RBI to defend the
rupee, outweighs the demand channel in the RBI's actual behaviour.

**The size does not match the model.** The network's RBI moves 0.15
points per point of Fed tightening at the default exchange coefficient and
0.05 at its floor. The data give 1 to 5 points per 1pp surprise. The two
are not the same quantity: the surprise moves expected US rates a year
ahead by 1pp, and the Fed's realised move after such a surprise can be
larger and more persistent than 1pp, so the data's ratio of RBI move to
Fed move is below its response per surprise. The estimate also rests on a
handful of large episodes, 2013 and 2022 above all, in which global
shocks other than the Fed may have moved the RBI too, and it extrapolates
from surprises whose median is 0.03pp. The magnitude is therefore not
taken as a measurement. But no plausible discount closes a gap of an
order of magnitude.

## Decision

- ADR 021's condition is resolved in favour of the headline's sign. The
  headline is no longer reported as conditional on the demand channel.
- The model is not recalibrated. The data establish that the RBI responds
  to the Fed far more than the network's RBI does; they do not give a
  value to calibrate to, for the reasons above.
- The headline range of ADR 019, a spread of 0.20 to 0.61 points, is
  reported as a lower bound on the burden a US tightening places on
  India's floating-rate borrowers relative to savers.
- The script is kept as the evidence.

## Consequences

- **What the model can claim now.** When US inflation makes the Fed
  tighten, the RBI follows, as its own record shows; floating-rate
  borrowers bear at least 0.2 to 0.6 points more consumption loss than
  savers, and the true figure is likely larger because the model's RBI
  follows the Fed less than the real one does.
- **Where the gap probably lives.** The network's RBI responds to the Fed
  only through inflation and output. The real RBI also defends the rupee
  and capital flows directly, which ADR 008 represents with an external
  weight whose magnitude is assumed. Calibrating that weight, or the
  exchange coefficient, so that the network reproduces the RBI's observed
  response to Fed surprises is the natural next step, by indirect
  inference against this projection.
- **This reverses the worry of ADR 019.** Testing the exchange coefficient
  suggested the headline might be overstated; testing the RBI's actual
  behaviour shows it is understated.
