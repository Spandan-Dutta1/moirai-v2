# 20. Proxy identification of the US policy shock

Date: 2026-09-30
Status: Accepted

## Context

The household path takes its shape from the US VAR's response to a policy
shock (ADR 010). The shock has been identified by a Cholesky ordering,
industrial production, then CPI, then the funds rate: output and prices do
not respond to the funds rate within the month, and the Fed responds to
both. The ordering is untestable, and on the 1985-2007 sample it produces
the price puzzle, the price level rising after a tightening.

An external instrument removes the ordering. The Bauer and Swanson (2023)
surprise, measured from futures in a 30-minute window around each FOMC
announcement and orthogonalised to public information, moves with the
policy shock and, by construction of the window, with nothing else.

## Decision

`identification.identify_proxy` implements the proxy SVAR of Stock and
Watson (2012, 2018) and Mertens and Ravn (2013). The policy column of B is
proportional to E[u z]; it is scaled to a one standard deviation shock and
signed so the policy variable rises. The other columns complete B so that
B B' = Sigma and are named `unidentified_*`. A `ProxyFirstStage` reports
the instrument's strength with a robust F statistic and warns below 10.

Seven tests on a simulated economy with a known impact matrix check that
the true column is recovered, the covariance reproduced, the answer
unchanged when the VAR's variables are reordered, and a noise instrument
flagged weak.

`scripts/check_proxy_svar.py` estimates the pipeline's VAR and compares
the two identifications.

## Result

**The instrument is strong.** Funds rate residual on the monthly
surprise, 233 months from February 1988 to June 2007: coefficient +0.80,
correlation +0.30, robust F = 18.0.

**Responses per 1pp peak funds rate move**, cumulative, percent of level:

| Months | Output: Cholesky | Output: proxy | Prices: Cholesky | Prices: proxy |
|---|---|---|---|---|
| 0 | +0.00 | +0.24 | +0.00 | -0.04 |
| 6 | +0.15 | +0.50 | +0.21 | +0.17 |
| 12 | -0.17 | +0.30 | +0.31 | +0.25 |
| 24 | -0.65 | -0.09 | +0.33 | +0.27 |
| 36 | -0.78 | -0.21 | +0.34 | +0.28 |

- **The price puzzle survives the proxy.** Prices rise under both, by
  similar amounts. The puzzle is not an artefact of the ordering. The
  remaining explanations are the sample and the specification: a short
  Great Moderation sample, and no commodity or forward-looking price
  variable (adding a commodity index did not remove it either; see the
  pipeline's note).
- **Output responds later and less under the proxy**, and rises for the
  first year. Output rising after a tightening surprise is the pattern
  associated with the Fed information effect (Nakamura and Steinsson
  2018): a surprise tightening can signal that the Fed sees a stronger
  economy. Bauer and Swanson's orthogonalisation is designed to remove
  this, and on monthly sums over 1988-2007 it may not remove all of it.

**The headline under each identification:**

| | Shock | Aggregate | Floating borrowers | Savers | Spread |
|---|---|---|---|---|---|
| Cholesky | 0.32 SD | -0.085% | -0.650% | -0.043% | -0.61pp |
| Proxy | 0.27 SD | -0.011% | -0.547% | +0.025% | -0.57pp |

The spread moves by 0.04 points. What identification changes is the
aggregate and the savers, through job losses: with output falling later
and less, fewer households lose work.

## Decision on the default

The pipeline keeps the Cholesky identification.

- The spread, the finding, is 0.57 to 0.61 across the two identifications,
  inside the range ADR 019 already reports. Switching would change the
  aggregate, which ADR 018 already declines to call a finding.
- The proxy does not resolve the anomaly that motivated it, and adds one
  of its own: output rising for a year after a tightening. Replacing one
  questionable shape with another would not make the path more credible.
- The choice is made on those grounds, not on which headline is preferred;
  both are reported here and the method is kept for use.

## Consequences

- The headline spread is robust to the identification of the US shock as
  well as to deposit pass-through and the MPC endpoints. Its range comes
  from the exchange coefficient (ADR 019).
- The price puzzle is now known not to come from the ordering. Its likely
  sources are the sample and the missing forward-looking variables, which
  a VAR with the one-year Treasury yield as the policy indicator (Gertler
  and Karadi 2015) and a longer sample using a shadow rate (Wu and Xia
  2016) would address. The household path holds inflation at baseline for
  India regardless (ADR 010).
- Bootstrap bands are not available for proxy identification; the
  bootstrap re-identifies only Cholesky models. A moving-block bootstrap
  (Jentsch and Lunsford 2019) would be needed.
