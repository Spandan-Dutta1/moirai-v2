# 30. The Gertler-Karadi specification does not clear the bar; Cholesky stays

Status: accepted, 7 November 2026

## Context

ADR 020 identified the US policy shock with the Bauer and Swanson (2023)
surprise (MPS_ORTH) and kept the Cholesky headline: the proxy left the price
puzzle in place and made output rise for a year after a tightening. Its
consequences named the next step, the specification of Gertler and Karadi
(2015): the 1-year Treasury yield as the policy indicator, the excess bond
premium in the VAR, and estimation in levels.

`scripts/check_gk_proxy.py` runs ADR 020's specification (A) and the
Gertler-Karadi specification (B) side by side under the same instrument,
12 lags, on 1985-2007 and 1985-2019. The excess bond premium is the Federal
Reserve's series (Gilchrist and Zakrajsek 2012; Favara et al. 2016 update),
read from `data/raw/manual/ebp_csv.csv`.

The rule was written before the script was run: B becomes the headline
identification only if, on 1985-2007, (1) the first-stage robust F exceeds 10,
(2) prices at 12 months are not above zero, (3) output at 6 months is not
above zero, and 1 to 3 also hold on 1985-2019.

## Result

A on 1985-2007 reproduces ADR 020 exactly (F 18.0 on 233 months; output
+0.50 at 6 months; prices +0.25 at 12), so the script computes what ADR 020
computed.

Responses per 1pp peak move of the policy indicator, percent of level:

| Specification | Sample | First-stage F | Months | Largest root | Output, 6m | Prices, 12m |
|---|---|---|---|---|---|---|
| A, ADR 020 | 1985-2007 | 18.0 | 233 | 0.947 | +0.50 | +0.25 |
| B, Gertler-Karadi | 1985-2007 | **4.8 (weak)** | 233 | 0.998 | +0.71 | -0.18 |
| A, ADR 020 | 1985-2019 | 12.8 | 383 | 0.900 | -0.71 | -0.24 |
| B, Gertler-Karadi | 1985-2019 | **5.8 (weak)** | 383 | 0.998 | -3.88 | -1.21 |

The rule applied to B: on 1985-2007, criterion 1 fails and criterion 3 fails;
on 1985-2019, criterion 1 fails. **B is not adopted.**

## Decision

The pipeline keeps the Cholesky identification, as ADR 020 decided.

- The instrument is weak for the 1-year yield in a levels VAR with the bond
  premium (F 4.8 and 5.8). With a weak instrument the proxy's responses are
  not evidence, however they look (Stock and Watson 2018). On 1985-2019 B looks
  like a textbook monetary shock (prices and output fall, no puzzle); that is
  exactly the result criterion 1 exists to refuse.
- The prediction written before the run expected a strong first stage for B.
  It was wrong, and is recorded as wrong.

## A finding the rule did not test

ADR 020's own specification on the longer sample has a strong instrument
(F 12.8, 383 months) and neither anomaly: prices fall from the start (-0.24 at
12 months) and so does output (-0.71 at 6). That is consistent with ADR 020's
suspicion that the price puzzle and the early output rise belong to the
1985-2007 sample rather than to the identification.

It is not adopted, for two reasons. It was not the pre-registered test, and
adopting it after seeing it would be the specification search the rule exists
to prevent. And 1985-2019 includes 2009-2015, when the funds rate was at its
lower bound, where a linear VAR in the funds rate is misspecified.

It is queued as its own pre-registered test: ADR 020's specification on
1985-2019 with the Wu and Xia (2016) shadow rate as the policy indicator, under
the same three criteria.

## Consequences

- The headline is unchanged; nothing in the model changes. The comparison run
  shows only the new script.
- The frontier route for the US shock is now narrowed to: (a) the long-sample
  test above, with a shadow rate; (b) a factor-augmented VAR (Bernanke, Boivin
  and Eliasz 2005) on FRED-MD, judged by the same three criteria; (c) weak-
  instrument-robust inference for SVAR-IV (Montiel Olea, Stock and Watson 2021)
  and a moving-block bootstrap (Jentsch and Lunsford 2019), so proxy results
  carry bands.
- Not tried, and not to be tried as a rescue of B: other instruments from the
  same workbook (MPS, unorthogonalised). Any such test would be pre-registered
  first.
- Numbering: the study notes reserved 030 for pinning the ALFRED vintage in
  `validate_2022`; that decision takes the next free number when it is made.
