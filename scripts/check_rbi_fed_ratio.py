"""How far the RBI moves per point the Fed's expected path moves (step 3b, ADR 027).

ADR 022 measured the RBI's repo rate response to a Fed surprise, but per
unit of surprise, and a surprise scaled to expected rates a year ahead is
not a Fed move of the same size. To calibrate how strongly the model's RBI
follows the Fed, both sides have to be measured the same way.

The US side here is the two-year Treasury yield (FRED GS2), which moves
with expected Fed policy and, unlike the funds rate, kept moving while the
funds rate sat at zero in 2012-2015. Both responses are estimated with the
same local projection, the same surprises, the same sample (August 2012 to
December 2023) and the same controls, and the ratio of the two is the
RBI's move per point of move in expected US policy: in the units the
network's caused moves are in.

Reads data and prints. Changes nothing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from check_rbi_response_to_fed import (
    RBI_WORKBOOK,
    local_projection,
    monthly_surprises,
)

from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)

START = date(2012, 8, 1)
HORIZONS = (0, 1, 3, 6, 9, 12)


def main() -> None:
    shock = monthly_surprises()
    last = max(shock)

    with FredAdapter() as fred:
        series = fred.fetch_series("GS2", observation_start="2011-01-01")
    two_year = {o.period: o.value for o in series.observations if o.value is not None}

    repo = ManualExcelAdapter(RBI_REPO_RATE).fetch(
        RBI_REPO_RATE.series_id, path=RBI_WORKBOOK, archive=False
    )
    repo_rate = {o.period: o.value for o in repo.observations if o.value is not None}
    cpi = ManualExcelAdapter(RBI_CPI_INFLATION).fetch(
        RBI_CPI_INFLATION.series_id, path=RBI_WORKBOOK, archive=False
    )
    inflation = {o.period: o.value for o in cpi.observations if o.value is not None}

    us = local_projection(two_year, shock, start=START, end=last)
    india = local_projection(repo_rate, shock, start=START, end=last, control=inflation)

    print(f"sample {START} to {last}, responses per 1pp Fed surprise (pp)")
    print(f"  {'m':>3}  {'US 2-year':>16}  {'RBI repo':>16}  {'RBI / US':>9}")
    ratios = []
    for (h, b_us, se_us, _), (_, b_in, se_in, _) in zip(us, india, strict=True):
        if np.isnan(b_us) or np.isnan(b_in):
            print(f"  {h:>3}  too few observations")
            continue
        ratio = b_in / b_us if abs(b_us) > 1e-9 else float("nan")
        ratios.append((h, ratio, b_us / se_us))
        print(
            f"  {h:>3}  {b_us:>+7.2f} (t {b_us / se_us:>+5.2f})  "
            f"{b_in:>+7.2f} (t {b_in / se_in:>+5.2f})  {ratio:>+9.2f}"
        )

    usable = [r for h, r, t in ratios if h >= 3 and abs(t) >= 2.0]
    print()
    if usable:
        print("  RBI move per point of expected US policy, horizons 3-12 months where the")
        print(f"  US response is significant: {min(usable):.2f} to {max(usable):.2f}")
    else:
        print("  the US response is not significant at 3-12 months, so no ratio is usable")
    print("  The network's dynamic RBI (ADR 026) moves 0.12 per point of the Fed's move;")
    print("  the static one 0.15.")


if __name__ == "__main__":
    main()
