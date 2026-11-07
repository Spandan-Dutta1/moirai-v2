"""The Gertler-Karadi specification against ADR 020's, under the same instrument (F1).

ADR 020 identified the US policy shock with the Bauer-Swanson surprise and kept
the Cholesky headline, because the proxy left the price puzzle in place and
made output rise for a year after a tightening. Its consequences named the
next step: the specification of Gertler and Karadi (2015).

    A  ADR 020   output growth, inflation, change in the funds rate; differenced
    B  GK        log output, log prices, the 1-year Treasury yield, the excess
                 bond premium; in levels; the 1-year yield is the policy indicator

Both are identified by MPS_ORTH (Bauer and Swanson 2023), 12 lags, on two
samples: 1985-2007 (ADR 020's) and 1985-2019.

Pre-registered rule (written before this script was run): specification B
becomes the headline identification only if, on 1985-2007,
    1. the first-stage robust F is above 10,
    2. prices at 12 months are not above zero (no price puzzle),
    3. output at 6 months is not above zero (no early output rise),
and 1 to 3 also hold on 1985-2019. Point estimates: the bootstrap does not
re-identify proxy models (ADR 020), so there are no bands here.

Reads data and prints. Writes nothing to the warehouse; the EBP file is read
without archiving.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import openpyxl

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.identification import identify_proxy
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import PreparedSeries, Transformation, align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter
from moirai.engine.data_fabric.ingestion.manual_csv import CsvSpec, ManualCsvAdapter, PeriodFormat
from moirai.engine.data_fabric.series.models import Frequency, Unit
from moirai.engine.data_fabric.series.temporal import restrict

warnings.filterwarnings("ignore")
configure_logging("ERROR")

MANUAL = get_paths().raw / "manual"
SURPRISES = MANUAL / "frbsf_monetary_policy_surprises.xlsx"
EBP_FILE = MANUAL / "ebp_csv.csv"
N_LAGS = 12
HORIZON = 48
SAMPLES = {"1985-2007": ("1985-01-01", "2007-06-01"), "1985-2019": ("1985-01-01", "2019-12-01")}

EBP_SPEC = CsvSpec(
    series_id="us_ebp",
    title="Excess bond premium (Gilchrist and Zakrajsek; Favara et al. update)",
    period_column="date",
    value_column="ebp",
    period_format=PeriodFormat.ISO_DATE,
    frequency=Frequency.MONTHLY,
    unit=Unit.PERCENT,
    unit_label="Percentage points",
    geography="US",
    publisher="Federal Reserve Board",
    source_url="https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv",
)


def monthly_surprises() -> dict[date, float]:
    workbook = openpyxl.load_workbook(SURPRISES, read_only=True, data_only=True)
    rows = [r for r in workbook["Monthly (update 2023)"].iter_rows(values_only=True) if any(r)]
    header = list(rows[0])
    y, m, s = header.index("Year"), header.index("Month"), header.index("MPS_ORTH")
    surprises = {
        date(int(r[y]), int(r[m]), 1): float(r[s])
        for r in rows[1:]
        if isinstance(r[y], (int, float)) and isinstance(r[s], (int, float))
    }
    workbook.close()
    return surprises


@dataclass(frozen=True)
class Outcome:
    label: str
    sample: str
    policy: str
    f_statistic: float
    n_overlap: int
    max_modulus: float
    output: dict[int, float]
    prices: dict[int, float]
    policy_path: dict[int, float]


def responses(irf, variable: str, shock: str, *, cumulate: bool) -> np.ndarray:
    path = np.asarray(irf.path(variable, shock), dtype=float)
    return np.cumsum(path) if cumulate else path


def run(
    label: str, sample: str, prepared: list[PreparedSeries], policy: str, *, levels: bool
) -> Outcome:
    periods, data = align(prepared)
    names = tuple(p.series_id for p in prepared)
    var = estimate_var(data, names, n_lags=N_LAGS, periods=periods, require_stable=False)
    proxy, first = identify_proxy(
        var, monthly_surprises(), policy, instrument_name="Bauer-Swanson MPS_ORTH"
    )
    irf = impulse_responses(proxy, horizon=HORIZON)
    shock = f"{policy}_shock"
    # Differenced variables are cumulated back to levels; level variables are not.
    rate = responses(irf, policy, shock, cumulate=not levels)
    peak = abs(rate[int(np.argmax(np.abs(rate)))])
    output = responses(irf, "indpro", shock, cumulate=not levels) / peak * 100.0
    prices = responses(irf, "cpiaucsl", shock, cumulate=not levels) / peak * 100.0
    horizons = (0, 3, 6, 12, 24, 36, 48)
    return Outcome(
        label=label,
        sample=sample,
        policy=policy,
        f_statistic=first.f_statistic,
        n_overlap=first.n_overlap,
        max_modulus=float(var.stability.max_modulus),
        output={h: float(output[h]) for h in horizons},
        prices={h: float(prices[h]) for h in horizons},
        policy_path={h: float(rate[h] / peak) for h in horizons},
    )


def main() -> None:
    ebp_all = ManualCsvAdapter(EBP_SPEC).ingest(EBP_FILE, archive=False)
    outcomes: list[Outcome] = []

    for sample, (start, end) in SAMPLES.items():
        with FredAdapter() as fred:
            raw = {
                code: fred.fetch_series(code, observation_start=start, observation_end=end)
                for code in ("INDPRO", "CPIAUCSL", "FEDFUNDS", "GS1")
            }
        ebp = restrict(ebp_all, start=date.fromisoformat(start), end=date.fromisoformat(end))

        # A: ADR 020's specification, transformations chosen by the unit rule.
        adr020 = [prepare(raw[c]) for c in ("INDPRO", "CPIAUCSL", "FEDFUNDS")]
        outcomes.append(run("A  ADR 020", sample, adr020, "fedfunds", levels=False))

        # B: Gertler-Karadi, in levels.
        gk = [
            prepare(raw["INDPRO"], transformation=Transformation.LOG, verify=False),
            prepare(raw["CPIAUCSL"], transformation=Transformation.LOG, verify=False),
            prepare(raw["GS1"], transformation=Transformation.NONE, verify=False),
            prepare(ebp, transformation=Transformation.NONE, verify=False),
        ]
        outcomes.append(run("B  Gertler-Karadi", sample, gk, "gs1", levels=True))

    print("F1  THE GERTLER-KARADI SPECIFICATION UNDER THE BAUER-SWANSON INSTRUMENT")
    print("    responses per 1pp peak move of the policy indicator, percent of level\n")
    for o in outcomes:
        print(f"{o.label}   sample {o.sample}   policy indicator {o.policy}")
        print(
            f"  first stage: robust F {o.f_statistic:.1f} on {o.n_overlap} months"
            f"{'  WEAK' if o.f_statistic < 10 else ''};   largest root {o.max_modulus:.3f}"
        )
        print(f"  {'h':>4} {'output':>9} {'prices':>9} {'policy':>9}")
        for h in o.output:
            print(f"  {h:>4} {o.output[h]:>+9.2f} {o.prices[h]:>+9.2f} {o.policy_path[h]:>+9.2f}")
        print()

    def verdict(o: Outcome) -> dict[str, bool]:
        return {
            "1. first stage F > 10": o.f_statistic > 10,
            "2. no price puzzle at 12 months": o.prices[12] <= 0,
            "3. no output rise at 6 months": o.output[6] <= 0,
        }

    print("THE PRE-REGISTERED RULE, specification B")
    passes = {}
    for o in outcomes:
        if not o.label.startswith("B"):
            continue
        checks = verdict(o)
        passes[o.sample] = all(checks.values())
        print(f"  {o.sample}")
        for name, ok in checks.items():
            print(f"    {name:<34} {'pass' if ok else 'FAIL'}")
    adopt = passes.get("1985-2007", False) and passes.get("1985-2019", False)
    print()
    print(
        "  DECISION: "
        + (
            "B clears every criterion on both samples; it becomes the headline (ADR 030)."
            if adopt
            else "B does not clear every criterion; Cholesky stays the headline, and "
            "ADR 030 records what was tried."
        )
    )
    print("  Point estimates only: proxy models have no bootstrap bands yet (ADR 020).")


if __name__ == "__main__":
    main()
