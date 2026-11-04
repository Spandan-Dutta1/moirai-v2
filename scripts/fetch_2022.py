"""Fetch 2022 policy rates and inflation for the five modelled banks.

Verification data for the network's spillover magnitudes. The Fed raised
425 basis points on its target range during calendar 2022 while the RBI
raised 225. If the model's spillover is right, a Fed move of that size
should imply roughly the RBI move the data shows. If it does not, the
gap is a measured failure rather than an assumption.

FRED renames and discontinues OECD-sourced series regularly, so each
quantity lists several candidate identifiers and the first that returns
data is used. Anything that fails is reported rather than silently
skipped.
"""

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import to_array
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("ERROR")

START, END = "2021-06-01", "2023-06-01"

SERIES = {
    "US policy rate (target upper)": ["DFEDTARU", "FEDFUNDS"],
    "US policy rate (effective)": ["FEDFUNDS"],
    "US inflation (PCE)": ["PCEPI", "CPIAUCSL"],
    "Euro area policy rate": ["ECBDFR", "ECBMRRFR", "IRSTCB01EZM156N"],
    "Euro area inflation (HICP)": ["CP0000EZ19M086NEST", "CPHPTT01EZM659N"],
    "UK policy rate": ["IUDSOIA", "IRSTCB01GBM156N", "BOERUKM"],
    "UK inflation (CPI)": ["GBRCPIALLMINMEI", "CPALTT01GBM659N"],
    "Japan policy rate": ["IRSTCI01JPM156N", "INTDSRJPM193N"],
    "Japan inflation (CPI)": ["JPNCPIALLMINMEI", "CPALTT01JPM659N"],
}


def summarise(label: str, candidates: list[str]) -> None:
    for code in candidates:
        try:
            with FredAdapter() as fred:
                series = fred.fetch_series(
                    code, observation_start=START, observation_end=END
                )
            periods, values = to_array(series)
            present = [
                (p, v) for p, v in zip(periods, values, strict=True) if v == v
            ]
            if not present:
                continue

            print(f"\n{label}")
            print(f"  series  : {code}  ({series.metadata.unit.value})")
            print(f"  coverage: {present[0][0]} to {present[-1][0]}")
            print(f"  {'period':<12} {'value':>10}")
            for period, value in present:
                if period.year in (2021, 2022, 2023) and period.month in (
                    1, 4, 6, 7, 10, 12
                ):
                    print(f"  {str(period):<12} {value:>10.3f}")
            return
        except Exception as error:  # noqa: BLE001
            print(f"  [{code} failed: {str(error)[:60]}]")

    print(f"\n{label}\n  NO SERIES WORKED: tried {candidates}")


for label, candidates in SERIES.items():
    summarise(label, candidates)

print("\n" + "=" * 70)
print("What to check in the output above")
print("=" * 70)
print("  US target upper bound: 0.25% in Jan 2022 to 4.50% in Dec 2022")
print("    would confirm the 425bp figure")
print("  Each bank's January 2022 rate and inflation are the starting")
print("    conditions; the 2022 average inflation is the condition applied")