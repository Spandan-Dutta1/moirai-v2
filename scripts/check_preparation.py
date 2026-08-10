"""Fetch real series and prepare them for estimation."""

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import align, check_stationarity, prepare, to_array
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("INFO")

SERIES = {
    "INDCPIALLMINMEI": "India CPI (index)",
    "INTDSRINM193N": "India discount rate (percent)",
}

prepared = []
with FredAdapter() as fred:
    for code, label in SERIES.items():
        series = fred.fetch_series(code, observation_start="2000-01-01")

        periods, values = to_array(series)
        raw = check_stationarity(code, values)

        print()
        print(f"{label}  [{series.metadata.unit.value}]")
        print(f"  observations : {len(values)}")
        print(f"  raw verdict  : {raw.verdict.value}")
        print(f"  ADF p        : {raw.adf_pvalue:.4f}   KPSS p: {raw.kpss_pvalue:.4f}")

        result = prepare(series)
        print(f"  transformation: {result.record.steps[0].value}")
        print(f"  reason        : {result.record.reason}")
        if result.stationarity:
            print(f"  after verdict : {result.stationarity.verdict.value}")
        prepared.append(result)

periods, matrix = align(prepared)
print()
print("aligned matrix:", matrix.shape)
print("sample        :", periods[0], "to", periods[-1])