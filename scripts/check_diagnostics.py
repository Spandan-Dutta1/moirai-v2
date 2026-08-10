"""Diagnose the canonical monetary VAR."""

from moirai.core.logging import configure_logging
from moirai.engine.causal.diagnostics import diagnose, granger_causality
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

configure_logging("WARNING")

CODES = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]

prepared = []
with FredAdapter() as fred:
    for code in CODES:
        prepared.append(
            prepare(
                fred.fetch_series(
                    code, observation_start="1960-01-01", observation_end="2019-12-01"
                )
            )
        )

periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
var = estimate_var(data, names, n_lags=6, periods=periods)

report = diagnose(var)

print(f"VAR({var.n_lags}) on {list(names)}, {var.n_observations} observations")
print()
print(f"{'test':28s} {'stat':>10}  {'p':>8}  {'verdict':>8}  severity")
print("-" * 72)
for outcome in report.outcomes:
    verdict = "pass" if outcome.passed else "FAIL"
    print(
        f"{outcome.name:28s} {outcome.statistic:>10.3f}  {outcome.pvalue:>8.4f}  "
        f"{verdict:>8s}  {outcome.severity.value}"
    )
print()
print("summary:", report.summary())
print("usable :", report.is_usable)
print()

print("GRANGER CAUSALITY (predictive precedence, not causation)")
for cause in names:
    for effect in names:
        if cause == effect:
            continue
        outcome = granger_causality(var, cause, effect, data)
        arrow = "->" if outcome.rejects_null else "-/->"
        print(f"  {cause[:9]:10s} {arrow:4s} {effect[:9]:10s}  p={outcome.pvalue:.4f}")