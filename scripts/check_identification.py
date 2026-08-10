"""Identify the canonical monetary VAR, and measure how much the ordering matters."""


from moirai.core.logging import configure_logging
from moirai.engine.causal.identification import (
    Sign,
    SignRestriction,
    identify_cholesky,
    identify_sign_restrictions,
    ordering_sensitivity,
)
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

print(f"VAR({var.n_lags}) on {list(names)}, {var.n_observations} observations")
print()

# --- Cholesky, the standard ordering -------------------------------------
model = identify_cholesky(var, ordering=names)
print("CHOLESKY  ordering:", " -> ".join(names))
print(f"  reconstruction error : {model.reconstruction_error():.2e}")
print(f"  max shock correlation: {model.max_off_diagonal_correlation():.2e}")
print()
print("  impact matrix B (rows = variables, columns = shocks)")
for i, name in enumerate(names):
    row = "  ".join(f"{model.impact[i, j]:+.5f}" for j in range(len(names)))
    print(f"    {name[:9]:10s} {row}")
print()

# --- how much does the ordering matter? ----------------------------------
print("ORDERING SENSITIVITY  (impact response, all 6 orderings)")
for response_of in names:
    sensitivity = ordering_sensitivity(var, response_of, "fedfunds")
    print(
        f"  {response_of[:9]:10s} to fedfunds: "
        f"[{sensitivity.minimum:+.5f}, {sensitivity.maximum:+.5f}]  "
        f"median {sensitivity.median:+.5f}  "
        f"sign flips: {sensitivity.sign_flips}"
    )
print()

# --- sign restrictions ----------------------------------------------------
shock_names = ("supply", "demand", "monetary")
restrictions = (
    SignRestriction(variable="fedfunds", shock="monetary", sign=Sign.POSITIVE),
    SignRestriction(variable="indpro", shock="monetary", sign=Sign.NEGATIVE),
    SignRestriction(variable="cpiaucsl", shock="monetary", sign=Sign.NEGATIVE),
)

identified = identify_sign_restrictions(var, restrictions, shock_names, n_draws=5_000)
print("SIGN RESTRICTIONS  (contractionary monetary shock)")
print(f"  draws accepted : {identified.n_accepted} / {identified.n_draws} "
      f"({identified.acceptance_rate:.1%})")
print()
for variable in names:
    low, mid, high = identified.impact_quantiles(variable, "monetary")
    print(f"  {variable[:9]:10s} impact: {mid:+.5f}  [{low:+.5f}, {high:+.5f}]  (16-84%)")
print()
print("The interval is the identified set, not sampling uncertainty.")
print("Sign restrictions answer with a range because the assumption is weaker.")