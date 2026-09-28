"""Sign-restriction identification of the Indian monetary VAR.

ADR 007 found that the recursive Indian VAR has output rising after a
rate rise in all four specifications tried, and listed sign restrictions
as an untried alternative. This applies them to the pipeline's VAR as it
stands. Everything below was fixed before the first run.

The VAR. Exactly as in run_pipeline_india.py: IIP growth, CPI inflation
and the repo rate (differenced by the preparation layer), from January
2012, six lags. Not respecified (ADR 007).

The restrictions, on one shock named "policy":
    repo rate    rises on impact                     horizon 0
    IIP growth   does not rise for three months      horizons 0, 1, 2
    inflation    unrestricted, because the price puzzle is in question
The other two shocks are unrestricted.

The draws. 20,000 Haar-uniform rotations, seed 42, at the VAR's point
estimates. The set therefore reflects identification uncertainty only,
not estimation uncertainty, and is narrower than a Bayesian set would be.

The normalisation. Each draw's responses are scaled to a 100bp impact on
the repo rate, so draws with different shock sizes are comparable.

What is informative. IIP growth cannot rise at horizons 0 to 2 in any
accepted draw, by construction, so a negative response there says nothing.
The finding is in the unrestricted horizons. The classification uses the
cumulative IIP growth response over months 3 to 24 in each accepted draw:
    mostly negative   at least 84% of draws negative
    positive          at most 16% of draws negative
    straddles zero    anything between
An acceptance rate near zero is reported as its own finding: the
restrictions would then be inconsistent with the covariance and dynamics.

No household numbers are produced (ADRs 006, 007).
"""

import warnings
from datetime import date

import numpy as np

from moirai.core.logging import configure_logging
from moirai.core.paths import get_paths
from moirai.engine.causal.identification import (
    Sign,
    SignRestriction,
    identify_cholesky,
    identify_sign_restrictions,
)
from moirai.engine.causal.irf import impulse_responses
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import estimate_var
from moirai.engine.data_fabric.ingestion.manual_excel import (
    RBI_CPI_INFLATION,
    RBI_IIP,
    RBI_REPO_RATE,
    ManualExcelAdapter,
)
from moirai.engine.data_fabric.series.temporal import drop_missing, restrict

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
configure_logging("ERROR")

# ---- fixed before the first run --------------------------------------------
START = date(2012, 1, 1)
N_LAGS = 6
N_DRAWS = 20_000
SEED = 42
HORIZON = 36
RESTRICTED = (0, 1, 2)
WINDOW = (3, 24)
MOSTLY, HARDLY = 0.84, 0.16
LOW_ACCEPTANCE = 0.01

IIP, CPI, REPO = "in_iip_yoy", "in_cpi_inflation", "in_repo_rate"
SHOCKS = ("other_1", "other_2", "policy")

PATH = get_paths().raw / "manual" / "rbi_select_economic_indicators.xlsx"
if not PATH.exists():
    raise SystemExit(f"missing {PATH}")

prepared = [
    prepare(drop_missing(restrict(ManualExcelAdapter(spec).ingest(PATH), start=START)))
    for spec in (RBI_IIP, RBI_CPI_INFLATION, RBI_REPO_RATE)
]
periods, data = align(prepared)
names = tuple(item.series_id for item in prepared)
assert names == (IIP, CPI, REPO), names
var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)

print("=" * 76)
print("THE INDIAN VAR, AS IN run_pipeline_india.py")
print("=" * 76)
for item in prepared:
    print(f"  {item.series_id:<18} {item.record.steps[0].value:<12} n={len(item)}")
print(f"  sample {periods[0]} to {periods[-1]}, {data.shape[0]} obs, VAR({N_LAGS})")
print(f"  max eigenvalue modulus {var.stability.max_modulus:.3f}")

restrictions = (
    SignRestriction(variable=REPO, shock="policy", sign=Sign.POSITIVE),
    SignRestriction(variable=IIP, shock="policy", sign=Sign.NEGATIVE, horizons=RESTRICTED),
    SignRestriction(variable=CPI, shock="policy", sign=Sign.UNRESTRICTED),
)
identified = identify_sign_restrictions(var, restrictions, SHOCKS, n_draws=N_DRAWS, seed=SEED)

print()
print("=" * 76)
print("SIGN RESTRICTIONS")
print("=" * 76)
print("  repo rises on impact; IIP growth does not rise at months 0, 1, 2;")
print("  inflation unrestricted; the other two shocks unrestricted")
print(f"  draws {N_DRAWS:,}, accepted {identified.n_accepted:,}, "
      f"acceptance rate {identified.acceptance_rate:.2%}")

impact_only = identify_sign_restrictions(
    var,
    (restrictions[0], restrictions[1].model_copy(update={"horizons": (0,)})),
    SHOCKS,
    n_draws=N_DRAWS,
    seed=SEED,
)
repo_only = identify_sign_restrictions(var, restrictions[:1], SHOCKS, n_draws=N_DRAWS, seed=SEED)
print(f"  for scale: repo positive alone accepts {repo_only.acceptance_rate:.2%};")
print(f"    adding IIP on impact only, {impact_only.acceptance_rate:.2%}")
if identified.acceptance_rate < LOW_ACCEPTANCE:
    print("  ACCEPTANCE NEAR ZERO: the restrictions sit badly with the covariance")

# ---- responses per 100bp of repo on impact --------------------------------
repo_paths = identified.response_paths(REPO, "policy", HORIZON)
scale = 1.0 / repo_paths[:, 0]  # repo is in percent, so 1.0 is 100bp
iip = identified.response_paths(IIP, "policy", HORIZON) * scale[:, None]
cpi = identified.response_paths(CPI, "policy", HORIZON) * scale[:, None]
repo_level = np.cumsum(repo_paths * scale[:, None], axis=1)  # repo enters differenced

cholesky = impulse_responses(identify_cholesky(var, ordering=names), horizon=HORIZON)
chol_scale = 1.0 / cholesky.path(REPO, f"{REPO}_shock")[0]
chol_iip = cholesky.path(IIP, f"{REPO}_shock") * chol_scale
chol_cpi = cholesky.path(CPI, f"{REPO}_shock") * chol_scale


def summary(label: str, paths: np.ndarray, reference: np.ndarray | None) -> None:
    print(f"\n  {label}, percentage points per 100bp on impact")
    print(f"  {'h':>4} {'min':>8} {'16%':>8} {'median':>8} {'84%':>8} {'max':>8} "
          f"{'<0 share':>9} {'Cholesky':>9}")
    for h in (0, 1, 2, 3, 6, 9, 12, 18, 24, 36):
        column = paths[:, h]
        q16, q50, q84 = np.quantile(column, (0.16, 0.5, 0.84))
        chol = f"{reference[h]:>+9.3f}" if reference is not None else f"{'':>9}"
        mark = " restricted" if label.startswith("IIP") and h in RESTRICTED else ""
        print(f"  {h:>4} {column.min():>+8.3f} {q16:>+8.3f} {q50:>+8.3f} {q84:>+8.3f} "
              f"{column.max():>+8.3f} {np.mean(column < 0):>9.1%} {chol}{mark}")


print()
print("=" * 76)
print("THE IDENTIFIED SET")
print("=" * 76)
summary("IIP growth", iip, chol_iip)
summary("CPI inflation (unrestricted)", cpi, chol_cpi)
summary("repo rate level (cumulated)", repo_level, None)

lo, hi = WINDOW
cumulative = iip[:, lo : hi + 1].sum(axis=1)
chol_cumulative = float(chol_iip[lo : hi + 1].sum())
share_negative = float(np.mean(cumulative < 0))
q16, q50, q84 = np.quantile(cumulative, (0.16, 0.5, 0.84))
peak_h = np.argmax(np.abs(iip[:, lo:]), axis=1) + lo
peak = iip[np.arange(len(iip)), peak_h]

print()
print("=" * 76)
print(f"CLASSIFICATION: CUMULATIVE IIP GROWTH RESPONSE, MONTHS {lo} TO {hi}")
print("=" * 76)
print(f"  range across accepted draws : [{cumulative.min():+.2f}, {cumulative.max():+.2f}]")
print(f"  16% / median / 84%          : {q16:+.2f} / {q50:+.2f} / {q84:+.2f}")
print(f"  share of draws negative     : {share_negative:.1%}")
print(f"  Cholesky, same window       : {chol_cumulative:+.2f}")
print(f"  largest unrestricted move   : median {np.median(peak):+.2f} at median month "
      f"{int(np.median(peak_h))}; {np.mean(peak > 0):.1%} of draws peak upward")

if identified.acceptance_rate < LOW_ACCEPTANCE:
    verdict = "restrictions nearly inconsistent with the data"
elif share_negative >= MOSTLY:
    verdict = "mostly negative"
elif share_negative <= HARDLY:
    verdict = "positive despite the restriction"
else:
    verdict = "straddles zero"
print(f"\n  outcome: {verdict}")
print("  the set is conditional on the VAR's point estimates; the VAR fails its")
print("  autocorrelation diagnostics for the mechanical reason in ADR 006")

# ---- added after the first run --------------------------------------------
# The per-100bp scaling above was fixed in advance and turned out to explode
# where a draw barely moves the repo rate. Signs are unaffected, since every
# accepted draw raises the repo rate on impact, so the classification stands.
# These diagnostics were added to show what the accepted shocks are.

print()
print("=" * 76)
print("WHAT THE ACCEPTED SHOCKS ARE (added after the first run)")
print("=" * 76)
repo_sd = identified.response_paths(REPO, "policy", 0)[:, 0]
iip_sd = identified.response_paths(IIP, "policy", 0)[:, 0]
print("  per one standard deviation shock, 16% / median / 84%:")
print("    repo impact, bp           : "
      + " / ".join(f"{v * 100:.1f}" for v in np.quantile(repo_sd, (0.16, 0.5, 0.84))))
print("    IIP growth impact, pp     : "
      + " / ".join(f"{v:+.2f}" for v in np.quantile(iip_sd, (0.16, 0.5, 0.84))))

residuals = var.residuals
dated = periods[len(periods) - len(residuals):]
sd = residuals.std(axis=0)
print(f"  residual sd: IIP {sd[0]:.2f}pp, CPI {sd[1]:.2f}pp, repo {sd[2] * 100:.1f}bp")
pandemic = np.array([date(2020, 3, 1) <= p <= date(2021, 12, 1) for p in dated])
share = float((residuals[pandemic, 0] ** 2).sum() / (residuals[:, 0] ** 2).sum())
print(f"  IIP residual variance from March 2020 to December 2021: {share:.0%} "
      f"of the total, in {pandemic.sum()} of {len(dated)} months")
print("  a shock that moves the repo rate a few basis points and IIP growth by")
print("  several points is mostly output variance, not monetary policy")
