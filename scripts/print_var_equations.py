"""Print the headline VAR(12) as equations: coefficients, standard errors, fit.

Same data and settings as run_pipeline.py: INDPRO, CPIAUCSL, FEDFUNDS from FRED,
1985-01 to 2007-06, prepared by the unit rule, 12 lags.
Run from C:\\dev\\moirai with the project's venv.
"""

from __future__ import annotations

import numpy as np

from moirai.core.logging import configure_logging
from moirai.engine.causal.preparation import align, prepare
from moirai.engine.causal.var import _build_design, estimate_var
from moirai.engine.data_fabric.ingestion.fred import FredAdapter

START, END = "1985-01-01", "2007-06-01"
CODES = ("INDPRO", "CPIAUCSL", "FEDFUNDS")
N_LAGS = 12
LABEL = {
    "indpro": "output growth (monthly log change of IP)",
    "cpiaucsl": "inflation (monthly log change of CPI)",
    "fedfunds": "change in the funds rate (percentage points)",
}


def report(data: np.ndarray, names: tuple[str, ...], periods: tuple) -> None:
    var = estimate_var(data, names, n_lags=N_LAGS, periods=periods)
    x, y = _build_design(data, N_LAGS)
    k, p = len(names), N_LAGS
    xtx_inv = np.linalg.inv(x.T @ x)
    dof = x.shape[0] - x.shape[1]

    print(f"VAR({p}) on {list(names)}: {var.n_observations} observations, "
          f"{1 + k * p} regressors per equation, {k * (1 + k * p)} coefficients in all\n")

    for i, dep in enumerate(names):
        resid = var.residuals[:, i]
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        beta = np.concatenate([[var.intercept[i]], np.concatenate(
            [var.coefficients[lag][i] for lag in range(p)])])
        tss = float(((y[:, i] - y[:, i].mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / tss

        print("=" * 78)
        print(f"EQUATION {i + 1}: dependent variable = {dep}_t, {LABEL[dep]}")
        print(f"R^2 {r2:.3f}   residual sd {np.sqrt(sigma2):.6f}")
        print("=" * 78)
        t0 = beta[0] / se[0]
        print(f"  constant                  {beta[0]:+.6f}   se {se[0]:.6f}   t {t0:+.2f}")
        print(f"  {'lag':>3}" + "".join(f"   {n + '_(t-lag)':>24}" for n in names))
        for lag in range(p):
            cells = []
            for j in range(k):
                col = 1 + lag * k + j
                cells.append(f"{beta[col]:+.4f} (t {beta[col] / se[col]:+5.2f})")
            print(f"  {lag + 1:>3}   " + "   ".join(f"{c:>24}" for c in cells))
        sums = [sum(var.coefficients[lag][i][j] for lag in range(p)) for j in range(k)]
        print("  sum of the 12 lag coefficients, by regressor: "
              + ", ".join(f"{n} {s:+.4f}" for n, s in zip(names, sums, strict=True)))
        print()

    corr = np.corrcoef(var.residuals, rowvar=False)
    print("RESIDUAL COVARIANCE AND CORRELATION (the identification problem)")
    for i, a in enumerate(names):
        print(f"  {a:<10} " + " ".join(f"{var.sigma_u[i, j]:+.3e}" for j in range(k))
              + "    corr " + " ".join(f"{corr[i, j]:+.3f}" for j in range(k)))
    print(f"\nmax eigenvalue modulus {var.stability.max_modulus:.3f}, "
          f"half life {var.stability.half_life:.1f} months")


if __name__ == "__main__":
    configure_logging("ERROR")
    with FredAdapter() as fred:
        raw = [fred.fetch_series(c, observation_start=START, observation_end=END) for c in CODES]
    prepared = [prepare(s) for s in raw]
    periods, data = align(prepared)
    report(data, tuple(s.series_id for s in prepared), periods)
