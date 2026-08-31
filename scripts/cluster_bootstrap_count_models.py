"""Participant-cluster bootstrap for the zero-inflated / hurdle sensitivity analysis.

The manuscript's primary negative binomial GLM uses participant-clustered robust
standard errors. statsmodels' ZINB and hurdle implementations report only
model-based standard errors, which are anti-conservative under repeated measures
(~20 person-days per participant) and therefore not comparable to the primary
model's inference.

This script refits each count model on bootstrap samples drawn by resampling
PARTICIPANTS with replacement (the cluster bootstrap), so that standard errors
and confidence intervals for all specifications account for within-person
dependence on the same footing.

Reports, for each of the 10 within-/between-person PC terms: the full-sample
point estimate, the cluster-bootstrap SE, the percentile 95% CI, and a
bootstrap p-value. The negative binomial is bootstrapped alongside the two-part
models as a check: its bootstrap SEs should track the analytic cluster-robust
SEs from the primary GLM.

Usage:
    python scripts/cluster_bootstrap_count_models.py [--reps 1000]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.truncated_model import HurdleCountModel

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pm_day_features.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

PID_COL = "expiwell_id_clean"
CRISIS_COL = "crisis_PM_from_full"
PC_ORIENTATION = {"PC1": 1.0, "PC2": -1.0, "PC3": -1.0, "PC4": 1.0, "PC5": 1.0}
PC_COLS = [f"PC{i}_{p}" for i in range(1, 6) for p in ("within", "between")]
SEED = 7


def load():
    f = pd.read_csv(DATA_PATH)
    for pc, s in PC_ORIENTATION.items():
        for part in ("within", "between"):
            f[f"{pc}_{part}"] = pd.to_numeric(f[f"{pc}_{part}"], errors="coerce") * s
    return f


def count_part_coefs(y: np.ndarray, X: np.ndarray, model: str) -> np.ndarray | None:
    """Return the 10 PC coefficients from each model's count/severity component."""
    try:
        if model == "NB":
            r = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=500)
            return np.asarray(r.params)[1:11]      # skip intercept, drop alpha
        if model == "ZINB":
            r = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X, p=2).fit(disp=0, maxiter=1000)
            k = X.shape[1]
            return np.asarray(r.params)[k + 1:2 * k]      # skip inflation block and count intercept
        if model == "Hurdle":
            r = HurdleCountModel(y, X, dist="negbin", zerodist="poisson").fit(disp=0, maxiter=500)
            return np.asarray(r.results_count.params)[1:11]
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    f = load()
    y_all = pd.to_numeric(f[CRISIS_COL], errors="coerce").to_numpy(float)
    X_all = sm.add_constant(f[PC_COLS].astype(float), has_constant="add").to_numpy(float)
    pids = f[PID_COL].astype(str).to_numpy()
    uniq = np.unique(pids)
    idx_by_pid = {p: np.flatnonzero(pids == p) for p in uniq}

    models = ["NB", "ZINB", "Hurdle"]
    point = {m: count_part_coefs(y_all, X_all, m) for m in models}
    for m in models:
        if point[m] is None:
            raise RuntimeError(f"Full-sample fit failed for {m}")

    rng = np.random.default_rng(SEED)
    draws = {m: [] for m in models}
    fails = {m: 0 for m in models}

    for b in range(args.reps):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_pid[p] for p in take])
        yb, Xb = y_all[rows], X_all[rows]
        for m in models:
            c = count_part_coefs(yb, Xb, m)
            if c is None or not np.all(np.isfinite(c)):
                fails[m] += 1
            else:
                draws[m].append(c)
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{args.reps} replicates", flush=True)

    # Analytic cluster-robust NB GLM (the manuscript's primary model) for reference.
    Xd = sm.add_constant(f[PC_COLS].astype(float), has_constant="add")
    pois = sm.GLM(y_all, Xd, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pids})
    alpha = max(pois.pearson_chi2 / pois.df_resid - 1.0, 1e-6)
    nb_glm = sm.GLM(y_all, Xd, family=sm.families.NegativeBinomial(alpha=alpha)).fit(
        cov_type="cluster", cov_kwds={"groups": pids})

    rows = []
    for m in models:
        D = np.vstack(draws[m])
        np.save(OUT_DIR / f"reviewer_11b_draws_{m}.npy", D)
        for j, term in enumerate(PC_COLS):
            est = float(point[m][j])
            col = D[:, j]
            se = float(col.std(ddof=1))
            # Robust scale from the interquartile range. When the optimizer fails
            # on a minority of resamples the naive SD is destroyed by outliers,
            # while the IQR-based scale still describes the bulk of the draws.
            q25, q75 = np.percentile(col, [25, 75])
            se_robust = float((q75 - q25) / 1.349)
            unstable = float(np.mean(np.abs(col - est) > 2.0))
            lo, hi = np.percentile(col, [2.5, 97.5])
            z = est / se if se > 0 else np.nan
            rows.append({
                "model": m, "term": term, "beta": est,
                "boot_SE": se, "boot_SE_robust": se_robust,
                "frac_unstable_reps": unstable,
                "boot_median": float(np.median(col)),
                "boot_CI_lo": float(lo), "boot_CI_hi": float(hi),
                "z": z, "p": float(2 * norm.sf(abs(z))),
                "sig_05": bool(2 * norm.sf(abs(z)) < 0.05),
                "n_reps_used": int(D.shape[0]),
            })
    out = pd.DataFrame(rows)

    print("\nOptimizer stability (share of resamples deviating >2.0 from the point estimate):")
    for m in models:
        sub = out[out.model == m]
        print(f"  {m:<8} max across terms = {sub['frac_unstable_reps'].max():.1%}"
              f"   median naive SE = {sub['boot_SE'].median():.3f}"
              f"   median robust SE = {sub['boot_SE_robust'].median():.3f}")

    prim = pd.DataFrame({
        "term": PC_COLS,
        "primary_beta": [float(nb_glm.params[t]) for t in PC_COLS],
        "primary_clusterrobust_SE": [float(nb_glm.bse[t]) for t in PC_COLS],
        "primary_p": [float(nb_glm.pvalues[t]) for t in PC_COLS],
        "primary_sig_05": [bool(nb_glm.pvalues[t] < 0.05) for t in PC_COLS],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "reviewer_11b_cluster_bootstrap.csv", index=False)
    prim.to_csv(OUT_DIR / "reviewer_11b_primary_reference.csv", index=False)

    print("\nBootstrap replicates used / failed:")
    for m in models:
        print(f"  {m:<8} used={len(draws[m]):<5} failed={fails[m]}")

    print("\n=== Primary NB GLM (analytic cluster-robust) ===")
    print(prim.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m in models:
        print(f"\n=== {m} (participant-cluster bootstrap) ===")
        sub = out[out["model"] == m][["term", "beta", "boot_SE", "boot_CI_lo", "boot_CI_hi", "p", "sig_05"]]
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Significance agreement with primary NB (p<.05) ===")
    for m in models:
        sub = out[out["model"] == m].set_index("term")
        agree = sum(bool(sub.loc[t, "sig_05"]) == bool(prim.set_index("term").loc[t, "primary_sig_05"])
                    for t in PC_COLS)
        flips = sum(np.sign(sub.loc[t, "beta"]) != np.sign(prim.set_index("term").loc[t, "primary_beta"])
                    for t in PC_COLS)
        print(f"  {m:<8} same significance status: {agree}/10   sign flips: {flips}/10")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
