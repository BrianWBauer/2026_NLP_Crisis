"""Two-part decomposition of the crisis-symptom model under a hurdle specification.

The primary negative binomial GLM estimates a single coefficient per predictor.
A hurdle model separates two questions that the single-equation model collapses:
whether a participant reports ANY crisis symptoms on a given day (zero component),
and how severe symptoms are GIVEN that some occurred (count component).

Both components are reported with participant-cluster bootstrap standard errors so
that inference is comparable to the primary model's analytic clustered errors.
"""

from __future__ import annotations

import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from statsmodels.discrete.truncated_model import HurdleCountModel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pm_day_features.csv"
OUT = ROOT / "outputs" / "tables"
PID, CRISIS = "expiwell_id_clean", "crisis_PM_from_full"
ORIENT = {"PC1": 1.0, "PC2": -1.0, "PC3": -1.0, "PC4": 1.0, "PC5": 1.0}
PC = [f"PC{i}_{p}" for i in range(1, 6) for p in ("within", "between")]
SEED = 7


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=1000)
    reps = ap.parse_args().reps

    f = pd.read_csv(DATA)
    for pc, s in ORIENT.items():
        for part in ("within", "between"):
            f[f"{pc}_{part}"] = pd.to_numeric(f[f"{pc}_{part}"], errors="coerce") * s
    y = f[CRISIS].to_numpy(float)
    X = sm.add_constant(f[PC].astype(float), has_constant="add").to_numpy(float)
    pid = f[PID].astype(str).to_numpy()
    uniq = np.unique(pid)
    idx = {p: np.flatnonzero(pid == p) for p in uniq}

    def fit(yy, XX):
        r = HurdleCountModel(yy, XX, dist="negbin", zerodist="poisson").fit(disp=0, maxiter=500)
        return (np.asarray(r.results_zero.params)[1:11],
                np.asarray(r.results_count.params)[1:11])

    pz, pc_ = fit(y, X)
    rng = np.random.default_rng(SEED)
    Z, C = [], []
    for b in range(reps):
        rows = np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq), True)])
        try:
            z, c = fit(y[rows], X[rows]); Z.append(z); C.append(c)
        except Exception:
            pass
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{reps}", flush=True)
    Z, C = np.vstack(Z), np.vstack(C)

    # primary NB GLM for reference
    Xd = sm.add_constant(f[PC].astype(float), has_constant="add")
    p0 = sm.GLM(y, Xd, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pid})
    a = max(p0.pearson_chi2 / p0.df_resid - 1.0, 1e-6)
    nb = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=a)).fit(
        cov_type="cluster", cov_kwds={"groups": pid})

    rows = []
    for j, t in enumerate(PC):
        for lbl, pt, D in [("zero (any symptoms)", pz[j], Z[:, j]),
                           ("count (severity | any)", pc_[j], C[:, j])]:
            se = float(D.std(ddof=1)); z = pt / se; p = float(2 * norm.sf(abs(z)))
            rows.append({"term": t, "component": lbl, "beta": float(pt), "boot_SE": se,
                         "p": p, "sig_05": p < 0.05,
                         "primary_nb_beta": float(nb.params[t]),
                         "primary_nb_p": float(nb.pvalues[t]),
                         "primary_nb_sig_05": bool(nb.pvalues[t] < 0.05)})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "reviewer_11c_hurdle_decomposition.csv", index=False)

    print(f"\nHurdle decomposition, {len(Z)} bootstrap replicates\n")
    print(out[["term", "component", "beta", "boot_SE", "p", "sig_05"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    sig_any = {t: bool(out[(out.term == t) & out.sig_05].shape[0]) for t in PC}
    nb_sig = {t: bool(nb.pvalues[t] < 0.05) for t in PC}
    preserved = sum(sig_any[t] for t in PC if nb_sig[t])
    print(f"\nNB-significant terms also significant in >=1 hurdle component: "
          f"{preserved}/{sum(nb_sig.values())}")


if __name__ == "__main__":
    main()
