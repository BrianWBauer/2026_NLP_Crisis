"""Analyses requested in peer/collaborator review of the manuscript.

Implements three requested items:

  6a) How much of the Aim 1 cross-validated prediction accuracy is carried by the
      first five principal components alone, relative to the full 20-component
      set used in the primary prediction models?

  6b) Why does the in-sample OLS R-squared for the five-component within/between
      model exceed the out-of-sample cross-validated performance of the
      20-component prediction model? Quantifies the optimism gap directly by
      cross-validating the same 10-term specification.

  11) Is the negative binomial preferred over zero-inflated and hurdle
      alternatives, given that ~32% of person-days score zero? Compares NB,
      ZINB, and NB hurdle by AIC/BIC and by predicted zero counts.

Modeling conventions follow the primary analyses: PM-window diary embeddings,
PCA and StandardScaler fit inside each training fold, Ridge(alpha=10),
participant-grouped 5-fold cross-validation, and pooled out-of-fold predictions.

Requires the protected local data. Writes derived tables to outputs/tables/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.truncated_model import HurdleCountModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "pm_day_features.csv"
EMBED_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ema_personality_plus_surveys_merged.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

PID_COL = "expiwell_id_clean"
DATE_COL = "ema_date"
CRISIS_COL = "crisis_PM_from_full"
DSI_COL = "dsi_PM_total"
AM_CRISIS_TOTAL = "dailyAM__Total Score from 5 Questions"

PC_ORIENTATION = {"PC1": 1.0, "PC2": -1.0, "PC3": -1.0, "PC4": 1.0, "PC5": 1.0}
PC_COLS = [f"PC{i}_{p}" for i in range(1, 6) for p in ("within", "between")]

N_SPLITS = 5
RIDGE_ALPHA = 10.0
RANDOM_SEED = 7
PC_GRID = [5, 10, 20]

np.random.seed(RANDOM_SEED)


def fisher_r_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z, crit, se = np.arctanh(r), norm.ppf(1 - (1 - level) / 2), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_pcs: int | None) -> np.ndarray:
    """Participant-grouped OOF predictions; PCA/scaler fit on training folds only.

    n_pcs=None skips PCA (used to cross-validate the 10-term PC specification).
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    yhat = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=groups):
        assert not (set(groups[tr]) & set(groups[te]))
        X_tr, X_te = X[tr], X[te]
        if n_pcs is not None:
            pca = PCA(n_components=n_pcs, random_state=RANDOM_SEED)
            X_tr, X_te = pca.fit_transform(X_tr), pca.transform(X_te)
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
        yhat[te] = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED).fit(X_tr, y[tr]).predict(X_te)
    assert not np.isnan(yhat).any()
    return yhat


def metrics(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray, **extra) -> dict:
    r = float(np.corrcoef(y, yhat)[0, 1])
    lo, hi = fisher_r_ci(r, len(y))
    return {
        **extra,
        "n": int(len(y)),
        "participants": int(len(np.unique(groups))),
        "pearson_r": r,
        "CI_lo": lo,
        "CI_hi": hi,
        "R2": float(r2_score(y, yhat)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
    }


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    feat = pd.read_csv(FEATURES_PATH)
    emb = np.load(EMBED_PATH)
    feat[DATE_COL] = pd.to_datetime(feat[DATE_COL]).dt.normalize()
    assert len(feat) == emb.shape[0] == 2511
    for pc, sign in PC_ORIENTATION.items():
        for part in ("within", "between"):
            feat[f"{pc}_{part}"] = pd.to_numeric(feat[f"{pc}_{part}"], errors="coerce") * sign
    return feat, emb


def next_day_outcome(feat: pd.DataFrame) -> pd.DataFrame:
    """AM crisis total on exactly ema_date + 1 day."""
    raw = pd.read_csv(RAW_PATH)
    raw["ema_dt"] = pd.to_datetime(raw["ema_dt"], errors="coerce")
    raw[DATE_COL] = raw["ema_dt"].dt.normalize()
    fnn = lambda s: (s.dropna().iloc[0] if s.notna().any() else np.nan)
    am = (raw.sort_values([PID_COL, "ema_dt"])
            .groupby([PID_COL, DATE_COL], as_index=False)
            .agg(crisis_AM_total=("crisis_AM_total" if "crisis_AM_total" in raw else AM_CRISIS_TOTAL, fnn)))
    am = am.loc[am["crisis_AM_total"].notna()].rename(columns={DATE_COL: "target_date"})
    link = feat[[PID_COL, DATE_COL]].copy()
    link["row"] = np.arange(len(link))
    link["target_date"] = link[DATE_COL] + pd.Timedelta(days=1)
    mg = link.merge(am, on=[PID_COL, "target_date"], how="left").sort_values("row")
    elig = mg.loc[mg["crisis_AM_total"].notna()].copy()
    assert ((elig["target_date"] - elig[DATE_COL]).dt.days == 1).all()
    return elig


# -----------------------------------------------------------------------------
# 6a) Component-count comparison
# -----------------------------------------------------------------------------
def analysis_6a(feat: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    g_all = feat[PID_COL].astype(str).to_numpy()
    rows = []

    for outcome, label in [(CRISIS_COL, "Crisis symptoms, concurrent"),
                           (DSI_COL, "DSI-SS, concurrent")]:
        y = pd.to_numeric(feat[outcome], errors="coerce").to_numpy(float)
        for k in PC_GRID:
            rows.append(metrics(y, cv_predict(emb, y, g_all, k), g_all,
                                pathway=label, n_components=k))

    elig = next_day_outcome(feat)
    idx = elig["row"].to_numpy()
    y = elig["crisis_AM_total"].to_numpy(float)
    g = elig[PID_COL].astype(str).to_numpy()
    for k in PC_GRID:
        rows.append(metrics(y, cv_predict(emb[idx], y, g, k), g,
                            pathway="Crisis symptoms, prospective", n_components=k))

    out = pd.DataFrame(rows)
    # Share of the 20-component performance retained by each smaller set.
    ref = out[out["n_components"] == 20].set_index("pathway")["pearson_r"]
    out["pct_of_20PC_r"] = out.apply(
        lambda r_: 100.0 * r_["pearson_r"] / ref[r_["pathway"]], axis=1)
    return out


# -----------------------------------------------------------------------------
# 6b) In-sample vs out-of-sample fit for the 10-term PC specification
# -----------------------------------------------------------------------------
def analysis_6b(feat: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    y = pd.to_numeric(feat[CRISIS_COL], errors="coerce").to_numpy(float)
    g = feat[PID_COL].astype(str).to_numpy()
    Xpc = feat[PC_COLS].astype(float).to_numpy()

    # In-sample OLS on the 10 within/between PC terms (the Table 3 sensitivity R^2).
    ols = sm.OLS(y, sm.add_constant(Xpc)).fit()

    # Same specification, participant-grouped cross-validation.
    yhat_cv = cv_predict(Xpc, y, g, n_pcs=None)

    # Within-person-only and between-person-only in-sample OLS, to show how much
    # of the in-sample R^2 comes from person-level means.
    w_idx = [i for i, c in enumerate(PC_COLS) if c.endswith("_within")]
    b_idx = [i for i, c in enumerate(PC_COLS) if c.endswith("_between")]
    ols_w = sm.OLS(y, sm.add_constant(Xpc[:, w_idx])).fit()
    ols_b = sm.OLS(y, sm.add_constant(Xpc[:, b_idx])).fit()

    yhat20 = cv_predict(emb, y, g, 20)
    yhat5 = cv_predict(emb, y, g, 5)

    return pd.DataFrame([
        {"quantity": "In-sample OLS R2, 10 PC terms (5 within + 5 between)",
         "value": float(ols.rsquared), "evaluation": "in-sample, all 2,511 rows"},
        {"quantity": "In-sample OLS R2, 5 within-person terms only",
         "value": float(ols_w.rsquared), "evaluation": "in-sample"},
        {"quantity": "In-sample OLS R2, 5 between-person terms only",
         "value": float(ols_b.rsquared), "evaluation": "in-sample"},
        {"quantity": "Cross-validated R2, same 10 PC terms",
         "value": float(r2_score(y, yhat_cv)), "evaluation": "participant-grouped OOF"},
        {"quantity": "Cross-validated Pearson r, same 10 PC terms",
         "value": float(np.corrcoef(y, yhat_cv)[0, 1]), "evaluation": "participant-grouped OOF"},
        {"quantity": "Cross-validated R2, 20-component embedding Ridge",
         "value": float(r2_score(y, yhat20)), "evaluation": "participant-grouped OOF"},
        {"quantity": "Cross-validated Pearson r, 20-component embedding Ridge",
         "value": float(np.corrcoef(y, yhat20)[0, 1]), "evaluation": "participant-grouped OOF"},
        {"quantity": "Cross-validated R2, 5-component embedding Ridge",
         "value": float(r2_score(y, yhat5)), "evaluation": "participant-grouped OOF"},
        {"quantity": "Cross-validated Pearson r, 5-component embedding Ridge",
         "value": float(np.corrcoef(y, yhat5)[0, 1]), "evaluation": "participant-grouped OOF"},
    ])


# -----------------------------------------------------------------------------
# 11) NB vs zero-inflated NB vs hurdle
# -----------------------------------------------------------------------------
def analysis_11(feat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare NB against zero-inflated and hurdle alternatives.

    All four models are fit by maximum likelihood on the same 10 PC terms so
    that log-likelihoods, AIC, and BIC are directly comparable. Note that these
    fits use model-based standard errors; the manuscript's primary NB GLM uses
    participant-clustered robust standard errors, so p-values from these fits
    are not comparable to the primary model's and are anti-conservative under
    repeated measures.
    """
    y = pd.to_numeric(feat[CRISIS_COL], errors="coerce").to_numpy(float)
    X = sm.add_constant(feat[PC_COLS].astype(float), has_constant="add").to_numpy(float)
    ones = np.ones((len(y), 1))

    fits = {
        "NB": NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=500),
        "ZINB (intercept-only inflation)": ZeroInflatedNegativeBinomialP(
            y, X, exog_infl=ones, p=2).fit(disp=0, maxiter=1000),
        "ZINB (covariate inflation)": ZeroInflatedNegativeBinomialP(
            y, X, exog_infl=X, p=2).fit(disp=0, maxiter=1000),
        "NB hurdle": HurdleCountModel(
            y, X, dist="negbin", zerodist="poisson").fit(disp=0, maxiter=500),
    }

    def nb_expected_zeros(mu, alpha):
        n = 1.0 / alpha
        return float(np.sum((n / (n + mu)) ** n))

    rows = []
    for name, res in fits.items():
        k = int(len(np.atleast_1d(res.params)))
        try:
            if name == "NB":
                ez = nb_expected_zeros(res.predict(), res.params[-1])
            else:
                ez = float(res.predict(which="prob")[:, 0].sum())
        except Exception:
            ez = np.nan
        rows.append({
            "model": name,
            "k_params": k,
            "loglik": float(res.llf),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "expected_zeros": ez,
            "converged": bool(res.mle_retvals.get("converged", True)),
        })

    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out["dAIC_vs_best"] = out["AIC"] - out["AIC"].min()
    out["dBIC_vs_best"] = out["BIC"] - out["BIC"].min()

    n_zero = int((y == 0).sum())
    out["zero_error"] = out["expected_zeros"] - n_zero
    return out, {"observed_zeros": n_zero, "n": int(len(y)),
                 "pct_zero": 100.0 * n_zero / len(y)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat, emb = load_data()

    print("=" * 78); print("6a) COMPONENT COUNT: PC1-5 vs 10 vs 20"); print("=" * 78)
    t6a = analysis_6a(feat, emb)
    print(t6a.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t6a.to_csv(OUT_DIR / "reviewer_6a_component_count.csv", index=False)

    print(); print("=" * 78); print("6b) IN-SAMPLE vs OUT-OF-SAMPLE FIT"); print("=" * 78)
    t6b = analysis_6b(feat, emb)
    print(t6b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t6b.to_csv(OUT_DIR / "reviewer_6b_insample_vs_cv.csv", index=False)

    print(); print("=" * 78); print("11) NB vs ZERO-INFLATED NB vs HURDLE"); print("=" * 78)
    t11, info = analysis_11(feat)
    print(f"  Observed zeros: {info['observed_zeros']}/{info['n']} ({info['pct_zero']:.1f}%)")
    print(t11.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    t11.to_csv(OUT_DIR / "reviewer_11_count_model_comparison.csv", index=False)

    print(f"\nSaved tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
