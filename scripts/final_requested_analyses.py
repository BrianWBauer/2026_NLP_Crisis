"""Final requested analyses for the NLP Crisis EMA manuscript.

This script intentionally implements only the two analyses requested for the
final manuscript revision:
  1) contextual-effect contrasts (beta_between - beta_within) for PC1-PC5
     using the primary negative-binomial model and participant-clustered vcov;
  2) a VADER sentiment baseline, both as a standalone grouped-CV Ridge model
     and as within-/between-person covariates in the primary NB model.

Inputs are the existing local processed file used by notebook 05:
    ../data/processed/pm_day_features.csv

No raw text or participant-level data are written to the repository by this
script. Output tables are written under ../outputs/tables when the script is
run in the project environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -----------------------------------------------------------------------------
# Configuration: matches the existing project notebooks/manuscript
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pm_day_features.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PID_COL = "expiwell_id_clean"
TEXT_COL = "pm_day_text"
CRISIS_COL = "crisis_PM_from_full"

N_SPLITS = 5
RIDGE_ALPHA = 10.0
RANDOM_SEED = 7

# Orient PC2 and PC3 so the crisis-relevant pole is positive in all final output.
# Multiplying both the within- and between-person terms by -1 leaves fitted
# values/model fit unchanged; only coefficient orientation changes.
PC_ORIENTATION = {"PC1": 1.0, "PC2": -1.0, "PC3": -1.0, "PC4": 1.0, "PC5": 1.0}
PC_COLS = [f"PC{i}_{part}" for i in range(1, 6) for part in ("within", "between")]

np.random.seed(RANDOM_SEED)


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def crisis_orient_pcs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for pc, sign in PC_ORIENTATION.items():
        for part in ("within", "between"):
            col = f"{pc}_{part}"
            out[col] = pd.to_numeric(out[col], errors="coerce") * sign
    return out


def fit_primary_nb(df: pd.DataFrame):
    """Fit the manuscript's primary NB GLM and participant-clustered vcov."""
    use = df[[PID_COL, CRISIS_COL] + PC_COLS].copy()
    use[CRISIS_COL] = pd.to_numeric(use[CRISIS_COL], errors="coerce")
    for c in PC_COLS:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=[PID_COL, CRISIS_COL] + PC_COLS).copy()

    y = use[CRISIS_COL].to_numpy(dtype=float)
    groups = use[PID_COL].astype(str).to_numpy()
    X = sm.add_constant(use[PC_COLS].astype(float), has_constant="add")

    # Same method-of-moments approximation used in notebook 05/manuscript.
    pois = sm.GLM(y, X, family=sm.families.Poisson())
    res_pois = pois.fit(cov_type="cluster", cov_kwds={"groups": groups})
    alpha_hat = max((res_pois.pearson_chi2 / res_pois.df_resid) - 1.0, 1e-6)

    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat))
    res_nb = nb.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return use, res_nb, float(alpha_hat)


def contextual_effect_table(res_nb) -> pd.DataFrame:
    """Compute beta_between - beta_within using the clustered covariance matrix."""
    V = res_nb.cov_params()
    rows: list[dict[str, float | str]] = []
    for i in range(1, 6):
        pc = f"PC{i}"
        w = f"{pc}_within"
        b = f"{pc}_between"
        contrast = float(res_nb.params[b] - res_nb.params[w])
        var_contrast = float(V.loc[b, b] + V.loc[w, w] - 2.0 * V.loc[b, w])
        se = float(np.sqrt(max(var_contrast, 0.0)))
        z = contrast / se if se > 0 else np.nan
        p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append({"PC": pc, "contrast_B_minus_W": contrast, "SE": se, "z": z, "p": p})
    return pd.DataFrame(rows)


def vader_scores(texts: pd.Series) -> np.ndarray:
    analyzer = SentimentIntensityAnalyzer()
    return np.array(
        [analyzer.polarity_scores(str(t))["compound"] for t in texts.fillna("")],
        dtype=float,
    )


def fisher_r_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Conventional Fisher-z CI, matching the manuscript's simple r reporting."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    crit = norm.ppf(1.0 - (1.0 - level) / 2.0)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def sentiment_ridge(df: pd.DataFrame) -> pd.DataFrame:
    """Standalone VADER compound -> PM crisis total under existing Ridge/GroupKFold CV."""
    use = df[[PID_COL, CRISIS_COL, "vader_compound"]].copy()
    use[CRISIS_COL] = pd.to_numeric(use[CRISIS_COL], errors="coerce")
    use["vader_compound"] = pd.to_numeric(use["vader_compound"], errors="coerce")
    use = use.dropna(subset=[PID_COL, CRISIS_COL, "vader_compound"]).copy()

    X = use[["vader_compound"]].to_numpy(dtype=float)
    y = use[CRISIS_COL].to_numpy(dtype=float)
    groups = use[PID_COL].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    yhat = np.full(len(y), np.nan, dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)),
            ]
        )
        pipe.fit(X[tr], y[tr])
        yhat[te] = pipe.predict(X[te])

    r = float(np.corrcoef(y, yhat)[0, 1])
    ci_lo, ci_hi = fisher_r_ci(r, len(y))
    return pd.DataFrame(
        [
            {
                "pathway": "VADER sentiment -> PM crisis symptoms",
                "n": int(len(y)),
                "participants": int(use[PID_COL].nunique()),
                "pearson_r": r,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
                "R2": float(r2_score(y, yhat)),
                "MAE": float(mean_absolute_error(y, yhat)),
                "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
                "ridge_alpha": RIDGE_ALPHA,
                "group_folds": gkf.n_splits,
            }
        ]
    )


def add_sentiment_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    person_mean = out.groupby(PID_COL)["vader_compound"].transform("mean")
    out["sentiment_between"] = person_mean
    out["sentiment_within"] = out["vader_compound"] - person_mean
    return out


def sentiment_adjusted_nb(df: pd.DataFrame, alpha_hat_primary: float, primary_res) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add within/between VADER sentiment to the primary NB specification."""
    sent_cols = ["sentiment_within", "sentiment_between"]
    cols = PC_COLS + sent_cols
    use = df[[PID_COL, CRISIS_COL] + cols].copy()
    use[CRISIS_COL] = pd.to_numeric(use[CRISIS_COL], errors="coerce")
    for c in cols:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=[PID_COL, CRISIS_COL] + cols).copy()

    y = use[CRISIS_COL].to_numpy(dtype=float)
    groups = use[PID_COL].astype(str).to_numpy()
    X = sm.add_constant(use[cols].astype(float), has_constant="add")

    # Hold alpha at the primary-model value so this is a direct covariate addition.
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat_primary))
    res = nb.fit(cov_type="cluster", cov_kwds={"groups": groups})

    rows = []
    ci = res.conf_int()
    for term in cols:
        rows.append(
            {
                "term": term,
                "beta": float(res.params[term]),
                "SE": float(res.bse[term]),
                "z": float(res.tvalues[term]),
                "p": float(res.pvalues[term]),
                "CI_lo": float(ci.loc[term, 0]),
                "CI_hi": float(ci.loc[term, 1]),
            }
        )
    full = pd.DataFrame(rows)

    compare_rows = []
    for term in PC_COLS:
        p0 = float(primary_res.pvalues[term])
        p1 = float(res.pvalues[term])
        compare_rows.append(
            {
                "term": term,
                "primary_beta": float(primary_res.params[term]),
                "primary_p": p0,
                "sentiment_adjusted_beta": float(res.params[term]),
                "sentiment_adjusted_p": p1,
                "primary_sig_p_lt_05": p0 < 0.05,
                "adjusted_sig_p_lt_05": p1 < 0.05,
                "significance_holds": (p0 < 0.05) == (p1 < 0.05),
            }
        )
    compare = pd.DataFrame(compare_rows)
    return full, compare


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Required protected/local data file not found: {DATA_PATH}\n"
            "The repository intentionally does not contain the analytic data. "
            "Run this script in the existing project environment where data/processed/pm_day_features.csv is available."
        )

    df = pd.read_csv(DATA_PATH)
    require_columns(df, [PID_COL, TEXT_COL, CRISIS_COL] + PC_COLS)
    if len(df) != 2511:
        raise ValueError(f"Expected 2,511 analytic entries; found {len(df):,}.")

    # Final manuscript orientation.
    df = crisis_orient_pcs(df)

    # 1) Contextual effects.
    primary_use, primary_res, alpha_hat = fit_primary_nb(df)
    contextual = contextual_effect_table(primary_res)
    contextual.to_csv(OUT_DIR / "contextual_effect_contrasts.csv", index=False)

    # 2) VADER sentiment baseline.
    df["vader_compound"] = vader_scores(df[TEXT_COL])
    pd.DataFrame(
        {"row_id": np.arange(len(df), dtype=int), "vader_compound": df["vader_compound"].to_numpy()}
    ).to_csv(OUT_DIR / "vader_sentiment_scores.csv", index=False)

    ridge = sentiment_ridge(df)
    ridge.to_csv(OUT_DIR / "sentiment_baseline_ridge.csv", index=False)

    df = add_sentiment_decomposition(df)
    sent_nb, pc_compare = sentiment_adjusted_nb(df, alpha_hat, primary_res)
    sent_nb.to_csv(OUT_DIR / "sentiment_adjusted_nb.csv", index=False)
    pc_compare.to_csv(OUT_DIR / "sentiment_pc_comparison.csv", index=False)

    print(f"Primary NB N={len(primary_use):,}; alpha={alpha_hat:.6f}")
    print("\nContextual effects (beta_B - beta_W):")
    print(contextual.to_string(index=False))
    print("\nStandalone VADER Ridge:")
    print(ridge.to_string(index=False))
    print("\nSentiment-adjusted NB:")
    print(sent_nb.to_string(index=False))
    print("\nPC coefficient comparison:")
    print(pc_compare.to_string(index=False))
    print(f"\nSaved requested outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
