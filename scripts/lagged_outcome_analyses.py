"""Autocorrelation descriptives and AR(1)-benchmarked prediction models.

Addresses the reviewer point that the Aim 1 prediction models (manuscript
Table 2) never account for the outcome's correlation with itself: a daily
symptom score is strongly self-predicting, so the relevant question is not
"does text predict the outcome" but "does text predict the outcome beyond the
outcome's own persistence".

PART A - Autocorrelation descriptives
    Lag-1 autocorrelation of each daily outcome (PM crisis, PM DSI-SS, AM
    crisis, AM DSI-SS), reported four ways:
      - pooled r over consecutive-day pairs (confounds between- and
        within-person variance; this is the number most papers report)
      - within-person r after person-mean centering BOTH y_t and y_{t-1}
        (the actual within-person persistence)
      - ICC(1) from a one-way random-effects decomposition (how much of the
        pooled autocorrelation is just stable between-person differences)
      - median / IQR of per-participant lag-1 r among participants with at
        least MIN_PAIRS_PER_PERSON consecutive-day pairs

PART B - AR(1) benchmark for each Table 2 analysis
    For each of the four Aim 1 analyses, three models are fit under the
    identical cross-validation scheme and compared on a COMMON subsample (rows
    with an observed lag-1 value, so the three models are strictly nested in
    sample as well as in predictors):
      1. lag_only        : AR(1) term alone
      2. text_only       : 20 embedding PCs alone
      3. text_plus_lag   : both
    The headline quantity is the incremental value of text over the AR(1)
    baseline (delta r, delta R-squared), with participant-cluster bootstrap
    confidence intervals computed on the pooled out-of-fold predictions.

    The text-only model is additionally reported on the FULL Table 2 sample so
    the subsample restriction is visible rather than silent.

Lag construction
    Every lag is an EXACT calendar join (``lag_date = ema_date - 1 day``).
    A bare ``groupby(...).shift(1)`` is never used: on this dataset it attaches
    roughly 15% of rows to an observation 2-10 days away (quantified in
    verify_text_only_aim1.py).

Modeling specification
    Identical to verify_text_only_aim1.py - sentence-transformer embeddings
    (all-mpnet-base-v2, 768-d, L2-normalized), PCA to 20 components fit on
    training folds only, StandardScaler fit on training folds only,
    Ridge(alpha=10.0), GroupKFold(n_splits=5) grouped by participant, metrics on
    pooled out-of-fold predictions. Non-text predictors are standardized with a
    scaler fit on training folds only and appended after the text PCs.

    Because the primary comparison is an incremental-R-squared contrast, a fixed
    ridge penalty could in principle handicap the low-dimensional AR model
    relative to the 20-PC text model. The script therefore also reports a
    sensitivity specification in which alpha is chosen per fold by an inner
    participant-grouped CV over ALPHA_GRID, separately for every model.

Requires the protected local data (data/processed, data/raw), which are not in
the public repository. Result tables are written to outputs/tables/, covered by
the repository's *.csv ignore rule; no participant text or participant-level
data are written.

Usage:
    python scripts/lagged_outcome_analyses.py
    python scripts/lagged_outcome_analyses.py --n-boot 5000
    python scripts/lagged_outcome_analyses.py --skip-tuned   # fixed alpha only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the verified loaders/specification rather than restating them, so this
# script cannot silently drift from the published Table 2 pipeline.
from verify_text_only_aim1 import (  # noqa: E402
    DATE_COL,
    OUT_DIR,
    PID_COL,
    PM_CRISIS_COL,
    PM_DSI_COL,
    RANDOM_SEED,
    build_am_day_outcomes,
    fisher_r_ci,
    load_and_verify_alignment,
)
from cv_blocks import (  # noqa: E402
    ALPHA_GRID,
    N_PCS,
    N_SPLITS,
    RIDGE_ALPHA,
    bootstrap_delta,
    run_grouped_cv_blocks,
)

MIN_PAIRS_PER_PERSON = 5
DEFAULT_N_BOOT = 2000

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Panel construction
# -----------------------------------------------------------------------------
def build_panel(feat: pd.DataFrame, am_day: pd.DataFrame) -> pd.DataFrame:
    """Person-day panel carrying every daily outcome plus the embedding row id.

    One row per analytic PM diary (the Table 2 sample), augmented with the AM
    outcomes recorded on the SAME day and on the NEXT day, and with the lag-1
    value of each outcome. All joins are exact calendar-date joins.
    """
    panel = feat[[PID_COL, DATE_COL, PM_CRISIS_COL, PM_DSI_COL]].copy()
    panel["embed_row"] = np.arange(len(panel))

    am = am_day.rename(columns={DATE_COL: "_d"})[
        [PID_COL, "_d", "crisis_AM_total", "dsi_AM_total"]
    ]

    # AM outcomes recorded on the same calendar day as the diary.
    same = am.rename(
        columns={"crisis_AM_total": "crisis_AM_t", "dsi_AM_total": "dsi_AM_t"}
    )
    panel = panel.merge(
        same, left_on=[PID_COL, DATE_COL], right_on=[PID_COL, "_d"], how="left"
    ).drop(columns="_d")

    # AM outcomes recorded on exactly the next calendar day (the Table 2
    # prospective outcomes).
    nxt = am.rename(
        columns={"crisis_AM_total": "crisis_AM_next", "dsi_AM_total": "dsi_AM_next"}
    )
    panel["_target"] = panel[DATE_COL] + pd.Timedelta(days=1)
    panel = panel.merge(
        nxt, left_on=[PID_COL, "_target"], right_on=[PID_COL, "_d"], how="left"
    ).drop(columns=["_d", "_target"])

    # Lag-1 PM outcomes: the same participant's diary-day record one calendar
    # day earlier. Restricted to the analytic sample on both sides.
    pm_prev = feat[[PID_COL, DATE_COL, PM_CRISIS_COL, PM_DSI_COL]].rename(
        columns={
            PM_CRISIS_COL: "crisis_PM_lag1",
            PM_DSI_COL: "dsi_PM_lag1",
            DATE_COL: "_d",
        }
    )
    panel["_prev"] = panel[DATE_COL] - pd.Timedelta(days=1)
    panel = panel.merge(
        pm_prev, left_on=[PID_COL, "_prev"], right_on=[PID_COL, "_d"], how="left"
    ).drop(columns=["_d", "_prev"])

    # Lag-1 AM outcomes relative to the NEXT-day AM outcome are simply the AM
    # outcomes on the diary day itself (crisis_AM_t / dsi_AM_t above).

    assert panel["embed_row"].is_unique, "Duplicate rows produced by the panel joins"
    assert len(panel) == len(feat), "Panel join changed the row count"
    return panel


# -----------------------------------------------------------------------------
# PART A: autocorrelation descriptives
# -----------------------------------------------------------------------------
def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) via the standard ANOVA estimator."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    grand = df["y"].mean()
    gm = df.groupby("g")["y"]
    n_i = gm.size().to_numpy(float)
    m_i = gm.mean().to_numpy(float)
    k = len(n_i)
    n_tot = n_i.sum()
    if k < 2 or n_tot <= k:
        return float("nan")

    ss_between = float((n_i * (m_i - grand) ** 2).sum())
    ss_within = float(((df["y"] - df["g"].map(gm.mean())) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_tot - k)

    # Harmonic-style effective group size for unbalanced designs.
    n0 = (n_tot - (n_i**2).sum() / n_tot) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    if denom <= 0:
        return float("nan")
    return float((ms_between - ms_within) / denom)


def lag1_descriptives(
    pairs: pd.DataFrame, y_col: str, lag_col: str, label: str
) -> dict:
    """Pooled, within-person, and per-participant lag-1 autocorrelation."""
    d = pairs.loc[pairs[y_col].notna() & pairs[lag_col].notna()].copy()
    y = d[y_col].to_numpy(float)
    yl = d[lag_col].to_numpy(float)
    g = d[PID_COL].astype(str).to_numpy()

    r_pooled = float(np.corrcoef(y, yl)[0, 1])
    ci_lo, ci_hi = fisher_r_ci(r_pooled, len(y))

    # Within-person: center BOTH series on the person's mean of that series.
    dd = pd.DataFrame({"y": y, "yl": yl, "g": g})
    dd["y_c"] = dd["y"] - dd.groupby("g")["y"].transform("mean")
    dd["yl_c"] = dd["yl"] - dd.groupby("g")["yl"].transform("mean")
    ok = dd["y_c"].std() > 0 and dd["yl_c"].std() > 0
    r_within = float(np.corrcoef(dd["y_c"], dd["yl_c"])[0, 1]) if ok else float("nan")

    # Per-participant correlations (only where both series actually vary).
    per: list[float] = []
    for _, grp in dd.groupby("g"):
        if len(grp) >= MIN_PAIRS_PER_PERSON and grp["y"].std() > 0 and grp["yl"].std() > 0:
            per.append(float(np.corrcoef(grp["y"], grp["yl"])[0, 1]))
    per_arr = np.asarray(per, dtype=float)

    return {
        "outcome": label,
        "n_pairs": int(len(d)),
        "n_participants": int(dd["g"].nunique()),
        "r_pooled": r_pooled,
        "r_pooled_CI_lo": ci_lo,
        "r_pooled_CI_hi": ci_hi,
        "r_within_person": r_within,
        "ICC1": icc1(y, g),
        "n_participants_with_min_pairs": int(len(per_arr)),
        "r_perperson_median": float(np.median(per_arr)) if len(per_arr) else float("nan"),
        "r_perperson_q25": float(np.percentile(per_arr, 25)) if len(per_arr) else float("nan"),
        "r_perperson_q75": float(np.percentile(per_arr, 75)) if len(per_arr) else float("nan"),
    }


def part_a(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag-1 autocorrelation for all four daily outcomes."""
    specs = [
        (PM_CRISIS_COL, "crisis_PM_lag1", "PM crisis total (t) ~ PM crisis total (t-1)"),
        (PM_DSI_COL, "dsi_PM_lag1", "PM DSI-SS total (t) ~ PM DSI-SS total (t-1)"),
        ("crisis_AM_next", "crisis_AM_t", "AM crisis total (t+1) ~ AM crisis total (t)"),
        ("dsi_AM_next", "dsi_AM_t", "AM DSI-SS total (t+1) ~ AM DSI-SS total (t)"),
    ]
    rows = [lag1_descriptives(panel, y, lag, label) for y, lag, label in specs]

    # Cross-measure persistence that a reader will also want: how well does the
    # evening report predict the next morning's report of the same construct?
    rows.append(
        lag1_descriptives(
            panel, "crisis_AM_next", PM_CRISIS_COL,
            "AM crisis total (t+1) ~ PM crisis total (t)  [evening -> next morning]",
        )
    )
    rows.append(
        lag1_descriptives(
            panel, "dsi_AM_next", PM_DSI_COL,
            "AM DSI-SS total (t+1) ~ PM DSI-SS total (t)  [evening -> next morning]",
        )
    )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# PART B: AR(1)-benchmarked prediction
# -----------------------------------------------------------------------------
# (analysis label, outcome column, AR(1) predictor column, description)
AR_ANALYSES = [
    (
        "crisis_concurrent",
        PM_CRISIS_COL,
        "crisis_PM_lag1",
        "PM text (t) -> PM crisis (t); AR term = PM crisis (t-1)",
    ),
    (
        "dsi_concurrent",
        PM_DSI_COL,
        "dsi_PM_lag1",
        "PM text (t) -> PM DSI-SS (t); AR term = PM DSI-SS (t-1)",
    ),
    (
        "crisis_prospective",
        "crisis_AM_next",
        "crisis_AM_t",
        "PM text (t) -> AM crisis (t+1); AR term = AM crisis (t)",
    ),
    (
        "dsi_prospective",
        "dsi_AM_next",
        "dsi_AM_t",
        "PM text (t) -> AM DSI-SS (t+1); AR term = AM DSI-SS (t)",
    ),
]


def part_b(
    panel: pd.DataFrame,
    emb: np.ndarray,
    n_boot: int,
    tune_alpha: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit lag_only / text_only / text_plus_lag on a common subsample."""
    model_rows: list[dict] = []
    contrast_rows: list[dict] = []

    alpha_specs = [("fixed_alpha_10", False)]
    if tune_alpha:
        alpha_specs.append(("inner_cv_alpha", True))

    for label, y_col, lag_col, desc in AR_ANALYSES:
        full = panel.loc[panel[y_col].notna()]
        sub = full.loc[full[lag_col].notna()]

        idx = sub["embed_row"].to_numpy()
        y = sub[y_col].to_numpy(float)
        g = sub[PID_COL].astype(str).to_numpy()
        z_lag = sub[[lag_col]].to_numpy(float)

        # Reference point: text-only on the FULL Table 2 sample, so the cost of
        # restricting to rows with an observed lag is explicit.
        full_res = run_grouped_cv_blocks(
            emb[full["embed_row"].to_numpy()],
            None,
            full[y_col].to_numpy(float),
            full[PID_COL].astype(str).to_numpy(),
            alpha=RIDGE_ALPHA,
        )

        for spec_name, tune in alpha_specs:
            alpha = None if tune else RIDGE_ALPHA
            fits = {
                "lag_only": run_grouped_cv_blocks(None, z_lag, y, g, alpha=alpha),
                "text_only": run_grouped_cv_blocks(emb[idx], None, y, g, alpha=alpha),
                "text_plus_lag": run_grouped_cv_blocks(emb[idx], z_lag, y, g, alpha=alpha),
            }

            for model_name, res in fits.items():
                model_rows.append(
                    {
                        "analysis": label,
                        "description": desc,
                        "alpha_spec": spec_name,
                        "model": model_name,
                        "sample": "lag-available subsample",
                        **{k: v for k, v in res.items() if k != "yhat"},
                    }
                )

            if spec_name == "fixed_alpha_10":
                model_rows.append(
                    {
                        "analysis": label,
                        "description": desc,
                        "alpha_spec": spec_name,
                        "model": "text_only_FULL_SAMPLE",
                        "sample": "full Table 2 sample (reference)",
                        **{k: v for k, v in full_res.items() if k != "yhat"},
                    }
                )

            # Incremental value of each block over the other.
            for contrast, a, b in [
                ("text_plus_lag_vs_lag_only", "text_plus_lag", "lag_only"),
                ("text_plus_lag_vs_text_only", "text_plus_lag", "text_only"),
                ("text_only_vs_lag_only", "text_only", "lag_only"),
            ]:
                boot = bootstrap_delta(
                    y, fits[a]["yhat"], fits[b]["yhat"], g, n_boot=n_boot
                )
                contrast_rows.append(
                    {
                        "analysis": label,
                        "alpha_spec": spec_name,
                        "contrast": contrast,
                        "n": int(len(y)),
                        "participants": int(len(np.unique(g))),
                        "r_a": fits[a]["pearson_r"],
                        "r_b": fits[b]["pearson_r"],
                        "R2_a": fits[a]["R2"],
                        "R2_b": fits[b]["R2"],
                        **boot,
                    }
                )

    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_part_a(tbl: pd.DataFrame) -> None:
    for _, r in tbl.iterrows():
        print(f"\n{r['outcome']}")
        print(f"  consecutive-day pairs        : {r['n_pairs']} "
              f"({r['n_participants']} participants)")
        print(f"  pooled lag-1 r               : {r['r_pooled']:.4f} "
              f"[{r['r_pooled_CI_lo']:.4f}, {r['r_pooled_CI_hi']:.4f}]")
        print(f"  within-person lag-1 r        : {r['r_within_person']:.4f}")
        print(f"  ICC(1) of the outcome        : {r['ICC1']:.4f}")
        print(f"  per-participant lag-1 r      : median {r['r_perperson_median']:.4f} "
              f"[IQR {r['r_perperson_q25']:.4f}, {r['r_perperson_q75']:.4f}] "
              f"over {int(r['n_participants_with_min_pairs'])} participants "
              f"with >= {MIN_PAIRS_PER_PERSON} pairs")


def print_part_b(models: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    for spec in models["alpha_spec"].unique():
        print("\n" + "-" * 78)
        print(f"ALPHA SPECIFICATION: {spec}")
        print("-" * 78)
        for label, _, _, desc in AR_ANALYSES:
            m = models.loc[(models["analysis"] == label) & (models["alpha_spec"] == spec)]
            if m.empty:
                continue
            print(f"\n{label}")
            print(f"  {desc}")
            for _, r in m.iterrows():
                extra = ""
                if isinstance(r.get("alphas_selected"), str) and r["alphas_selected"]:
                    extra = f"  alphas={r['alphas_selected']}"
                print(
                    f"    {r['model']:<24s} n={int(r['n']):>5d}  "
                    f"r={r['pearson_r']:.4f}  R2={r['R2']:>8.4f}  "
                    f"MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}{extra}"
                )
            c = contrasts.loc[
                (contrasts["analysis"] == label) & (contrasts["alpha_spec"] == spec)
            ]
            for _, r in c.iterrows():
                print(
                    f"      {r['contrast']:<30s} "
                    f"dr={r['delta_r']:+.4f} [{r['delta_r_CI_lo']:+.4f}, "
                    f"{r['delta_r_CI_hi']:+.4f}]   "
                    f"dR2={r['delta_R2']:+.4f} [{r['delta_R2_CI_lo']:+.4f}, "
                    f"{r['delta_R2_CI_hi']:+.4f}]"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="participant-cluster bootstrap replicates for contrast CIs")
    parser.add_argument("--skip-tuned", action="store_true",
                        help="report only the fixed alpha=10 specification")
    args = parser.parse_args()

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    feat, emb = load_and_verify_alignment()
    am_day = build_am_day_outcomes()
    panel = build_panel(feat, am_day)
    print(f"  Analytic PM diaries        : {len(panel)} "
          f"({panel[PID_COL].nunique()} participants)")
    print(f"  Embeddings                 : {emb.shape}, L2-normalized, row-aligned")
    print(f"  CV                         : GroupKFold({N_SPLITS}) by participant, "
          f"PCA({N_PCS}) + StandardScaler fit in-fold")
    print(f"  Ridge alpha                : {RIDGE_ALPHA} (fixed); "
          f"inner-CV grid {list(ALPHA_GRID)}")
    print(f"  Bootstrap replicates       : {args.n_boot} (participant clusters)")
    print("  All lags are exact calendar joins (date -/+ 1 day), never shift()")

    for col in ("crisis_PM_lag1", "dsi_PM_lag1", "crisis_AM_t", "dsi_AM_t",
                "crisis_AM_next", "dsi_AM_next"):
        print(f"    {col:<16s} available on {int(panel[col].notna().sum()):>5d} rows")

    print()
    print("=" * 78)
    print("PART A - LAG-1 AUTOCORRELATION DESCRIPTIVES")
    print("=" * 78)
    tbl_a = part_a(panel)
    print_part_a(tbl_a)

    print()
    print("=" * 78)
    print("PART B - AR(1) BENCHMARK FOR EACH TABLE 2 ANALYSIS")
    print("=" * 78)
    print("  All three models within an analysis use the SAME rows (those with an")
    print("  observed lag-1 value), so delta R-squared is directly interpretable.")
    models, contrasts = part_b(
        panel, emb, n_boot=args.n_boot, tune_alpha=not args.skip_tuned
    )
    print_part_b(models, contrasts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_a = OUT_DIR / "lag1_autocorrelation_descriptives.csv"
    p_m = OUT_DIR / "ar1_benchmark_models.csv"
    p_c = OUT_DIR / "ar1_benchmark_contrasts.csv"
    tbl_a.to_csv(p_a, index=False)
    models.drop(columns=[c for c in ("yhat",) if c in models.columns]).to_csv(p_m, index=False)
    contrasts.to_csv(p_c, index=False)
    print(f"\nSaved: {p_a}")
    print(f"Saved: {p_m}")
    print(f"Saved: {p_c}")


if __name__ == "__main__":
    main()
