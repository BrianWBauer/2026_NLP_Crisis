"""Covariate sensitivity analyses for Aim 1: day of week and demographics.

Addresses the reviewer request to adjust the text-prediction models for day of
week, age, and sex at birth. Existing scripts are not modified.

WHY THE TWO COVARIATE SETS ARE TREATED DIFFERENTLY
--------------------------------------------------
Day of week varies *within* participant. It can therefore confound a
within-person association and is tested directly as an additional predictor
block (Part A).

Age and sex are *time-invariant* over a two-week diary window. Any analysis that
person-mean-centers its predictors and outcome has already removed them, along
with every other stable participant characteristic, measured or not. Adding them
as covariates to a within-person estimate cannot change it. So the honest
question is not "do the within-person results survive adjustment" - they
survive by construction - but a different one:

    How much of the text models' accuracy is between-person rather than
    within-person, and can that between-person component be explained by
    demographics that sentence embeddings are known to encode?

This matters here specifically because H2 (within > between) was not supported,
so the between-person component is carrying real weight. Parts B-E answer it:

  Part B  Can text recover age and sex at all? If not, the confound has no
          mechanism and the rest is moot.
  Part C  Decompose each text model's out-of-fold predictions into
          between- and within-person components.
  Part D  Day-level models: text, demographics, and both.
  Part E  Person-level models (n = 123, the true between-person sample size):
          predict each participant's MEAN outcome from their mean embedding,
          from age and sex, and from both.
  Part F  Fold stability for Part E over random participant-to-fold partitions.

Part E is the one to read for the confound question, but quote it through Part
F. Note the sample size: a between-person claim rests on 123 observations, not
2511, however many person-days feed it, and with 123 participants in 5
deterministic GroupKFold folds a single partition's R-squared can sit a full SD
from the mean over partitions.

Usage:
    python scripts/covariate_sensitivity.py [--n-boot 2000] [--fold-repeats 30]
                                            [--skip-tuned]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv_blocks import (
    RANDOM_SEED,
    bootstrap_delta,
    repeated_cv,
    run_grouped_cv_blocks,
)
from verify_text_only_aim1 import (
    DATE_COL,
    OUT_DIR,
    PID_COL,
    PM_CRISIS_COL,
    PM_DSI_COL,
    PROJECT_ROOT,
    RAW_PATH,
    build_am_day_outcomes,
    load_and_verify_alignment,
)

BASELINE_PATH = PROJECT_ROOT / "data" / "all_data" / "ema_screener_baseline.csv"
BASELINE_ID_COL = "Baseline ID"
MAX_PLAUSIBLE_AGE = 80

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DOW_REFERENCE = 0  # Monday is the omitted category

# Manuscript-reported demographics, used as a provenance check on the join.
MANUSCRIPT_AGE_MEAN = 26.95
MANUSCRIPT_MALE_N = 75


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_demographics() -> pd.DataFrame:
    """One row per analytic participant with age and sex at birth.

    Bridges the analytic participant id to the baseline export:
        expiwell_id_clean -> baseline_id (raw merged file) -> Baseline ID
    Both hops are asserted to be 1:1 and complete for all 123 participants.
    """
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline export not found: {BASELINE_PATH}\n"
            "The protected data are intentionally absent from the public repository."
        )

    raw = pd.read_csv(RAW_PATH, low_memory=False)
    per_pid = raw.dropna(subset=["baseline_id"]).groupby(PID_COL)["baseline_id"]
    assert per_pid.nunique().max() == 1, "a participant maps to >1 baseline_id"
    bridge = per_pid.first().astype(str)

    base = pd.read_csv(BASELINE_PATH, low_memory=False)
    base[BASELINE_ID_COL] = base[BASELINE_ID_COL].astype(str)
    for col in ("age", "sex_at_birth"):
        nun = base.groupby(BASELINE_ID_COL)[col].nunique(dropna=True)
        assert nun.max() <= 1, f"{col} is not constant within participant"
    per_base = base.groupby(BASELINE_ID_COL).agg(
        age=("age", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        sex_at_birth=(
            "sex_at_birth",
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan,
        ),
    )

    # Provenance: the full baseline file must reproduce the reported demographics.
    all_age = pd.to_numeric(per_base["age"], errors="coerce")
    clean_age = all_age.where(all_age < MAX_PLAUSIBLE_AGE).dropna()
    assert abs(clean_age.mean() - MANUSCRIPT_AGE_MEAN) < 0.01, (
        f"baseline age mean {clean_age.mean():.2f} does not reproduce the "
        f"manuscript value {MANUSCRIPT_AGE_MEAN}"
    )
    assert (per_base["sex_at_birth"] == "Male").sum() == MANUSCRIPT_MALE_N, (
        "baseline sex counts do not reproduce the manuscript value"
    )

    demo = bridge.to_frame("baseline_id")
    demo = demo.join(per_base, on="baseline_id")
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")

    # Guard against the birth-year data-entry error present in the full export.
    bad = demo["age"] >= MAX_PLAUSIBLE_AGE
    if bad.any():
        raise AssertionError(
            f"implausible age in the analytic sample: "
            f"{demo.loc[bad, 'age'].to_dict()} - clean before proceeding"
        )

    demo["male"] = (demo["sex_at_birth"] == "Male").astype(float)
    assert demo["sex_at_birth"].isin(["Male", "Female"]).all(), (
        f"unexpected sex_at_birth values: {demo['sex_at_birth'].unique()}"
    )
    assert demo["age"].notna().all(), "missing age in the analytic sample"
    demo.index = demo.index.astype(str)
    return demo[["age", "male", "sex_at_birth"]]


def build_panel(feat: pd.DataFrame, am_day: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    """One row per analytic PM diary, with same-day, next-day, calendar and
    person-level columns attached. All temporal links are exact calendar joins.
    """
    panel = feat[[PID_COL, DATE_COL, PM_CRISIS_COL, PM_DSI_COL]].copy()
    panel["row"] = np.arange(len(panel))
    panel[PID_COL] = panel[PID_COL].astype(str)

    am = am_day.copy()
    am[PID_COL] = am[PID_COL].astype(str)
    nxt = am.rename(
        columns={
            DATE_COL: "_join_date",
            "crisis_AM_total": "crisis_AM_next",
            "dsi_AM_total": "dsi_AM_next",
        }
    )
    panel["_join_date"] = panel[DATE_COL] + pd.Timedelta(days=1)
    panel = panel.merge(
        nxt[[PID_COL, "_join_date", "crisis_AM_next", "dsi_AM_next"]],
        on=[PID_COL, "_join_date"],
        how="left",
    )
    gaps = (panel["_join_date"] - panel[DATE_COL]).dt.days
    assert (gaps == 1).all(), "next-day link is not exactly +1 calendar day"
    panel = panel.drop(columns="_join_date")

    # Calendar covariate (varies within person).
    panel["dow"] = panel[DATE_COL].dt.dayofweek
    panel["is_weekend"] = panel["dow"].isin([5, 6]).astype(float)

    # Person-level covariates (constant within person).
    panel = panel.join(demo, on=PID_COL)
    assert panel["age"].notna().all() and panel["male"].notna().all(), (
        "demographics did not attach to every analytic row"
    )

    assert panel["row"].is_unique and len(panel) == len(feat), "panel row count changed"
    return panel.sort_values("row").reset_index(drop=True)


def dow_dummies(dow: np.ndarray) -> np.ndarray:
    """Six indicator columns, Monday omitted."""
    cols = [d for d in range(7) if d != DOW_REFERENCE]
    return np.column_stack([(dow == d).astype(float) for d in cols])


# ---------------------------------------------------------------------------
# Outcome specifications (the four Table 2 analyses)
# ---------------------------------------------------------------------------
ANALYSES = [
    ("crisis_concurrent", PM_CRISIS_COL, "PM crisis total, same evening"),
    ("dsi_concurrent", PM_DSI_COL, "PM DSI-SS total, same evening"),
    ("crisis_prospective", "crisis_AM_next", "AM crisis total, next morning"),
    ("dsi_prospective", "dsi_AM_next", "AM DSI-SS total, next morning"),
]


def _between_within(y: np.ndarray, groups: np.ndarray) -> dict:
    """Variance share of y that is between-person, plus ICC-style n0 correction."""
    df = pd.DataFrame({"y": y, "g": groups})
    gm = df.groupby("g")["y"].transform("mean")
    between = gm.to_numpy()
    within = y - between
    tot = np.var(y, ddof=1)
    return {
        "var_between_share": float(np.var(between, ddof=1) / tot) if tot > 0 else np.nan,
        "var_within_share": float(np.var(within, ddof=1) / tot) if tot > 0 else np.nan,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Part A - day of week
# ---------------------------------------------------------------------------
def part_a(panel: pd.DataFrame, emb: np.ndarray, n_boot: int, alpha) -> tuple:
    print("\n" + "=" * 78)
    print("PART A - day of week (a WITHIN-person covariate)")
    print("=" * 78)

    rows, contrasts = [], []
    for label, ycol, desc in ANALYSES:
        mask = panel[ycol].notna().to_numpy()
        sub = panel.loc[mask]
        y = sub[ycol].to_numpy(float)
        g = sub[PID_COL].to_numpy()
        X = emb[sub["row"].to_numpy()]
        D = dow_dummies(sub["dow"].to_numpy())

        print(f"\n{label}  ({desc})   n={mask.sum()}  participants={len(np.unique(g))}")
        fits = {}
        for name, Xe, Z in [
            ("dow_only", None, D),
            ("text_only", X, None),
            ("text_plus_dow", X, D),
        ]:
            r = run_grouped_cv_blocks(Xe, Z, y, g, alpha=alpha, label=f"{label}:{name}")
            fits[name] = r
            rows.append({"analysis": label, "model": name, **{
                k: v for k, v in r.items() if k not in ("yhat", "analysis")}})
            print(f"   {name:16s} r={r['pearson_r']:+.4f}  R2={r['R2']:+.4f}  "
                  f"p={r['n_predictors']}")

        for a, b, tag in [
            ("text_only", "text_plus_dow", "adding dow to text"),
            ("dow_only", "text_plus_dow", "adding text to dow"),
        ]:
            d = bootstrap_delta(y, fits[b]["yhat"], fits[a]["yhat"], g, n_boot=n_boot)
            contrasts.append({"analysis": label, "contrast": tag,
                              "model_a": b, "model_b": a, **d})
            print(f"   D R2 {tag:22s} {d['delta_R2']:+.4f} "
                  f"[{d['delta_R2_CI_lo']:+.4f}, {d['delta_R2_CI_hi']:+.4f}]")

    # Descriptive: does the outcome actually vary by weekday?
    print("\n  Outcome means by weekday (descriptive):")
    for label, ycol, _ in ANALYSES:
        m = panel.groupby("dow")[ycol].mean()
        rng = m.max() - m.min()
        sd = panel[ycol].std()
        print(f"   {label:20s} range across days = {rng:.3f} "
              f"({rng / sd:.3f} SD)  " +
              " ".join(f"{DAY_NAMES[d]}={m.get(d, np.nan):.2f}" for d in range(7)))
    return rows, contrasts


# ---------------------------------------------------------------------------
# Part B - can text recover demographics?
# ---------------------------------------------------------------------------
def part_b(panel: pd.DataFrame, emb: np.ndarray, alpha) -> list:
    print("\n" + "=" * 78)
    print("PART B - can the embeddings recover age and sex?")
    print("   If text cannot recover demographics, the confound has no mechanism.")
    print("=" * 78)

    rows = []
    g = panel[PID_COL].to_numpy()
    X = emb[panel["row"].to_numpy()]

    for target in ("age", "male"):
        y = panel[target].to_numpy(float)
        r = run_grouped_cv_blocks(X, None, y, g, alpha=alpha, label=f"day_level:{target}")
        extra = {}
        if target == "male":
            extra["AUROC"] = float(roc_auc_score(y, r["yhat"]))
        print(f"  day-level  {target:4s} n={r['n']:5d}  r={r['pearson_r']:+.4f}  "
              f"R2={r['R2']:+.4f}" +
              (f"  AUROC={extra['AUROC']:.4f}" if extra else ""))
        rows.append({"level": "day", "target": target,
                     **{k: v for k, v in r.items() if k != "yhat"}, **extra})

    # Person level: mean embedding per participant.
    pX, pids = person_mean_embeddings(panel, emb)
    demo = panel.groupby(PID_COL)[["age", "male"]].first().loc[pids]
    for target in ("age", "male"):
        y = demo[target].to_numpy(float)
        r = run_grouped_cv_blocks(pX, None, y, np.asarray(pids), alpha=alpha,
                                  label=f"person_level:{target}")
        extra = {}
        if target == "male":
            extra["AUROC"] = float(roc_auc_score(y, r["yhat"]))
        print(f"  person-lvl {target:4s} n={r['n']:5d}  r={r['pearson_r']:+.4f}  "
              f"R2={r['R2']:+.4f}" +
              (f"  AUROC={extra['AUROC']:.4f}" if extra else ""))
        rows.append({"level": "person", "target": target,
                     **{k: v for k, v in r.items() if k != "yhat"}, **extra})
    return rows


def person_mean_embeddings(panel: pd.DataFrame, emb: np.ndarray):
    """Mean embedding per participant, L2-renormalized, in a stable id order."""
    pids = sorted(panel[PID_COL].unique())
    mats = []
    for p in pids:
        idx = panel.loc[panel[PID_COL] == p, "row"].to_numpy()
        v = emb[idx].mean(axis=0)
        n = np.linalg.norm(v)
        mats.append(v / n if n > 0 else v)
    return np.vstack(mats), pids


# ---------------------------------------------------------------------------
# Part C - between/within decomposition of the text models' predictions
# ---------------------------------------------------------------------------
def _decompose(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray) -> dict:
    """Split the agreement between y and yhat into between- and within-person parts."""
    d = pd.DataFrame({"y": y, "p": yhat, "g": groups})
    ym = d.groupby("g")["y"].transform("mean").to_numpy()
    pm = d.groupby("g")["p"].transform("mean").to_numpy()
    per = d.groupby("g").agg(y=("y", "mean"), p=("p", "mean"))
    return {
        "r_between": _corr(per["p"].to_numpy(), per["y"].to_numpy()),
        "r_within": _corr(yhat - pm, y - ym),
        "participants": len(per),
    }


def part_c(panel: pd.DataFrame, emb: np.ndarray, alpha, n_repeats: int) -> list:
    """Between/within decomposition, averaged over random participant partitions.

    The deterministic GroupKFold partition is reported too, but the mean and SD
    over n_repeats partitions are the values to quote: the single-partition
    between-person correlations move by several hundredths across partitions,
    whereas the within-person correlations are stable to about .01. That
    contrast is itself the finding, so it has to be measured rather than
    assumed from one split.
    """
    print("\n" + "=" * 78)
    print("PART C - how much of the text models' accuracy is BETWEEN-person?")
    print(f"   deterministic partition plus {n_repeats} random partitions")
    print("=" * 78)
    rows = []
    for label, ycol, _ in ANALYSES:
        sub = panel.loc[panel[ycol].notna()]
        y = sub[ycol].to_numpy(float)
        g = sub[PID_COL].to_numpy()
        X = emb[sub["row"].to_numpy()]

        fit = run_grouped_cv_blocks(X, None, y, g, alpha=alpha, label=f"{label}:text_only")
        det = _decompose(y, fit["yhat"], g)
        shares_y = _between_within(y, g)
        shares_p = _between_within(fit["yhat"], g)

        btw, wth, pld = [], [], []
        for i in range(n_repeats):
            f = run_grouped_cv_blocks(X, None, y, g, alpha=alpha,
                                      group_seed=RANDOM_SEED + i)
            dec = _decompose(y, f["yhat"], g)
            btw.append(dec["r_between"])
            wth.append(dec["r_within"])
            pld.append(f["pearson_r"])
        btw, wth, pld = np.array(btw), np.array(wth), np.array(pld)

        print(f"\n  {label}  n={len(y)}  participants={det['participants']}")
        print(f"    pooled   r  det={fit['pearson_r']:+.4f}   "
              f"mean={pld.mean():+.4f} sd={pld.std(ddof=1):.4f}")
        print(f"    between  r  det={det['r_between']:+.4f}   "
              f"mean={btw.mean():+.4f} sd={btw.std(ddof=1):.4f}")
        print(f"    within   r  det={det['r_within']:+.4f}   "
              f"mean={wth.mean():+.4f} sd={wth.std(ddof=1):.4f}")
        print(f"    outcome variance between-person    {shares_y['var_between_share']:.3f}")
        print(f"    PREDICTION variance between-person {shares_p['var_between_share']:.3f}")

        rows.append({
            "analysis": label, "n": len(y), "participants": det["participants"],
            "n_repeats": n_repeats,
            "r_pooled": fit["pearson_r"],
            "r_between": det["r_between"],
            "r_within": det["r_within"],
            "r_pooled_mean": pld.mean(), "r_pooled_sd": pld.std(ddof=1),
            "r_between_mean": btw.mean(), "r_between_sd": btw.std(ddof=1),
            "r_between_min": btw.min(), "r_between_max": btw.max(),
            "r_within_mean": wth.mean(), "r_within_sd": wth.std(ddof=1),
            "r_within_min": wth.min(), "r_within_max": wth.max(),
            "outcome_var_between_share": shares_y["var_between_share"],
            "outcome_var_within_share": shares_y["var_within_share"],
            "prediction_var_between_share": shares_p["var_between_share"],
            "prediction_var_within_share": shares_p["var_within_share"],
        })
    return rows


# ---------------------------------------------------------------------------
# Part D - day-level demographic adjustment
# ---------------------------------------------------------------------------
def part_d(panel: pd.DataFrame, emb: np.ndarray, n_boot: int, alpha) -> tuple:
    print("\n" + "=" * 78)
    print("PART D - day-level models with age and sex")
    print("   NOTE: these are person-level covariates under participant-grouped CV,")
    print("   so they can only act through between-person variance.")
    print("=" * 78)
    rows, contrasts = [], []
    for label, ycol, _ in ANALYSES:
        sub = panel.loc[panel[ycol].notna()]
        y = sub[ycol].to_numpy(float)
        g = sub[PID_COL].to_numpy()
        X = emb[sub["row"].to_numpy()]
        Zd = sub[["age", "male"]].to_numpy(float)

        print(f"\n{label}  n={len(y)}")
        fits = {}
        for name, Xe, Z in [
            ("demo_only", None, Zd),
            ("text_only", X, None),
            ("text_plus_demo", X, Zd),
        ]:
            r = run_grouped_cv_blocks(Xe, Z, y, g, alpha=alpha, label=f"{label}:{name}")
            fits[name] = r
            rows.append({"analysis": label, "model": name,
                         **{k: v for k, v in r.items() if k not in ("yhat", "analysis")}})
            print(f"   {name:16s} r={r['pearson_r']:+.4f}  R2={r['R2']:+.4f}")

        for a, b, tag in [
            ("text_only", "text_plus_demo", "adding demo to text"),
            ("demo_only", "text_plus_demo", "adding text to demo"),
        ]:
            d = bootstrap_delta(y, fits[b]["yhat"], fits[a]["yhat"], g, n_boot=n_boot)
            contrasts.append({"analysis": label, "contrast": tag,
                              "model_a": b, "model_b": a, **d})
            print(f"   D R2 {tag:22s} {d['delta_R2']:+.4f} "
                  f"[{d['delta_R2_CI_lo']:+.4f}, {d['delta_R2_CI_hi']:+.4f}]")
    return rows, contrasts


# ---------------------------------------------------------------------------
# Part E - person-level between-person analysis (the decisive one)
# ---------------------------------------------------------------------------
def part_e(panel: pd.DataFrame, emb: np.ndarray, n_boot: int, alpha) -> tuple:
    print("\n" + "=" * 78)
    print("PART E - BETWEEN-person analysis at the person level (n = 123)")
    print("   Outcome is each participant's mean score. This is the real sample")
    print("   size behind any between-person claim.")
    print("=" * 78)
    pX, pids = person_mean_embeddings(panel, emb)
    pidx = {p: i for i, p in enumerate(pids)}
    rows, contrasts = [], []

    for label, ycol, _ in ANALYSES:
        sub = panel.loc[panel[ycol].notna()]
        per = sub.groupby(PID_COL).agg(
            y=(ycol, "mean"), age=("age", "first"), male=("male", "first"),
            n_days=(ycol, "size"))
        per = per.loc[[p for p in pids if p in per.index]]
        keep = np.array([pidx[p] for p in per.index])
        y = per["y"].to_numpy(float)
        Zd = per[["age", "male"]].to_numpy(float)
        X = pX[keep]
        g = np.asarray(per.index)  # one participant per row

        print(f"\n{label}  participants={len(y)}  "
              f"median days/ppt={per['n_days'].median():.0f}")
        fits = {}
        for name, Xe, Z in [
            ("demo_only", None, Zd),
            ("text_only", X, None),
            ("text_plus_demo", X, Zd),
        ]:
            r = run_grouped_cv_blocks(Xe, Z, y, g, alpha=alpha, label=f"{label}:{name}")
            fits[name] = r
            rows.append({"analysis": label, "model": name,
                         **{k: v for k, v in r.items() if k not in ("yhat", "analysis")}})
            print(f"   {name:16s} r={r['pearson_r']:+.4f}  R2={r['R2']:+.4f}")

        for a, b, tag in [
            ("text_only", "text_plus_demo", "adding demo to text"),
            ("demo_only", "text_plus_demo", "adding text to demo"),
        ]:
            d = bootstrap_delta(y, fits[b]["yhat"], fits[a]["yhat"], g, n_boot=n_boot)
            contrasts.append({"analysis": label, "contrast": tag,
                              "model_a": b, "model_b": a, **d})
            print(f"   D R2 {tag:22s} {d['delta_R2']:+.4f} "
                  f"[{d['delta_R2_CI_lo']:+.4f}, {d['delta_R2_CI_hi']:+.4f}]")
    return rows, contrasts


# ---------------------------------------------------------------------------
# Part F - fold stability for the person-level models
# ---------------------------------------------------------------------------
def part_f(panel: pd.DataFrame, emb: np.ndarray, n_repeats: int, alpha) -> list:
    """Repeated CV over random participant-to-fold partitions, person level.

    Part E runs 123 participants through 5 deterministic GroupKFold folds, which
    is precisely the regime where fold assignment moves the metric: a single
    partition's R-squared can sit a full SD away from the mean over partitions.
    Quote these means and SDs, not Part E's single-partition point estimates.
    """
    print("\n" + "=" * 78)
    print(f"PART F - person-level fold stability ({n_repeats} random partitions)")
    print("=" * 78)
    pX, pids = person_mean_embeddings(panel, emb)
    pidx = {p: i for i, p in enumerate(pids)}
    rows = []
    print("%-20s %-15s %8s %7s %8s %7s" %
          ("analysis", "model", "r_mean", "r_sd", "R2_mean", "R2_sd"))
    for label, ycol, _ in ANALYSES:
        sub = panel.loc[panel[ycol].notna()]
        per = sub.groupby(PID_COL).agg(
            y=(ycol, "mean"), age=("age", "first"), male=("male", "first"))
        per = per.loc[[p for p in pids if p in per.index]]
        keep = np.array([pidx[p] for p in per.index])
        y = per["y"].to_numpy(float)
        Zd = per[["age", "male"]].to_numpy(float)
        X = pX[keep]
        g = np.asarray(per.index)
        for name, Xe, Z in [
            ("demo_only", None, Zd),
            ("text_only", X, None),
            ("text_plus_demo", X, Zd),
        ]:
            s = repeated_cv(Xe, Z, y, g, n_repeats=n_repeats, alpha=alpha)
            rows.append({"analysis": label, "model": name,
                         "participants": len(y), **s})
            print("%-20s %-15s %+8.4f %7.4f %+8.4f %7.4f" % (
                label, name, s["r_mean"], s["r_sd"], s["R2_mean"], s["R2_sd"]))
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fold-repeats", type=int, default=30,
                    help="random participant-to-fold partitions for Parts C and F")
    ap.add_argument("--skip-tuned", action="store_true",
                    help="skip the inner-CV ridge tuning pass")
    args = ap.parse_args()

    feat, emb = load_and_verify_alignment()
    am_day = build_am_day_outcomes()
    demo = load_demographics()
    print(f"demographics attached for {len(demo)} participants")
    print(f"  age  mean={demo['age'].mean():.2f} sd={demo['age'].std():.2f} "
          f"min={demo['age'].min():g} max={demo['age'].max():g}")
    print(f"  sex at birth: " +
          ", ".join(f"{k}={v}" for k, v in demo['sex_at_birth'].value_counts().items()))

    panel = build_panel(feat, am_day, demo)
    print(f"panel rows={len(panel)} participants={panel[PID_COL].nunique()}")
    print("  weekday counts: " +
          " ".join(f"{DAY_NAMES[d]}={int((panel['dow'] == d).sum())}" for d in range(7)))
    spans = panel.groupby(PID_COL)["dow"].nunique()
    print(f"  participants spanning >=5 distinct weekdays: "
          f"{int((spans >= 5).sum())}/{len(spans)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha = 10.0

    a_rows, a_con = part_a(panel, emb, args.n_boot, alpha)
    b_rows = part_b(panel, emb, alpha)
    c_rows = part_c(panel, emb, alpha, args.fold_repeats)
    d_rows, d_con = part_d(panel, emb, args.n_boot, alpha)
    e_rows, e_con = part_e(panel, emb, args.n_boot, alpha)
    f_rows = part_f(panel, emb, args.fold_repeats, alpha)

    outputs = {
        "covariate_personlevel_fold_stability.csv": f_rows,
        "covariate_dayofweek_models.csv": a_rows,
        "covariate_dayofweek_contrasts.csv": a_con,
        "covariate_text_recovers_demographics.csv": b_rows,
        "covariate_between_within_decomposition.csv": c_rows,
        "covariate_demographic_models.csv": d_rows,
        "covariate_demographic_contrasts.csv": d_con,
        "covariate_personlevel_models.csv": e_rows,
        "covariate_personlevel_contrasts.csv": e_con,
    }

    if not args.skip_tuned:
        print("\n" + "=" * 78)
        print("TUNED-ALPHA REPLICATION (inner participant-grouped CV per fold)")
        print("=" * 78)
        t_d_rows, t_d_con = part_d(panel, emb, args.n_boot, None)
        t_e_rows, t_e_con = part_e(panel, emb, args.n_boot, None)
        t_a_rows, t_a_con = part_a(panel, emb, args.n_boot, None)
        outputs["covariate_demographic_models_tuned.csv"] = t_d_rows
        outputs["covariate_demographic_contrasts_tuned.csv"] = t_d_con
        outputs["covariate_personlevel_models_tuned.csv"] = t_e_rows
        outputs["covariate_personlevel_contrasts_tuned.csv"] = t_e_con
        outputs["covariate_dayofweek_models_tuned.csv"] = t_a_rows
        outputs["covariate_dayofweek_contrasts_tuned.csv"] = t_a_con

    print("\n" + "=" * 78)
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(OUT_DIR / name, index=False)
        print(f"wrote {OUT_DIR / name}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
