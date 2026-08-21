"""Independent verification of the Aim 1 text-only prediction results.

This script re-derives, from scratch and with explicit assertions, the four
text-only Aim 1 results:

  1) crisis concurrent   : PM diary text (day t) -> PM crisis total (day t)
  2) crisis prospective  : PM diary text (day t) -> AM crisis total (day t+1)
  3) DSI-SS concurrent   : PM diary text (day t) -> PM DSI-SS total (day t)
  4) DSI-SS prospective  : PM diary text (day t) -> AM DSI-SS total (day t+1)

Both prospective analyses link the diary to the *exact* next calendar day
(``target_date = ema_date + 1 day``) joined on participant and date. A bare
``groupby(...).shift(-1)`` is deliberately NOT used anywhere: on this dataset it
attaches roughly 15% of diaries to a morning 2-10 days later (see the DIAGNOSTICS
section, which quantifies this).

Modeling specification, identical across all four analyses:
  - predictor : sentence-transformer embeddings of PM-window diary text
                (sentence-transformers/all-mpnet-base-v2, 768-d, L2-normalized),
                read from the cached data/processed/embeddings.npy
  - PCA to 20 components, fit on training folds only
  - StandardScaler, fit on training folds only
  - Ridge(alpha=10.0)
  - GroupKFold(n_splits=5) grouped by participant
  - metrics computed on pooled out-of-fold predictions

Text-only by construction: no questionnaire or numeric covariates enter any model.

Requires the protected local data (data/processed, data/raw), which are not in
the public repository. Result tables are written to outputs/tables/, which is
covered by the repository's *.csv ignore rule; no participant text or
participant-level data are written by this script.

Usage:
    python scripts/verify_text_only_aim1.py
    python scripts/verify_text_only_aim1.py --verify-embeddings   # re-encode a
        sample to confirm embeddings.npy provenance (downloads the ST model)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration: mirrors the project notebooks
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "pm_day_features.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "pm_day_clean.csv"
EMBED_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
ROWMAP_PATH = PROJECT_ROOT / "data" / "processed" / "embedding_rows.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ema_personality_plus_surveys_merged.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

PID_COL = "expiwell_id_clean"
DT_COL = "ema_dt"
DATE_COL = "ema_date"
TEXT_COL = "ema_text"

AM_CRISIS_TOTAL = "dailyAM__Total Score from 5 Questions"
PM_CRISIS_TOTAL = "dailyPM__Total Score from 5 Questions"
PM_CRISIS_COL = "crisis_PM_from_full"   # renamed PM_CRISIS_TOTAL in 01_data_cleaning
PM_DSI_COL = "dsi_PM_total"

PM_START_HOUR = 15          # 01_data_cleaning: PM/evening window
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EXPECTED_N = 2511
EXPECTED_PARTICIPANTS = 123
EMBED_DIM = 768

N_SPLITS = 5
N_PCS = 20
RIDGE_ALPHA = 10.0
RANDOM_SEED = 7

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Helpers reproduced from the project notebooks
# -----------------------------------------------------------------------------
def first_nonnull(x: pd.Series):
    """Day-level aggregator used throughout 01_data_cleaning / 04c / 04d."""
    x = x.dropna()
    return x.iloc[0] if len(x) else np.nan


def find_dsi_cols_by_prefix(df: pd.DataFrame, prefix: str) -> dict[str, str | None]:
    """Find DSI-SS items A-D by item stem. Verbatim from 01_data_cleaning."""
    cols = [c for c in df.columns if c.startswith(prefix)]

    def pick(patterns):
        pats = [re.compile(p, re.IGNORECASE) for p in patterns]
        hits = [c for c in cols if any(p.search(c) for p in pats)]
        hits = sorted(hits, key=len)
        return hits[0] if hits else None

    return {
        "A": pick([r"thoughts of killing myself", r"\bkilling myself\b"]),
        "B": pick([r"definite plan", r"formulated.*plan", r"considered possible ways", r"\bplans?\b"]),
        "C": pick([r"little or no control", r"control over", r"under my control", r"\bcontrol\b"]),
        "D": pick([r"impulses to kill myself", r"\bimpulses\b"]),
    }


def dsi_total(df: pd.DataFrame, prefix: str) -> pd.Series:
    """DSI-SS total = sum of items A-D, requiring all four present (min_count=4)."""
    items = find_dsi_cols_by_prefix(df, prefix)
    if any(v is None for v in items.values()):
        raise ValueError(f"Could not detect all 4 DSI items for prefix {prefix!r}: {items}")
    cols = [items[k] for k in ("A", "B", "C", "D")]
    if len(set(cols)) != 4:
        raise ValueError(f"DSI item detection for {prefix!r} returned duplicates: {cols}")
    return df[cols].sum(axis=1, min_count=4)


def fisher_r_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform CI for a Pearson correlation."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    crit = norm.ppf(1.0 - (1.0 - level) / 2.0)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


# -----------------------------------------------------------------------------
# Data loading and alignment verification
# -----------------------------------------------------------------------------
def load_and_verify_alignment() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the analytic file + cached embeddings, asserting exact row alignment."""
    for p in (FEATURES_PATH, EMBED_PATH, ROWMAP_PATH, RAW_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required local data file not found: {p}\n"
                "The protected data are intentionally absent from the public repository."
            )

    feat = pd.read_csv(FEATURES_PATH)
    emb = np.load(EMBED_PATH)
    rowmap = pd.read_csv(ROWMAP_PATH)
    feat[DATE_COL] = pd.to_datetime(feat[DATE_COL]).dt.normalize()

    assert len(feat) == EXPECTED_N, f"Expected {EXPECTED_N} rows, found {len(feat)}"
    assert feat[PID_COL].nunique() == EXPECTED_PARTICIPANTS, (
        f"Expected {EXPECTED_PARTICIPANTS} participants, found {feat[PID_COL].nunique()}"
    )
    assert emb.shape == (EXPECTED_N, EMBED_DIM), f"Unexpected embedding shape {emb.shape}"
    assert len(rowmap) == EXPECTED_N

    # Row-for-row key alignment between the row map, the analytic file, and embeddings.
    assert rowmap[PID_COL].astype(str).equals(feat[PID_COL].astype(str)), \
        "embedding_rows.csv participant order does not match pm_day_features.csv"
    assert pd.to_datetime(rowmap[DATE_COL]).dt.normalize().equals(feat[DATE_COL]), \
        "embedding_rows.csv date order does not match pm_day_features.csv"
    assert (rowmap["embed_row"].to_numpy() == np.arange(EXPECTED_N)).all(), \
        "embedding_rows.csv embed_row is not 0..N-1 in order"

    # embeddings.npy was generated from pm_day_clean.csv; confirm the two processed
    # files are the same rows in the same order before reusing the cache.
    if CLEAN_PATH.exists():
        clean = pd.read_csv(CLEAN_PATH)
        clean[DATE_COL] = pd.to_datetime(clean[DATE_COL]).dt.normalize()
        assert clean[PID_COL].astype(str).equals(feat[PID_COL].astype(str)), \
            "pm_day_clean.csv and pm_day_features.csv participant order differ"
        assert clean[DATE_COL].equals(feat[DATE_COL]), \
            "pm_day_clean.csv and pm_day_features.csv date order differ"

    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"Embeddings are not L2-normalized (norm range {norms.min():.6f}-{norms.max():.6f})"
    )
    return feat, emb


def verify_embedding_provenance(sample_idx: list[int] | None = None) -> None:
    """Re-encode a sample of pm_day_text and compare against the cached vectors."""
    from sentence_transformers import SentenceTransformer

    clean = pd.read_csv(CLEAN_PATH)
    emb = np.load(EMBED_PATH)
    idx = sample_idx or [0, 1, 2, 500, 1000, 1500, 2000, EXPECTED_N - 1]

    model = SentenceTransformer(EMBED_MODEL)
    re_emb = model.encode(
        clean["pm_day_text"].astype(str).iloc[idx].tolist(), normalize_embeddings=True
    )
    cos = np.sum(re_emb * emb[idx], axis=1)
    print(f"  re-encoded {len(idx)} sampled rows with {EMBED_MODEL}")
    print(f"  cosine similarity vs cached: min={cos.min():.6f} max={cos.max():.6f}")
    assert cos.min() > 0.999, "Cached embeddings do not match a re-encode of pm_day_text"
    print("  PASS: embeddings.npy is all-mpnet-base-v2 applied to pm_day_text")


# -----------------------------------------------------------------------------
# Day-level outcomes from raw data
# -----------------------------------------------------------------------------
def build_am_day_outcomes() -> pd.DataFrame:
    """Person-day AM crisis and AM DSI-SS totals from the raw merged file."""
    raw = pd.read_csv(RAW_PATH)
    raw[DT_COL] = pd.to_datetime(raw[DT_COL], errors="coerce")
    raw[DATE_COL] = raw[DT_COL].dt.normalize()

    raw["crisis_AM_total"] = raw[AM_CRISIS_TOTAL]
    raw["dsi_AM_total"] = dsi_total(raw, "dailyAM__")

    return (
        raw.sort_values([PID_COL, DT_COL])
        .groupby([PID_COL, DATE_COL], as_index=False)
        .agg(
            crisis_AM_total=("crisis_AM_total", first_nonnull),
            dsi_AM_total=("dsi_AM_total", first_nonnull),
        )
    )


def link_exact_next_day(feat: pd.DataFrame, am_day: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Join each PM diary to the AM outcome on EXACTLY the next calendar day.

    Rows whose t+1 date has no AM record are dropped rather than reaching forward
    to the participant's next available observation.
    """
    link = feat[[PID_COL, DATE_COL]].copy()
    link["row"] = np.arange(len(link))
    link["target_date"] = link[DATE_COL] + pd.Timedelta(days=1)

    am = am_day.loc[am_day[outcome].notna(), [PID_COL, DATE_COL, outcome]]
    am = am.rename(columns={DATE_COL: "target_date"})

    merged = (
        link.merge(am, on=[PID_COL, "target_date"], how="left")
        .sort_values("row")
        .reset_index(drop=True)
    )
    elig = merged.loc[merged[outcome].notna()].copy()

    gaps = (elig["target_date"] - elig[DATE_COL]).dt.days
    assert (gaps == 1).all(), "Prospective link is not exactly +1 calendar day"
    assert elig["row"].is_unique, "Duplicate PM rows produced by the next-day join"
    return elig


# -----------------------------------------------------------------------------
# Cross-validated evaluation
# -----------------------------------------------------------------------------
def run_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, label: str) -> dict:
    """Participant-grouped CV with fold-internal PCA + scaling; pooled OOF metrics."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    yhat = np.full(len(y), np.nan, dtype=float)
    fold_sizes: list[int] = []
    leak_counts: list[int] = []

    for tr, te in gkf.split(X, y, groups=groups):
        # Explicit leakage check: no participant may appear in both partitions.
        leak_counts.append(len(set(groups[tr]) & set(groups[te])))

        pca = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
        X_tr = pca.fit_transform(X[tr])          # fit on TRAIN only
        X_te = pca.transform(X[te])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)        # fit on TRAIN only
        X_te = scaler.transform(X_te)

        model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED).fit(X_tr, y[tr])
        yhat[te] = model.predict(X_te)
        fold_sizes.append(int(len(te)))

    assert not np.isnan(yhat).any(), "Some observations received no out-of-fold prediction"
    assert max(leak_counts) == 0, f"Participant leakage detected: {leak_counts}"

    r = float(np.corrcoef(y, yhat)[0, 1])
    ci_lo, ci_hi = fisher_r_ci(r, len(y))
    return {
        "analysis": label,
        "n": int(len(y)),
        "participants": int(len(np.unique(groups))),
        "pearson_r": r,
        "CI_lo": ci_lo,
        "CI_hi": ci_hi,
        "R2": float(r2_score(y, yhat)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "fold_test_sizes": fold_sizes,
        "max_participant_overlap": int(max(leak_counts)),
    }


# -----------------------------------------------------------------------------
# Diagnostics explaining divergence from earlier exploratory values
# -----------------------------------------------------------------------------
def print_diagnostics(feat: pd.DataFrame) -> None:
    """Quantify why earlier exploratory numbers differ. Data checks only; no models."""
    raw = pd.read_csv(RAW_PATH)
    raw[DT_COL] = pd.to_datetime(raw[DT_COL], errors="coerce")
    raw[DATE_COL] = raw[DT_COL].dt.normalize()
    raw["hour"] = raw[DT_COL].dt.hour

    day = (
        raw.sort_values([PID_COL, DT_COL])
        .groupby([PID_COL, DATE_COL], as_index=False)
        .agg(
            day_text=(TEXT_COL, lambda s: " ".join(str(t) for t in s.dropna())),
            crisis_AM=(AM_CRISIS_TOTAL, first_nonnull),
            crisis_PM=(PM_CRISIS_TOTAL, first_nonnull),
        )
    )
    day["day_text"] = day["day_text"].fillna("").astype(str)
    day = day.loc[day["day_text"].str.strip().ne("")].reset_index(drop=True)

    print(f"  Broader day-level table (text only, no DSI filter): {len(day)} rows, "
          f"{day[PID_COL].nunique()} participants")
    print(f"    with non-null PM crisis: {int(day['crisis_PM'].notna().sum())} "
          "(the ~2,730-row sample behind the older concurrent r)")
    print(f"    manuscript analytic sample additionally requires non-null dsi_PM_total: "
          f"{EXPECTED_N}")

    # Whole-day vs PM-only text over the analytic rows.
    if CLEAN_PATH.exists():
        clean = pd.read_csv(CLEAN_PATH)
        clean[DATE_COL] = pd.to_datetime(clean[DATE_COL]).dt.normalize()
        mg = clean[[PID_COL, DATE_COL, "pm_day_text"]].merge(
            day[[PID_COL, DATE_COL, "day_text"]], on=[PID_COL, DATE_COL], how="left"
        )
        same = mg["pm_day_text"].astype(str).str.strip() == mg["day_text"].astype(str).str.strip()
        print(f"  PM-only text identical to whole-day text: {int(same.sum())}/{len(mg)} "
              f"({same.mean() * 100:.1f}%) -- all diary entries fall in the hour>={PM_START_HOUR} window")

    # What a bare shift(-1) would have done.
    day = day.sort_values([PID_COL, DATE_COL]).reset_index(drop=True)
    day["next_date"] = day.groupby(PID_COL)[DATE_COL].shift(-1)
    day["crisis_AM_next"] = day.groupby(PID_COL)["crisis_AM"].shift(-1)
    day["gap_days"] = (day["next_date"] - day[DATE_COL]).dt.days

    matched = day.loc[day["crisis_AM_next"].notna()]
    mis_dated = int((matched["gap_days"] != 1).sum())
    print(f"  Bare groupby().shift(-1) would yield {len(matched)} next-AM rows, of which "
          f"{mis_dated} ({mis_dated / len(matched) * 100:.1f}%) are NOT exactly +1 calendar day")
    print("    gap distribution among shift(-1) matches (days -> count):")
    counts = matched["gap_days"].value_counts().sort_index().head(8)
    print("      " + ", ".join(f"{int(k)}d:{int(v)}" for k, v in counts.items()))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-embeddings",
        action="store_true",
        help="Re-encode sampled diary text to confirm embeddings.npy provenance "
             "(requires sentence-transformers and downloads the model).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    feat, emb = load_and_verify_alignment()
    print(f"  PM analytic rows          : {len(feat)}")
    print(f"  Participants              : {feat[PID_COL].nunique()}")
    print(f"  Embeddings                : {emb.shape}, L2-normalized, row-aligned (asserted)")
    print(f"  PM text variable          : pm_day_text (ema_text, hour >= {PM_START_HOUR}, per person-day)")
    print(f"  PM crisis variable        : {PM_CRISIS_COL}  (raw: {PM_CRISIS_TOTAL})")
    print(f"  PM DSI-SS variable        : {PM_DSI_COL}  (sum of items A-D, min_count=4)")
    print(f"  AM crisis variable        : {AM_CRISIS_TOTAL}")
    print("  AM DSI-SS variable        : sum of dailyAM__ items A-D, min_count=4")

    am_items = find_dsi_cols_by_prefix(pd.read_csv(RAW_PATH, nrows=5), "dailyAM__")
    for key in ("A", "B", "C", "D"):
        print(f"      item {key}: {str(am_items[key])[:66]}...")

    if args.verify_embeddings:
        print("\n  Embedding provenance check:")
        verify_embedding_provenance()

    # Independently reproduce dsi_PM_total from raw to confirm the stored column.
    raw_check = pd.read_csv(RAW_PATH)
    raw_check[DT_COL] = pd.to_datetime(raw_check[DT_COL], errors="coerce")
    raw_check[DATE_COL] = raw_check[DT_COL].dt.normalize()
    raw_check["dsi_PM_total_rebuilt"] = dsi_total(raw_check, "dailyPM__")
    rebuilt = (
        raw_check.sort_values([PID_COL, DT_COL])
        .groupby([PID_COL, DATE_COL], as_index=False)
        .agg(dsi_PM_total_rebuilt=("dsi_PM_total_rebuilt", first_nonnull))
    )
    chk = feat[[PID_COL, DATE_COL, PM_DSI_COL]].merge(rebuilt, on=[PID_COL, DATE_COL], how="left")
    assert np.allclose(chk[PM_DSI_COL], chk["dsi_PM_total_rebuilt"]), \
        "dsi_PM_total in pm_day_features.csv does not match a rebuild from raw items"
    print("\n  dsi_PM_total rebuilt from raw items matches stored column: PASS")

    am_day = build_am_day_outcomes()
    groups_all = feat[PID_COL].astype(str).to_numpy()

    results: list[dict] = []

    # --- Concurrent analyses (full analytic sample) --------------------------
    for outcome_col, label in [
        (PM_CRISIS_COL, "crisis_concurrent_PMtext_to_PMcrisis"),
        (PM_DSI_COL, "dsi_concurrent_PMtext_to_PMdsi"),
    ]:
        y = pd.to_numeric(feat[outcome_col], errors="coerce").to_numpy(float)
        assert not np.isnan(y).any(), f"Unexpected missing values in {outcome_col}"
        results.append(run_grouped_cv(emb, y, groups_all, label))

    # --- Prospective analyses (exact t+1 calendar day) -----------------------
    for outcome_col, label in [
        ("crisis_AM_total", "crisis_prospective_PMtext_to_nextAMcrisis"),
        ("dsi_AM_total", "dsi_prospective_PMtext_to_nextAMdsi"),
    ]:
        elig = link_exact_next_day(feat, am_day, outcome_col)
        idx = elig["row"].to_numpy()
        y = elig[outcome_col].to_numpy(float)
        g = elig[PID_COL].astype(str).to_numpy()
        res = run_grouped_cv(emb[idx], y, g, label)
        res["prospective_match"] = "exact ema_date + 1 day"
        results.append(res)

    print()
    print("=" * 78)
    print("RESULTS (pooled out-of-fold predictions)")
    print("=" * 78)
    for res in results:
        print(f"\n{res['analysis']}")
        print(f"  N                 : {res['n']}")
        print(f"  Participants      : {res['participants']}")
        print(f"  Pearson r         : {res['pearson_r']:.6f}")
        print(f"  Fisher-z 95% CI   : [{res['CI_lo']:.6f}, {res['CI_hi']:.6f}]")
        print(f"  R2                : {res['R2']:.6f}")
        print(f"  MAE               : {res['MAE']:.6f}")
        print(f"  RMSE              : {res['RMSE']:.6f}")
        print(f"  Fold test sizes   : {res['fold_test_sizes']}")
        print(f"  Participant overlap train/test: {res['max_participant_overlap']} (0 = no leakage)")

    print()
    print("=" * 78)
    print("DIAGNOSTICS (data checks explaining divergence from exploratory values)")
    print("=" * 78)
    print_diagnostics(feat)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(results)
    out["fold_test_sizes"] = out["fold_test_sizes"].astype(str)
    out_path = OUT_DIR / "verification_text_only_aim1.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
