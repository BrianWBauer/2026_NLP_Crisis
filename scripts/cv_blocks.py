"""Shared cross-validation machinery for block-structured predictor sets.

Extends the Table 2 specification in verify_text_only_aim1.py to models that
combine the text block (sentence-transformer embeddings reduced by in-fold PCA)
with blocks of numeric predictors (lagged outcomes, questionnaire scores,
calendar and person-level covariates).

The specification is deliberately identical to the published Table 2 pipeline
wherever it overlaps:
  - PCA to N_PCS components, fit on training folds only
  - StandardScaler on the PCs, fit on training folds only
  - Ridge(alpha=RIDGE_ALPHA)
  - GroupKFold(n_splits=N_SPLITS) grouped by participant
  - metrics computed on pooled out-of-fold predictions

Numeric blocks are standardized by their own StandardScaler, also fit on
training folds only, and concatenated AFTER the text PCs. Constant-in-fold
columns are passed through as zeros rather than producing NaNs.

``run_grouped_cv_blocks`` accepts ``alpha=None`` to select the ridge penalty per
outer fold by an inner participant-grouped CV over ALPHA_GRID. This matters
whenever the reported quantity is an incremental-R-squared contrast between a
low-dimensional block (e.g. a single AR term) and the 20-dimensional text
block, because a single fixed penalty is not equally appropriate for both.

``bootstrap_delta`` compares two sets of pooled out-of-fold predictions by
resampling PARTICIPANTS with replacement. It is inference on the realized
out-of-fold predictions, not on the full fit-and-predict procedure; it is
therefore a statement about the predictions in hand and does not propagate
uncertainty in the fold assignment or the PCA basis.

``repeated_cv`` covers that remaining source of uncertainty. ``GroupKFold`` is
deterministic - it assigns participants to folds greedily by size - so adding or
removing a single row can reshuffle the entire fold structure and move the
reported r by a non-trivial amount. ``repeated_cv`` re-runs the whole pipeline
over independent random partitions of participants into folds and returns the
mean and SD of each metric, which is what should be quoted when comparing
numbers computed on slightly different samples.

This module is imported by other scripts and is not a command-line entry point.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
N_PCS = 20
RIDGE_ALPHA = 10.0
RANDOM_SEED = 7
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
INNER_SPLITS = 4


def _safe_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training statistics; zero out columns constant in train."""
    scaler = StandardScaler()
    tr = scaler.fit_transform(train)
    te = scaler.transform(test)
    dead = scaler.scale_ == 0
    if dead.any():
        tr[:, dead] = 0.0
        te[:, dead] = 0.0
    return tr, te


def _build_design(
    X_emb: np.ndarray | None,
    Z: np.ndarray | None,
    tr: np.ndarray,
    te: np.ndarray,
    n_pcs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble train/test design matrices for one fold.

    PCA and both scalers are fit on the training rows only.
    """
    parts_tr: list[np.ndarray] = []
    parts_te: list[np.ndarray] = []

    if X_emb is not None:
        k = min(n_pcs, len(tr), X_emb.shape[1])
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        a = pca.fit_transform(X_emb[tr])
        b = pca.transform(X_emb[te])
        a, b = _safe_scale(a, b)
        parts_tr.append(a)
        parts_te.append(b)

    if Z is not None:
        a, b = _safe_scale(Z[tr].astype(float), Z[te].astype(float))
        parts_tr.append(a)
        parts_te.append(b)

    if not parts_tr:
        raise ValueError("At least one of X_emb or Z must be provided")

    return np.hstack(parts_tr), np.hstack(parts_te)


def _select_alpha(
    X_emb: np.ndarray | None,
    Z: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    n_pcs: int,
) -> float:
    """Choose alpha by participant-grouped CV *within* the outer training rows."""
    g_tr = groups[tr]
    n_groups = len(np.unique(g_tr))
    if n_groups < 2:
        return RIDGE_ALPHA
    inner = GroupKFold(n_splits=min(INNER_SPLITS, n_groups))

    scores = np.zeros(len(ALPHA_GRID), dtype=float)
    for itr, ite in inner.split(tr, y[tr], groups=g_tr):
        # Indices into the ORIGINAL arrays, so _build_design can be reused.
        a_idx, b_idx = tr[itr], tr[ite]
        Xa, Xb = _build_design(X_emb, Z, a_idx, b_idx, n_pcs)
        for j, alpha in enumerate(ALPHA_GRID):
            model = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(Xa, y[a_idx])
            pred = model.predict(Xb)
            scores[j] += float(np.sum((y[b_idx] - pred) ** 2))
    return float(ALPHA_GRID[int(np.argmin(scores))])


def _random_group_folds(
    groups: np.ndarray, n_splits: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random participant-disjoint folds, balanced on participant count.

    An alternative to GroupKFold's deterministic size-greedy assignment, used by
    ``repeated_cv`` to quantify how much a reported metric depends on which
    participants happen to land in which fold.
    """
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    assign = {g: i % n_splits for i, g in enumerate(uniq)}
    fold_of_row = np.array([assign[g] for g in groups])

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        te = np.flatnonzero(fold_of_row == k)
        tr = np.flatnonzero(fold_of_row != k)
        if len(te) and len(tr):
            folds.append((tr, te))
    return folds


def run_grouped_cv_blocks(
    X_emb: np.ndarray | None,
    Z: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float | None = RIDGE_ALPHA,
    n_pcs: int = N_PCS,
    n_splits: int = N_SPLITS,
    label: str | None = None,
    group_seed: int | None = None,
) -> dict:
    """Participant-grouped CV over a text block and/or a numeric block.

    Parameters
    ----------
    X_emb : (n, d) embeddings, or None to omit the text block.
    Z     : (n, p) numeric predictors, or None to omit them.
    y     : (n,) outcome.
    groups: (n,) participant identifiers used for grouping.
    alpha : fixed ridge penalty, or None to select per fold by inner grouped CV.
    group_seed : None to use GroupKFold exactly as in the published Table 2
        pipeline; an integer to draw a random participant-to-fold assignment
        instead (used by ``repeated_cv``).

    Returns a dict of pooled out-of-fold metrics plus ``yhat`` (the pooled
    out-of-fold predictions), which callers use for bootstrap contrasts.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n = len(y)
    assert not np.isnan(y).any(), "Outcome contains missing values"
    if X_emb is not None:
        assert len(X_emb) == n, "X_emb and y lengths differ"
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        assert len(Z) == n, "Z and y lengths differ"
        assert not np.isnan(Z).any(), "Numeric predictor block contains missing values"

    if group_seed is None:
        folds = list(GroupKFold(n_splits=n_splits).split(np.zeros((n, 1)), y, groups=groups))
    else:
        folds = _random_group_folds(groups, n_splits, np.random.default_rng(group_seed))

    yhat = np.full(n, np.nan, dtype=float)
    fold_sizes: list[int] = []
    leak_counts: list[int] = []
    alphas_used: list[float] = []

    for tr, te in folds:
        leak_counts.append(len(set(groups[tr]) & set(groups[te])))
        a = _select_alpha(X_emb, Z, y, groups, tr, n_pcs) if alpha is None else alpha
        alphas_used.append(float(a))

        X_tr, X_te = _build_design(X_emb, Z, tr, te, n_pcs)
        model = Ridge(alpha=a, random_state=RANDOM_SEED).fit(X_tr, y[tr])
        yhat[te] = model.predict(X_te)
        fold_sizes.append(int(len(te)))

    assert not np.isnan(yhat).any(), "Some observations received no out-of-fold prediction"
    assert max(leak_counts) == 0, f"Participant leakage detected: {leak_counts}"

    r = float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 0 else float("nan")
    out = {
        "n": int(n),
        "participants": int(len(np.unique(groups))),
        "n_predictors": int(
            (min(n_pcs, X_emb.shape[1]) if X_emb is not None else 0)
            + (Z.shape[1] if Z is not None else 0)
        ),
        "pearson_r": r,
        "R2": float(r2_score(y, yhat)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "fold_test_sizes": str(fold_sizes),
        "max_participant_overlap": int(max(leak_counts)),
        "alphas_selected": ",".join(f"{a:g}" for a in alphas_used) if alpha is None else "",
        "yhat": yhat,
    }
    if label is not None:
        out = {"analysis": label, **out}
    return out


def repeated_cv(
    X_emb: np.ndarray | None,
    Z: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    n_repeats: int = 20,
    alpha: float | None = RIDGE_ALPHA,
    n_pcs: int = N_PCS,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Spread of pooled out-of-fold metrics across random participant-to-fold splits.

    Quantifies fold-assignment uncertainty, which the single deterministic
    GroupKFold run does not expose. Report the SD alongside any single-run r
    when that r is being compared against another run on a different sample.
    """
    rs: list[float] = []
    r2s: list[float] = []
    for i in range(n_repeats):
        res = run_grouped_cv_blocks(
            X_emb, Z, y, groups,
            alpha=alpha, n_pcs=n_pcs, n_splits=n_splits,
            group_seed=seed + 1000 * i,
        )
        rs.append(res["pearson_r"])
        r2s.append(res["R2"])

    r_arr = np.asarray(rs, dtype=float)
    r2_arr = np.asarray(r2s, dtype=float)
    return {
        "n_repeats": int(n_repeats),
        "r_mean": float(np.mean(r_arr)),
        "r_sd": float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else float("nan"),
        "r_min": float(np.min(r_arr)),
        "r_max": float(np.max(r_arr)),
        "R2_mean": float(np.mean(r2_arr)),
        "R2_sd": float(np.std(r2_arr, ddof=1)) if len(r2_arr) > 1 else float("nan"),
        "R2_min": float(np.min(r2_arr)),
        "R2_max": float(np.max(r2_arr)),
    }


def _r_and_r2(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else np.nan
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r, r2


def bootstrap_delta(
    y: np.ndarray,
    yhat_a: np.ndarray,
    yhat_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = RANDOM_SEED,
) -> dict:
    """Participant-cluster bootstrap CI for (model A - model B) in r and R-squared.

    Participants are resampled with replacement; all of a participant's rows
    travel together, which respects the dependence induced by repeated daily
    observations. Percentile intervals are returned.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    r_a, r2_a = _r_and_r2(y, yhat_a)
    r_b, r2_b = _r_and_r2(y, yhat_b)

    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    d_r = np.full(n_boot, np.nan, dtype=float)
    d_r2 = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        yb = y[rows]
        if np.std(yb) == 0:
            continue
        ra, r2a = _r_and_r2(yb, yhat_a[rows])
        rb, r2b = _r_and_r2(yb, yhat_b[rows])
        d_r[b] = ra - rb
        d_r2[b] = r2a - r2b

    lo_q = 100.0 * (1.0 - level) / 2.0
    hi_q = 100.0 - lo_q
    d_r = d_r[np.isfinite(d_r)]
    d_r2 = d_r2[np.isfinite(d_r2)]

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if len(arr) else float("nan")

    return {
        "delta_r": r_a - r_b,
        "delta_r_CI_lo": pct(d_r, lo_q),
        "delta_r_CI_hi": pct(d_r, hi_q),
        "delta_R2": r2_a - r2_b,
        "delta_R2_CI_lo": pct(d_r2, lo_q),
        "delta_R2_CI_hi": pct(d_r2, hi_q),
        "boot_reps_used": int(min(len(d_r), len(d_r2))),
        # Fraction of replicates in which A did not beat B; a two-sided
        # bootstrap p-value is 2 * min(this, 1 - this).
        "frac_reps_delta_R2_le_0": float(np.mean(d_r2 <= 0)) if len(d_r2) else float("nan"),
    }
