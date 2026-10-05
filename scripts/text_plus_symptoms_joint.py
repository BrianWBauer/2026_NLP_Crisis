"""Joint text + crisis-symptom prediction of daily suicide risk.

Addresses the reviewer point that the manuscript should report what text and
crisis symptoms give *together*. The existing head-to-head notebook
(04e_text_vs_crisis_headtohead) answered this for BINARY risk outcomes with a
logistic classifier; this script answers it for the CONTINUOUS outcomes and the
analytic sample used in manuscript Table 2, under Table 2's own
cross-validation specification, so the numbers are directly comparable to the
published text-only results.

Design
    For each outcome, a set of nested predictor blocks is cross-validated:
      - a self-report block alone (crisis total, crisis items, or crisis + DSI)
      - the text block alone (20 embedding PCs)
      - both together
    Within an outcome, every model is fit on the SAME rows - the complete cases
    across all predictor columns any model uses - so delta R-squared between
    models reflects predictors only, never a change of sample. The text-only
    model is also reported on the full Table 2 sample as a reference row.

    The headline contrast is text_plus_selfreport vs selfreport_only: the
    incremental validity of language over the questionnaire data a study would
    already have. The reverse contrast is reported as well, because
    "self-report adds to text" is the claim a passive-sensing framing cares
    about.

Outcomes
    1. PM DSI-SS total (t)        - concurrent suicidal ideation
    2. AM DSI-SS total (t+1)      - next-morning suicidal ideation
    3. AM crisis total (t+1)      - next-morning crisis symptoms
    The concurrent PM crisis outcome is deliberately omitted: the self-report
    block for that outcome would be the outcome itself.

    All t+1 outcomes use an EXACT calendar join (ema_date + 1 day). A bare
    groupby().shift(-1) is never used.

Specification
    Identical to verify_text_only_aim1.py / Table 2 - all-mpnet-base-v2
    embeddings, PCA(20) and StandardScaler fit on training folds only,
    Ridge(alpha=10.0), GroupKFold(5) by participant, metrics on pooled
    out-of-fold predictions. Self-report predictors are standardized by a
    scaler fit on training folds only. A sensitivity specification selects
    alpha per fold by inner participant-grouped CV, which matters because the
    self-report blocks are far lower-dimensional than the text block and a
    single fixed penalty is not equally appropriate for both.

Requires the protected local data (data/processed, data/raw), which are not in
the public repository. Result tables are written to outputs/tables/, covered by
the repository's *.csv ignore rule; no participant text or participant-level
data are written.

Usage:
    python scripts/text_plus_symptoms_joint.py
    python scripts/text_plus_symptoms_joint.py --n-boot 5000
    python scripts/text_plus_symptoms_joint.py --skip-tuned
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

from verify_text_only_aim1 import (  # noqa: E402
    DATE_COL,
    DT_COL,
    OUT_DIR,
    PID_COL,
    PM_CRISIS_COL,
    PM_DSI_COL,
    RANDOM_SEED,
    RAW_PATH,
    build_am_day_outcomes,
    find_dsi_cols_by_prefix,
    first_nonnull,
    load_and_verify_alignment,
)
from cv_blocks import (  # noqa: E402
    ALPHA_GRID,
    N_PCS,
    N_SPLITS,
    RIDGE_ALPHA,
    bootstrap_delta,
    repeated_cv,
    run_grouped_cv_blocks,
)

# The five crisis items share a Likert stem in the exported column names.
CRISIS_ITEM_STEM = "dailyPM__[ 0 = 'Not at all'"
N_CRISIS_ITEMS = 5
DEFAULT_N_BOOT = 2000

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Item-level predictors from raw
# -----------------------------------------------------------------------------
def find_crisis_item_cols(df: pd.DataFrame) -> list[str]:
    """The five PM crisis items that make up 'Total Score from 5 Questions'."""
    cols = [c for c in df.columns if c.startswith(CRISIS_ITEM_STEM)]
    if len(cols) != N_CRISIS_ITEMS:
        raise ValueError(
            f"Expected {N_CRISIS_ITEMS} PM crisis items matching {CRISIS_ITEM_STEM!r}, "
            f"found {len(cols)}"
        )
    return cols


def build_pm_items() -> pd.DataFrame:
    """Person-day PM crisis items and PM DSI-SS items, with a total-score check."""
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw[DT_COL] = pd.to_datetime(raw[DT_COL], errors="coerce")
    raw[DATE_COL] = raw[DT_COL].dt.normalize()

    crisis_cols = find_crisis_item_cols(raw)
    dsi_map = find_dsi_cols_by_prefix(raw, "dailyPM__")
    dsi_cols = [dsi_map[k] for k in ("A", "B", "C", "D")]
    if any(c is None for c in dsi_cols):
        raise ValueError(f"Could not detect all four PM DSI items: {dsi_map}")

    rename = {c: f"crisis_item_{i + 1}" for i, c in enumerate(crisis_cols)}
    rename.update({c: f"dsi_item_{k}" for c, k in zip(dsi_cols, "ABCD")})

    agg = {new: (old, first_nonnull) for old, new in rename.items()}
    day = (
        raw.sort_values([PID_COL, DT_COL])
        .groupby([PID_COL, DATE_COL], as_index=False)
        .agg(**agg)
    )

    # Confirm the five detected items really are the five scored questions.
    item_cols = [f"crisis_item_{i + 1}" for i in range(N_CRISIS_ITEMS)]
    tot = (
        raw.sort_values([PID_COL, DT_COL])
        .groupby([PID_COL, DATE_COL], as_index=False)
        .agg(total=("dailyPM__Total Score from 5 Questions", first_nonnull))
    )
    chk = day.merge(tot, on=[PID_COL, DATE_COL], how="inner")
    complete = chk[item_cols].notna().all(axis=1) & chk["total"].notna()
    recomputed = chk.loc[complete, item_cols].sum(axis=1)
    assert np.allclose(recomputed, chk.loc[complete, "total"]), (
        "Sum of the five detected PM crisis items does not reproduce "
        "'Total Score from 5 Questions'"
    )
    print(f"  Verified: sum of 5 detected PM crisis items == stored total on "
          f"{int(complete.sum())} complete person-days")
    return day


def build_panel(feat: pd.DataFrame, am_day: pd.DataFrame, pm_items: pd.DataFrame) -> pd.DataFrame:
    """Table 2 rows plus same-day item-level self-report and next-day outcomes."""
    panel = feat[[PID_COL, DATE_COL, PM_CRISIS_COL, PM_DSI_COL]].copy()
    panel["embed_row"] = np.arange(len(panel))

    panel = panel.merge(pm_items, on=[PID_COL, DATE_COL], how="left")

    nxt = am_day.rename(
        columns={
            DATE_COL: "_target",
            "crisis_AM_total": "crisis_AM_next",
            "dsi_AM_total": "dsi_AM_next",
        }
    )
    panel["_target"] = panel[DATE_COL] + pd.Timedelta(days=1)
    panel = panel.merge(nxt, on=[PID_COL, "_target"], how="left").drop(columns="_target")

    assert len(panel) == len(feat), "Panel join changed the row count"
    assert panel["embed_row"].is_unique, "Duplicate rows produced by the panel joins"
    return panel


# -----------------------------------------------------------------------------
# Predictor blocks and model grid
# -----------------------------------------------------------------------------
CRISIS_ITEMS = [f"crisis_item_{i + 1}" for i in range(N_CRISIS_ITEMS)]
DSI_ITEMS = [f"dsi_item_{k}" for k in "ABCD"]

# outcome label -> (outcome column, description, {block name: columns})
OUTCOME_SPECS: list[tuple[str, str, str, dict[str, list[str]]]] = [
    (
        "dsi_concurrent",
        PM_DSI_COL,
        "PM DSI-SS total (t): concurrent suicidal ideation",
        {
            "crisis_total": [PM_CRISIS_COL],
            "crisis_items": CRISIS_ITEMS,
        },
    ),
    (
        "dsi_prospective",
        "dsi_AM_next",
        "AM DSI-SS total (t+1): next-morning suicidal ideation",
        {
            "crisis_total": [PM_CRISIS_COL],
            "crisis_items": CRISIS_ITEMS,
            "selfreport_all": [PM_CRISIS_COL, PM_DSI_COL],
        },
    ),
    (
        "crisis_prospective",
        "crisis_AM_next",
        "AM crisis total (t+1): next-morning crisis symptoms",
        {
            "crisis_total": [PM_CRISIS_COL],
            "crisis_items": CRISIS_ITEMS,
            "selfreport_all": [PM_CRISIS_COL, PM_DSI_COL],
        },
    ),
]


def run_outcome(
    label: str,
    y_col: str,
    desc: str,
    blocks: dict[str, list[str]],
    panel: pd.DataFrame,
    emb: np.ndarray,
    n_boot: int,
    tune_alpha: bool,
) -> tuple[list[dict], list[dict]]:
    """Fit every self-report / text / joint model for one outcome.

    The complete-case sample is computed SEPARATELY for each self-report block,
    not once across all blocks. Each three-model comparison (block alone, text
    alone, both) therefore uses the largest sample available to it, while the
    three models inside a comparison still share identical rows and identical
    folds. Collapsing to one sample across blocks would discard rows for
    comparisons that do not need them, and - because GroupKFold is deterministic
    in participant size order - would also reshuffle the folds, which moves the
    reported r by more than the lost rows do.
    """
    model_rows: list[dict] = []
    contrast_rows: list[dict] = []

    alpha_specs = [("fixed_alpha_10", False)]
    if tune_alpha:
        alpha_specs.append(("inner_cv_alpha", True))

    full = panel.loc[panel[y_col].notna()]

    # Text-only on the full Table 2 sample, as a reference row per outcome.
    ref = run_grouped_cv_blocks(
        emb[full["embed_row"].to_numpy()],
        None,
        full[y_col].to_numpy(float),
        full[PID_COL].astype(str).to_numpy(),
        alpha=RIDGE_ALPHA,
    )
    model_rows.append(
        {
            "outcome": label,
            "description": desc,
            "alpha_spec": "fixed_alpha_10",
            "block": "(none)",
            "model": "text_only_FULL_SAMPLE",
            "sample": "full Table 2 sample (reference)",
            "selfreport_columns": "",
            **{k: v for k, v in ref.items() if k != "yhat"},
        }
    )

    for bname, cols in blocks.items():
        mask = panel[y_col].notna()
        for c in cols:
            mask &= panel[c].notna()
        sub = panel.loc[mask]

        idx = sub["embed_row"].to_numpy()
        y = sub[y_col].to_numpy(float)
        g = sub[PID_COL].astype(str).to_numpy()
        Z = sub[cols].to_numpy(float)

        for spec_name, tune in alpha_specs:
            alpha = None if tune else RIDGE_ALPHA
            fits = {
                "text_only": run_grouped_cv_blocks(emb[idx], None, y, g, alpha=alpha),
                bname: run_grouped_cv_blocks(None, Z, y, g, alpha=alpha),
                f"text_plus_{bname}": run_grouped_cv_blocks(emb[idx], Z, y, g, alpha=alpha),
            }

            for mname, res in fits.items():
                model_rows.append(
                    {
                        "outcome": label,
                        "description": desc,
                        "alpha_spec": spec_name,
                        "block": bname,
                        "model": mname,
                        "sample": f"complete cases for {bname}",
                        "selfreport_columns": "; ".join(cols) if mname != "text_only" else "",
                        **{k: v for k, v in res.items() if k != "yhat"},
                    }
                )

            joint = f"text_plus_{bname}"
            for cname, a, b in [
                (f"{joint}_vs_{bname}", joint, bname),
                (f"{joint}_vs_text_only", joint, "text_only"),
                (f"text_only_vs_{bname}", "text_only", bname),
            ]:
                boot = bootstrap_delta(y, fits[a]["yhat"], fits[b]["yhat"], g, n_boot=n_boot)
                contrast_rows.append(
                    {
                        "outcome": label,
                        "alpha_spec": spec_name,
                        "block": bname,
                        "contrast": cname,
                        "n": int(len(y)),
                        "participants": int(len(np.unique(g))),
                        "r_a": fits[a]["pearson_r"],
                        "r_b": fits[b]["pearson_r"],
                        "R2_a": fits[a]["R2"],
                        "R2_b": fits[b]["R2"],
                        **boot,
                    }
                )

    return model_rows, contrast_rows


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_results(models: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    for spec in models["alpha_spec"].unique():
        print("\n" + "-" * 78)
        print(f"ALPHA SPECIFICATION: {spec}")
        print("-" * 78)
        for label, _, desc, _ in OUTCOME_SPECS:
            m = models.loc[(models["outcome"] == label) & (models["alpha_spec"] == spec)]
            if m.empty:
                continue
            print(f"\n{label}")
            print(f"  {desc}")
            for block in m["block"].unique():
                mb = m.loc[m["block"] == block]
                print(f"    [block: {block}]")
                for _, r in mb.iterrows():
                    extra = ""
                    if isinstance(r.get("alphas_selected"), str) and r["alphas_selected"]:
                        extra = f"  alphas={r['alphas_selected']}"
                    print(
                        f"      {r['model']:<28s} n={int(r['n']):>5d} "
                        f"p={int(r['n_predictors']):>3d}  r={r['pearson_r']:.4f}  "
                        f"R2={r['R2']:>8.4f}  RMSE={r['RMSE']:.4f}{extra}"
                    )
                c = contrasts.loc[
                    (contrasts["outcome"] == label)
                    & (contrasts["alpha_spec"] == spec)
                    & (contrasts["block"] == block)
                ]
                for _, r in c.iterrows():
                    print(
                        f"        {r['contrast']:<42s} "
                        f"dr={r['delta_r']:+.4f} [{r['delta_r_CI_lo']:+.4f}, "
                        f"{r['delta_r_CI_hi']:+.4f}]  "
                        f"dR2={r['delta_R2']:+.4f} [{r['delta_R2_CI_lo']:+.4f}, "
                        f"{r['delta_R2_CI_hi']:+.4f}]"
                    )


def fold_stability(
    panel: pd.DataFrame, emb: np.ndarray, n_repeats: int
) -> pd.DataFrame:
    """Spread of each headline model's metrics across random fold assignments.

    GroupKFold is deterministic, so a single run gives no sense of how much the
    reported r depends on which participants landed in which fold. Any
    comparison between two runs on slightly different samples has to be read
    against this spread.
    """
    rows: list[dict] = []
    for label, y_col, _, blocks in OUTCOME_SPECS:
        cols = blocks["crisis_total"]
        mask = panel[y_col].notna()
        for c in cols:
            mask &= panel[c].notna()
        sub = panel.loc[mask]
        idx = sub["embed_row"].to_numpy()
        y = sub[y_col].to_numpy(float)
        g = sub[PID_COL].astype(str).to_numpy()
        Z = sub[cols].to_numpy(float)

        for mname, X_, Z_ in [
            ("text_only", emb[idx], None),
            ("crisis_total", None, Z),
            ("text_plus_crisis_total", emb[idx], Z),
        ]:
            rows.append(
                {
                    "outcome": label,
                    "model": mname,
                    "n": int(len(y)),
                    **repeated_cv(X_, Z_, y, g, n_repeats=n_repeats, alpha=RIDGE_ALPHA),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="participant-cluster bootstrap replicates for contrast CIs")
    parser.add_argument("--skip-tuned", action="store_true",
                        help="report only the fixed alpha=10 specification")
    parser.add_argument("--fold-repeats", type=int, default=20,
                        help="random fold assignments for the stability check (0 to skip)")
    args = parser.parse_args()

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    feat, emb = load_and_verify_alignment()
    am_day = build_am_day_outcomes()
    pm_items = build_pm_items()
    panel = build_panel(feat, am_day, pm_items)

    print(f"  Analytic PM diaries        : {len(panel)} "
          f"({panel[PID_COL].nunique()} participants)")
    print(f"  Embeddings                 : {emb.shape}, L2-normalized, row-aligned")
    print(f"  CV                         : GroupKFold({N_SPLITS}) by participant, "
          f"PCA({N_PCS}) + StandardScaler fit in-fold")
    print(f"  Ridge alpha                : {RIDGE_ALPHA} (fixed); "
          f"inner-CV grid {list(ALPHA_GRID)}")
    print(f"  Bootstrap replicates       : {args.n_boot} (participant clusters)")
    print("  Self-report blocks:")
    print(f"    crisis_total   : {PM_CRISIS_COL}")
    print(f"    crisis_items   : {', '.join(CRISIS_ITEMS)}")
    print(f"    selfreport_all : {PM_CRISIS_COL}, {PM_DSI_COL}")

    all_models: list[dict] = []
    all_contrasts: list[dict] = []
    for label, y_col, desc, blocks in OUTCOME_SPECS:
        m, c = run_outcome(
            label, y_col, desc, blocks, panel, emb,
            n_boot=args.n_boot, tune_alpha=not args.skip_tuned,
        )
        all_models.extend(m)
        all_contrasts.extend(c)

    models = pd.DataFrame(all_models)
    contrasts = pd.DataFrame(all_contrasts)

    print()
    print("=" * 78)
    print("RESULTS (pooled out-of-fold predictions)")
    print("=" * 78)
    print_results(models, contrasts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_m = OUT_DIR / "text_plus_symptoms_models.csv"
    p_c = OUT_DIR / "text_plus_symptoms_contrasts.csv"
    models.to_csv(p_m, index=False)
    contrasts.to_csv(p_c, index=False)
    print(f"\nSaved: {p_m}")
    print(f"Saved: {p_c}")

    if args.fold_repeats > 0:
        print()
        print("=" * 78)
        print(f"FOLD-ASSIGNMENT STABILITY ({args.fold_repeats} random participant splits)")
        print("=" * 78)
        print("  GroupKFold is deterministic, so a single run hides this spread. Two")
        print("  runs on samples differing by a handful of rows can differ by roughly")
        print("  the SD below purely because the folds were rebuilt.")
        stab = fold_stability(panel, emb, args.fold_repeats)
        for label, _, _, _ in OUTCOME_SPECS:
            s = stab.loc[stab["outcome"] == label]
            if s.empty:
                continue
            print(f"\n{label}")
            for _, r in s.iterrows():
                print(
                    f"    {r['model']:<28s} n={int(r['n']):>5d}  "
                    f"r={r['r_mean']:.4f} (SD {r['r_sd']:.4f}, "
                    f"range {r['r_min']:.4f}-{r['r_max']:.4f})  "
                    f"R2={r['R2_mean']:>8.4f} (SD {r['R2_sd']:.4f})"
                )
        p_s = OUT_DIR / "text_plus_symptoms_fold_stability.csv"
        stab.to_csv(p_s, index=False)
        print(f"\nSaved: {p_s}")


if __name__ == "__main__":
    main()
