"""Verify that every number quoted in the supplement's prose matches its source table.

The supplement's tables are generated from outputs/tables/, so they cannot drift.
The prose is hand-written, so it can. This script closes that gap: for each
quantity cited in the text, it recomputes the formatted value from the CSV and
asserts that exact string appears in the document. A rerun of the analyses that
moves a number therefore fails here rather than silently leaving stale prose.

Usage:
    python scripts/verify_supplement.py [--doc "NLP_Crisis_EMA Supplement.docx"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def dec(x, places: int = 3) -> str:
    s = f"{float(x):.{places}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def ci(lo, hi, places: int = 3) -> str:
    return f"[{dec(lo, places)}, {dec(hi, places)}]"


def delta(df, key_col, key, contrast_col, contrast, places=3) -> str:
    r = df[(df[key_col] == key) & (df[contrast_col] == contrast)]
    assert len(r) == 1, f"{key}/{contrast}: matched {len(r)} rows"
    r = r.iloc[0]
    return f'{dec(r["delta_R2"], places)} (95% CI {ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"], places)})'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="NLP_Crisis_EMA Supplement.docx")
    args = ap.parse_args()

    doc = Document(ROOT / args.doc)
    text = " ".join(p.text for p in doc.paragraphs)
    text = " ".join(text.split())

    arb = pd.read_csv(TABLES / "ar1_benchmark_contrasts.csv")
    arb = arb[arb["alpha_spec"] == "fixed_alpha_10"]
    ard = pd.read_csv(TABLES / "ar1_decomposition_models.csv")
    arc = pd.read_csv(TABLES / "ar1_decomposition_contrasts.csv")
    tps = pd.read_csv(TABLES / "text_plus_symptoms_contrasts.csv")
    tps = tps[tps["alpha_spec"] == "fixed_alpha_10"]
    bw = pd.read_csv(TABLES / "covariate_between_within_decomposition.csv")
    dow = pd.read_csv(TABLES / "covariate_dayofweek_models.csv")
    dowc = pd.read_csv(TABLES / "covariate_dayofweek_contrasts.csv")
    plc = pd.read_csv(TABLES / "covariate_personlevel_contrasts.csv")
    pfs = pd.read_csv(TABLES / "covariate_personlevel_fold_stability.csv")
    rec = pd.read_csv(TABLES / "covariate_text_recovers_demographics.csv")

    def bwv(analysis, col, places=2):
        return dec(bw.loc[bw["analysis"] == analysis, col].iloc[0], places)

    def fs(analysis, model, col, places=3):
        r = pfs[(pfs["analysis"] == analysis) & (pfs["model"] == model)]
        assert len(r) == 1
        return dec(r.iloc[0][col], places)

    claims: list[tuple[str, str]] = []

    # S2: diary text added to the autoregressive term
    for a in ["crisis_concurrent", "dsi_concurrent", "crisis_prospective", "dsi_prospective"]:
        claims.append((f"S2 text+lag vs lag, {a}",
                       delta(arb, "analysis", a, "contrast", "text_plus_lag_vs_lag_only")))

    # S3: between/within split of the lagged predictor
    btw = ard[ard["model"] == "lag_between_only"]["R2"]
    wth = ard[ard["model"] == "lag_within_only"]["R2"]
    claims.append(("S3 between-part R2 range",
                   f"R-squared between {dec(btw.min(), 2)} and {dec(btw.max(), 2)}"))
    claims.append(("S3 within-part R2 range",
                   f"R-squared between {dec(wth.min())} and {dec(wth.max())}"))
    claims.append(("S3 text added to within part",
                   delta(arc, "analysis", "crisis_concurrent", "contrast",
                         "text_plus_lag_within_vs_lag_within")))
    claims.append(("S3 text added to both parts, concurrent",
                   delta(arc, "analysis", "crisis_concurrent", "contrast",
                         "text_plus_lag_both_vs_lag_both")))
    claims.append(("S3 text added to both parts, prospective",
                   delta(arc, "analysis", "crisis_prospective", "contrast",
                         "text_plus_lag_both_vs_lag_both")))

    # S4: diary text added to concurrently reported symptoms
    claims.append(("S4 crisis prospective vs crisis total",
                   delta(tps, "outcome", "crisis_prospective", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S4 dsi prospective vs crisis total",
                   delta(tps, "outcome", "dsi_prospective", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S4 dsi concurrent vs crisis total",
                   delta(tps, "outcome", "dsi_concurrent", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S4 dsi concurrent vs crisis items",
                   delta(tps, "outcome", "dsi_concurrent", "contrast",
                         "text_plus_crisis_items_vs_crisis_items")))
    for c, tag in [("text_only_vs_crisis_total", "total"),
                   ("text_only_vs_crisis_items", "items")]:
        r = tps[(tps["outcome"] == "dsi_prospective") & (tps["contrast"] == c)].iloc[0]
        claims.append((f"S4 dsi prospective text vs crisis {tag}",
                       f'{dec(r["delta_R2"])}, 95% CI '
                       f'{ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"])}'))

    # S5: between/within decomposition of the text models
    claims.append(("S5 concurrent within-person r",
                   f'within-person correlations of {bwv("crisis_concurrent", "r_within_mean")} '
                   f'and {bwv("dsi_concurrent", "r_within_mean")}'))
    claims.append(("S5 prospective between-person r",
                   f'between-person correlations of {bwv("crisis_prospective", "r_between_mean")} '
                   f'and {bwv("dsi_prospective", "r_between_mean")}'))
    claims.append(("S5 prospective within-person r",
                   f'correlations of only {bwv("crisis_prospective", "r_within_mean")} '
                   f'and {bwv("dsi_prospective", "r_within_mean")}'))
    claims.append(("S5 within-person SD range",
                   f'standard deviations between {dec(bw["r_within_sd"].min())} '
                   f'and {dec(bw["r_within_sd"].max())}'))

    # S6: day of week
    d_alone = dow[dow["model"] == "dow_only"]["R2"]
    claims.append(("S6 dow-only R2 range",
                   f"R-squared between {dec(d_alone.min())} and {dec(d_alone.max())}"))
    claims.append(("S6 adding dow to text, crisis concurrent",
                   delta(dowc, "analysis", "crisis_concurrent", "contrast",
                         "adding dow to text")))
    claims.append(("S6 adding dow to text, dsi prospective",
                   delta(dowc, "analysis", "dsi_prospective", "contrast",
                         "adding dow to text")))

    # S7: age and sex
    pa = rec[(rec["level"] == "person") & (rec["target"] == "age")].iloc[0]
    pm = rec[(rec["level"] == "person") & (rec["target"] == "male")].iloc[0]
    claims.append(("S7 person-level age recovery",
                   f'correlation of {dec(pa["pearson_r"], 2)} at the participant level'))
    claims.append(("S7 person-level sex AUROC",
                   f'curve of {dec(pm["AUROC"], 2)}'))
    claims.append(("S7 adding demo to text, crisis concurrent",
                   delta(plc, "analysis", "crisis_concurrent", "contrast",
                         "adding demo to text")))
    claims.append(("S7 adding demo to text, crisis prospective",
                   delta(plc, "analysis", "crisis_prospective", "contrast",
                         "adding demo to text")))
    r1 = plc[(plc["analysis"] == "crisis_concurrent") &
             (plc["contrast"] == "adding text to demo")].iloc[0]
    r2 = plc[(plc["analysis"] == "crisis_prospective") &
             (plc["contrast"] == "adding text to demo")].iloc[0]
    claims.append(("S7 adding text to demo, both outcomes",
                   f'R-squared by {dec(r1["delta_R2"])} '
                   f'(95% CI {ci(r1["delta_R2_CI_lo"], r1["delta_R2_CI_hi"])}) '
                   f'and {dec(r2["delta_R2"])} '
                   f'(95% CI {ci(r2["delta_R2_CI_lo"], r2["delta_R2_CI_hi"])})'))
    claims.append(("S7 demo-only person-level R2",
                   f'R-squared of {fs("crisis_concurrent", "demo_only", "R2_mean")} '
                   f'and {fs("crisis_prospective", "demo_only", "R2_mean")}'))

    failures = []
    for label, needle in claims:
        ok = " ".join(needle.split()) in text
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        expected in prose: {needle}")
            failures.append(label)

    print()
    if failures:
        print(f"{len(failures)} of {len(claims)} CLAIM(S) NOT FOUND: {failures}")
        raise SystemExit(1)
    print(f"all {len(claims)} prose claims match their source tables")


if __name__ == "__main__":
    main()
