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

    tps = pd.read_csv(TABLES / "text_plus_symptoms_contrasts.csv")
    tps = tps[tps["alpha_spec"] == "fixed_alpha_10"]
    bw = pd.read_csv(TABLES / "covariate_between_within_decomposition.csv")

    def bwv(analysis, col, places=2):
        return dec(bw.loc[bw["analysis"] == analysis, col].iloc[0], places)

    claims: list[tuple[str, str]] = []

    # S8: diary text added to concurrently reported symptoms
    claims.append(("S8 crisis prospective vs crisis total",
                   delta(tps, "outcome", "crisis_prospective", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S8 dsi prospective vs crisis total",
                   delta(tps, "outcome", "dsi_prospective", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S8 dsi concurrent vs crisis total",
                   delta(tps, "outcome", "dsi_concurrent", "contrast",
                         "text_plus_crisis_total_vs_crisis_total")))
    claims.append(("S8 dsi concurrent vs crisis items",
                   delta(tps, "outcome", "dsi_concurrent", "contrast",
                         "text_plus_crisis_items_vs_crisis_items")))
    for c, tag in [("text_only_vs_crisis_total", "total"),
                   ("text_only_vs_crisis_items", "items")]:
        r = tps[(tps["outcome"] == "dsi_prospective") & (tps["contrast"] == c)].iloc[0]
        claims.append((f"S8 dsi prospective text vs crisis {tag}",
                       f'{dec(r["delta_R2"])}, 95% CI '
                       f'{ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"])}'))

    # S9: between/within decomposition of the text models
    claims.append(("S9 concurrent within-person r",
                   f'within-person correlations of {bwv("crisis_concurrent", "r_within_mean")} '
                   f'and {bwv("dsi_concurrent", "r_within_mean")}'))
    claims.append(("S9 prospective between-person r",
                   f'between-person correlations of {bwv("crisis_prospective", "r_between_mean")} '
                   f'and {bwv("dsi_prospective", "r_between_mean")}'))
    claims.append(("S9 prospective within-person r",
                   f'correlations of only {bwv("crisis_prospective", "r_within_mean")} '
                   f'and {bwv("dsi_prospective", "r_within_mean")}'))
    claims.append(("S9 within-person SD range",
                   f'standard deviations between {dec(bw["r_within_sd"].min())} '
                   f'and {dec(bw["r_within_sd"].max())}'))

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
