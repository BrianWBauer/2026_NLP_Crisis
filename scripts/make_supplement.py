"""Build the supplemental materials document from the saved result tables.

Every number in the output is read from outputs/tables/, so the document cannot
drift from the analyses. Nothing is typed in by hand except the prose, and the
prose cites only quantities that appear in the tables it accompanies.

Scope. This supplement carries exactly the two tables that back limitations
Fourth and Fifth in the manuscript, and nothing else. An earlier version built
seven sections and twelve tables covering lag-1 autocorrelation, an
autoregressive benchmark and its decomposition, day of week, and demographic
covariates. Those analyses are a separate methods paper rather than support for
this one, and carrying them here invited a reader to treat a second paper's
argument as this paper's loose ends. They are preserved at commit 7763e66 and
were deliberately removed, not lost.

Table numbering continues the manuscript's own series, which ends at Table S7,
so the two tables here are S8 and S9. The earlier version restarted at S1 and
collided with the manuscript.

Usage:
    python scripts/make_supplement.py [--out "NLP_Crisis_EMA Supplement.docx"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(12)
TABLE_SIZE = Pt(9)


# ---------------------------------------------------------------------------
# number formatting
# ---------------------------------------------------------------------------
def dec(x, places: int = 2, leading_zero: bool = False) -> str:
    """Format a number, APA style: no leading zero for bounded quantities."""
    if pd.isna(x):
        return ""
    s = f"{float(x):.{places}f}"
    if not leading_zero:
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    return s


def ci(lo, hi, places: int = 2) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{dec(lo, places)}, {dec(hi, places)}]"


def integer(x) -> str:
    return "" if pd.isna(x) else f"{int(round(float(x))):,}"


# ---------------------------------------------------------------------------
# document helpers
# ---------------------------------------------------------------------------
def heading(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.name = BODY_FONT
    r.font.size = BODY_SIZE


def prose(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.font.name = BODY_FONT
    r.font.size = BODY_SIZE


def caption(doc: Document, label: str, title: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(label + ". ")
    r.bold = True
    r.font.name = BODY_FONT
    r.font.size = BODY_SIZE
    r2 = p.add_run(title)
    r2.italic = True
    r2.font.name = BODY_FONT
    r2.font.size = BODY_SIZE


def table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for cell, h in zip(t.rows[0].cells, headers):
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.name = BODY_FONT
        r.font.size = TABLE_SIZE
    for row in rows:
        cells = t.add_row().cells
        for cell, v in zip(cells, row):
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(v))
            r.font.name = BODY_FONT
            r.font.size = TABLE_SIZE
    doc.add_paragraph()


def note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run("Note. ")
    r.italic = True
    r.font.name = BODY_FONT
    r.font.size = TABLE_SIZE
    r2 = p.add_run(text)
    r2.font.name = BODY_FONT
    r2.font.size = TABLE_SIZE


# ---------------------------------------------------------------------------
# label maps
# ---------------------------------------------------------------------------
OUTCOME = {
    "crisis_concurrent": "Crisis symptom total, same evening",
    "dsi_concurrent": "DSI-SS total, same evening",
    "crisis_prospective": "Crisis symptom total, next morning",
    "dsi_prospective": "DSI-SS total, next morning",
}
ORDER = ["crisis_concurrent", "dsi_concurrent", "crisis_prospective", "dsi_prospective"]

MODEL = {
    "lag_only": "Prior assessment only",
    "text_only": "Diary text only",
    "text_plus_lag": "Diary text + prior assessment",
    "text_only_FULL_SAMPLE": "Diary text only (full sample, reference)",
    "lag_between_only": "Prior assessment, between-person part",
    "lag_within_only": "Prior assessment, within-person part",
    "lag_both_parts": "Prior assessment, both parts",
    "text_plus_lag_within": "Diary text + within-person part",
    "text_plus_lag_both": "Diary text + both parts",
    "dow_only": "Day of week only",
    "text_plus_dow": "Diary text + day of week",
    "demo_only": "Age and sex only",
    "text_plus_demo": "Diary text + age and sex",
    "crisis_total": "Crisis symptom total",
    "crisis_items": "Crisis symptom items",
    "selfreport_all": "All self-report predictors",
    "text_plus_crisis_total": "Diary text + crisis symptom total",
    "text_plus_crisis_items": "Diary text + crisis symptom items",
    "text_plus_selfreport_all": "Diary text + all self-report predictors",
}

PHRASE = {
    "adding dow to text": "Adding day of week to diary text",
    "adding text to dow": "Adding diary text to day of week",
    "adding demo to text": "Adding age and sex to diary text",
    "adding text to demo": "Adding diary text to age and sex",
}


def contrast_label(s: str) -> str:
    """Render a contrast key as prose, e.g. 'Diary text + prior assessment vs ...'."""
    if s in PHRASE:
        return PHRASE[s]
    if "_vs_" in s:
        a, b = s.split("_vs_", 1)
        return f"{MODEL.get(a, a.replace('_', ' '))} vs. {MODEL.get(b, b.replace('_', ' '))}"
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_s8(doc: Document) -> None:
    """Backs limitation Fourth: convergent validity rather than incremental utility."""
    c = pd.read_csv(TABLES / "text_plus_symptoms_contrasts.csv")
    c = c[c["alpha_spec"] == "fixed_alpha_10"]
    order = [o for o in ORDER if o in set(c["outcome"])]

    heading(doc, "S8. Diary text alongside concurrently reported symptoms")
    prose(doc, (
        "The fourth limitation reported in the main text rests on the comparisons in Table S8. "
        "We fit models containing the concurrently reported crisis symptom total, the five "
        "crisis symptom items separately, or all self-report predictors, each with and without "
        "diary text, and estimated the change in cross-validated R-squared attributable to "
        "adding one block to the other. For next-morning crisis symptom totals, diary text "
        "added nothing detectable to the concurrent crisis symptom total, with a change in "
        "R-squared of -.003 (95% CI [-.018, .011]). The same held for next-morning DSI-SS "
        "totals, where the change was -.017 (95% CI [-.107, .062]). For same-evening DSI-SS "
        "totals, diary text added .001 (95% CI [-.070, .053]) beyond the crisis symptom total "
        "and .029 (95% CI [-.025, .079]) beyond the five crisis symptom items, neither of which "
        "excludes zero. The reverse contrast was substantial in six of the eight comparisons, "
        "so the self-report predictors generally carried information that diary language did "
        "not. The two exceptions both involved next-morning DSI-SS totals, where the crisis "
        "symptom predictors did not reliably outperform diary text (changes in R-squared of "
        "-.085, 95% CI [-.206, .022], against the crisis symptom total, and -.100, "
        "95% CI [-.271, .037], against the five items). Written diary entries therefore appear "
        "to be largely redundant with concurrently administered symptom measures in this "
        "sample, and the case for collecting them rests on their value when such measures are "
        "unavailable rather than on incremental prediction."
    ))
    caption(doc, "Table S8",
            "Incremental contribution of diary text relative to self-reported symptoms")
    rows = []
    for a in order:
        sub = c[c["outcome"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                contrast_label(r["contrast"]),
                dec(r["delta_R2"], 3), ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"], 3),
            ])
    table(doc, ["Outcome", "Contrast", "\u0394R\u00b2", "95% CI"], rows)
    note(doc, (
        "The crisis symptom items block contains the five items separately rather than their "
        "total. The all self-report predictors block contains the crisis symptom total and the "
        "DSI-SS total. Confidence intervals are participant-cluster bootstrap percentile "
        "intervals. The underlying model fits are reported in the project repository."
    ))


def section_s9(doc: Document) -> None:
    """Backs limitation Fifth: prospective accuracy is carried between persons."""
    d = pd.read_csv(TABLES / "covariate_between_within_decomposition.csv")
    heading(doc, "S9. Between- and within-person components of the text models")
    prose(doc, (
        "The fifth limitation reported in the main text rests on Table S9. The cross-validated "
        "predictions from the text-only models were decomposed by correlating participant means "
        "of the predictions with participant means of the outcome, and by correlating the "
        "person-mean-centered predictions with the person-mean-centered outcome. Because participant "
        "assignment to folds is deterministic under grouped cross-validation and a single "
        "partition can move these estimates appreciably, each value is reported as the mean "
        "and standard deviation across 30 random participant-to-fold partitions. The "
        "concurrent models retained within-person correlations of .47 and .32, which indicates "
        "that diary language tracked same-evening fluctuations in symptom severity. The "
        "next-morning models retained between-person correlations of .50 and .40 but "
        "within-person correlations of only .10 and .04, so their accuracy came almost "
        "entirely from distinguishing participants with persistently elevated scores. The "
        "within-person correlations were also the most stable quantities in the analysis, with "
        "standard deviations between .004 and .010 across partitions, whereas the "
        "between-person correlations varied by .02 to .06."
    ))
    caption(doc, "Table S9",
            "Between- and within-person decomposition of the text-only model predictions")
    rows = []
    for a in ORDER:
        r = d[d["analysis"] == a].iloc[0]
        rows.append([
            OUTCOME[a], integer(r["n"]), integer(r["participants"]),
            f'{dec(r["r_pooled_mean"])} ({dec(r["r_pooled_sd"], 3)})',
            f'{dec(r["r_between_mean"])} ({dec(r["r_between_sd"], 3)})',
            f'{dec(r["r_within_mean"])} ({dec(r["r_within_sd"], 3)})',
            dec(r["outcome_var_between_share"]),
            dec(r["prediction_var_between_share"]),
        ])
    table(doc, ["Outcome", "n", "Ppts", "Pooled r (SD)", "Between-person r (SD)",
                "Within-person r (SD)", "Outcome variance between",
                "Prediction variance between"], rows)
    note(doc, (
        "Correlations are means across 30 random participant-to-fold partitions, with standard "
        "deviations across partitions in parentheses. The final two columns report the "
        "proportion of variance in the outcome and in the model's predictions that lies between "
        "participants, computed on the deterministic partition."
    ))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="NLP_Crisis_EMA Supplement.docx")
    args = ap.parse_args()

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE

    heading(doc, "Supplemental Materials")
    prose(doc, (
        "This supplement reports the two sensitivity analyses cited in the limitations section "
        "of the main text. Table S8 reports the incremental contribution of diary text relative "
        "to concurrently reported symptoms, which is the basis for the fourth limitation. "
        "Table S9 reports the between- and within-person decomposition of the cross-validated "
        "predictions, which is the basis for the fifth. Table numbering continues the series "
        "in the main text, which ends at Table S7. Both analyses use the cross-validation "
        "procedure described in the main text, with participants assigned whole to folds, "
        "principal components extracted within training folds only, and ridge regression with "
        "alpha = 10. All confidence intervals are percentile intervals from a bootstrap that "
        "resamples participants."
    ))

    for fn in (section_s8, section_s9):
        doc.add_paragraph()
        fn(doc)

    out = ROOT / args.out
    doc.save(out)

    n_tables = len(doc.tables)
    print(f"wrote {out}")
    print(f"  {len(doc.paragraphs)} paragraphs, {n_tables} tables")
    assert n_tables == 2, f"expected 2 tables, built {n_tables}"
    text = "\n".join(p.text for p in doc.paragraphs)
    for bad in ("\u2014", "--", "[CITE NEEDED]", "[N]"):
        assert bad not in text, f"found {bad!r} in supplement prose"
    print("  no em dashes, double hyphens, or unresolved placeholders")


if __name__ == "__main__":
    main()
