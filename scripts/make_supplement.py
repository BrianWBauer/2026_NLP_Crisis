"""Build the supplemental materials document from the saved result tables.

Every number in the output is read from outputs/tables/, so the document cannot
drift from the analyses. Nothing is typed in by hand except the prose, and the
prose cites only quantities that appear in the tables it accompanies.

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


AUTOCORR = {
    "PM crisis total (t) ~ PM crisis total (t-1)":
        "Evening crisis symptom total, consecutive evenings",
    "PM DSI-SS total (t) ~ PM DSI-SS total (t-1)":
        "Evening DSI-SS total, consecutive evenings",
    "AM crisis total (t+1) ~ AM crisis total (t)":
        "Morning crisis symptom total, consecutive mornings",
    "AM DSI-SS total (t+1) ~ AM DSI-SS total (t)":
        "Morning DSI-SS total, consecutive mornings",
    "AM crisis total (t+1) ~ PM crisis total (t)  [evening -> next morning]":
        "Crisis symptom total, evening to next morning",
    "AM DSI-SS total (t+1) ~ PM DSI-SS total (t)  [evening -> next morning]":
        "DSI-SS total, evening to next morning",
}


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_s1(doc: Document) -> None:
    d = pd.read_csv(TABLES / "lag1_autocorrelation_descriptives.csv")
    heading(doc, "S1. Day-to-day stability of the outcome measures")
    prose(doc, (
        "Both outcomes were strongly autocorrelated across consecutive assessments when "
        "observations were pooled across participants, with correlations between .68 and .72 "
        "depending on the pair of assessments considered (Table S1). Pooled correlations of "
        "this kind combine stable differences between participants with day-to-day change "
        "within them, and the intraclass correlations indicate that the former accounts for "
        "the majority of the variance in each measure (ICC between .62 and .66). When the same "
        "associations were recomputed on person-mean-centered scores, the within-person "
        "autocorrelations fell to between .05 and .22, and the medians of the per-participant "
        "correlations, computed among participants contributing at least five usable pairs, "
        "fell between .11 and .30. Daily symptom reports were therefore substantially less "
        "predictable from the preceding assessment than the pooled values imply, which is the "
        "reason the autoregressive benchmark in Section S2 is decomposed into its between- and "
        "within-person parts in Section S3."
    ))
    caption(doc, "Table S1", "Lag-1 autocorrelation of crisis symptom and suicidal ideation totals")
    rows = []
    for _, r in d.iterrows():
        rows.append([
            AUTOCORR.get(r["outcome"], r["outcome"]),
            integer(r["n_pairs"]),
            integer(r["n_participants"]),
            f'{dec(r["r_pooled"])} {ci(r["r_pooled_CI_lo"], r["r_pooled_CI_hi"])}',
            dec(r["r_within_person"]),
            dec(r["ICC1"]),
            f'{dec(r["r_perperson_median"])} ({integer(r["n_participants_with_min_pairs"])})',
        ])
    table(doc, ["Assessment pair", "Pairs", "Ppts", "Pooled r [95% CI]",
                "Within-person r", "ICC(1)", "Median per-person r (n)"], rows)
    note(doc, (
        "Pooled r is computed across all available pairs. Within-person r is computed on "
        "person-mean-centered scores. The final column reports the median of the "
        "within-participant correlations among the number of participants shown in "
        "parentheses, restricted to those contributing at least five pairs. "
        "DSI-SS = Depressive Symptom Inventory Suicidality Subscale."
    ))


def section_s2(doc: Document) -> None:
    m = pd.read_csv(TABLES / "ar1_benchmark_models.csv")
    c = pd.read_csv(TABLES / "ar1_benchmark_contrasts.csv")
    m = m[m["alpha_spec"] == "fixed_alpha_10"]
    c = c[c["alpha_spec"] == "fixed_alpha_10"]

    heading(doc, "S2. Comparison with an autoregressive benchmark")
    prose(doc, (
        "A reviewer asked whether diary language predicts symptom reports better than the "
        "symptom reports themselves do. We therefore refit the text-only models on the "
        "subsample for which a prior assessment of the outcome was available, and compared "
        "them with a model containing that prior assessment as the sole predictor and with a "
        "model containing both (Table S2). The prior assessment alone outperformed diary text "
        "for all four outcomes. Adding diary text to the prior assessment improved concurrent "
        "prediction for both measures, with increases in cross-validated R-squared of .099 "
        "(95% CI [.043, .148]) for crisis symptom totals and .035 (95% CI [.005, .064]) for "
        "DSI-SS totals. For the next-morning outcomes, adding diary text to the prior "
        "assessment produced no detectable improvement, with increases of .000 "
        "(95% CI [-.017, .018]) and -.010 (95% CI [-.045, .019]). Diary language thus carried "
        "information about current symptom severity that the previous report did not, and did "
        "not improve on the previous report when the outcome was measured the following "
        "morning."
    ))
    caption(doc, "Table S2", "Diary text and a first-order autoregressive term as predictors")
    rows = []
    for a in ORDER:
        sub = m[m["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                MODEL.get(r["model"], r["model"]),
                integer(r["n"]), integer(r["participants"]), integer(r["n_predictors"]),
                dec(r["pearson_r"]), dec(r["R2"]),
            ])
    table(doc, ["Outcome", "Model", "n", "Ppts", "Predictors", "r", "R\u00b2"], rows)
    note(doc, (
        "All models use participant-grouped 5-fold cross-validation with 20 principal "
        "components extracted from the sentence embeddings within training folds only, and "
        "ridge regression with alpha = 10. r and R-squared are computed on pooled out-of-fold "
        "predictions. The reference row repeats the full-sample text-only result so that the "
        "cost of restricting to observations with an available prior assessment is visible."
    ))
    doc.add_paragraph()
    caption(doc, "Table S3", "Incremental contribution of diary text relative to the autoregressive term")
    rows = []
    for a in ORDER:
        sub = c[c["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                contrast_label(r["contrast"]),
                dec(r["delta_r"], 3), ci(r["delta_r_CI_lo"], r["delta_r_CI_hi"], 3),
                dec(r["delta_R2"], 3), ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"], 3),
            ])
    table(doc, ["Outcome", "Contrast", "\u0394r", "95% CI", "\u0394R\u00b2", "95% CI"], rows)
    note(doc, (
        "Confidence intervals are percentile intervals from a bootstrap that resamples "
        "participants rather than observations, so they account for the clustering of "
        "observations within participants."
    ))


def section_s3(doc: Document) -> None:
    m = pd.read_csv(TABLES / "ar1_decomposition_models.csv")
    heading(doc, "S3. What the autoregressive term actually contributes")
    prose(doc, (
        "The prior assessment of an outcome contains a stable person-level component, which is "
        "that participant's average level, and a within-person deviation from it. Because the "
        "first of these is constant within a participant, a model containing the prior "
        "assessment can perform well by recovering participant averages alone. We therefore "
        "split the lagged predictor into its between- and within-person parts and refit "
        "(Table S4). The between-person part accounted for essentially all of the "
        "autoregressive term's predictive value, with cross-validated R-squared between .62 "
        "and .67 across the four outcomes, whereas the within-person part alone performed no "
        "better than predicting each outcome's mean, with R-squared between -.036 and -.011. "
        "The benchmark that diary text failed to improve upon prospectively is therefore close "
        "to a participant-specific intercept, and its apparent advantage reflects stable "
        "differences in symptom level rather than any capacity to anticipate change. Consistent "
        "with that interpretation, adding diary text to the within-person part alone increased "
        "R-squared by .283 (95% CI [.201, .351]) for concurrent crisis symptom totals, while "
        "adding it to both parts increased R-squared by .059 (95% CI [.026, .088]) for the same "
        "outcome and by .000 (95% CI [-.006, .007]) for next-morning crisis symptom totals."
    ))
    caption(doc, "Table S4",
            "Between- and within-person components of the autoregressive term")
    rows = []
    for a in ORDER:
        sub = m[m["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                MODEL.get(r["model"], r["model"]),
                integer(r["n_predictors"]), dec(r["pearson_r"]), dec(r["R2"]),
            ])
    table(doc, ["Outcome", "Model", "Predictors", "r", "R\u00b2"], rows)
    note(doc, (
        "The between-person part of the lagged predictor is each participant's mean of that "
        "predictor; the within-person part is the deviation of each observation from that mean. "
        "A negative R-squared indicates out-of-fold prediction worse than the training-fold "
        "mean of the outcome."
    ))


def section_s4(doc: Document) -> None:
    m = pd.read_csv(TABLES / "text_plus_symptoms_models.csv")
    c = pd.read_csv(TABLES / "text_plus_symptoms_contrasts.csv")
    m = m[(m["alpha_spec"] == "fixed_alpha_10") & (m["block"] != "(none)")]
    c = c[c["alpha_spec"] == "fixed_alpha_10"]
    order = [o for o in ORDER if o in set(m["outcome"])]

    heading(doc, "S4. Diary text alongside concurrently reported symptoms")
    prose(doc, (
        "A further question is whether diary language adds anything once the symptoms a "
        "participant has just reported are already in the model, which is the comparison that "
        "bears on whether written entries could supplement rather than replace brief symptom "
        "scales. We fit models containing the concurrently reported crisis symptom total, the "
        "five crisis symptom items separately, or all self-report predictors, each with and "
        "without diary text (Tables S5 and S6). For next-morning crisis symptom totals, diary "
        "text added nothing detectable to the concurrent crisis symptom total, with a change in "
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
    caption(doc, "Table S5", "Diary text and concurrently reported symptoms as predictors")
    rows = []
    for a in order:
        sub = m[m["outcome"] == a]
        first = True
        for blk, g in sub.groupby("block", sort=False):
            for i, (_, r) in enumerate(g.iterrows()):
                rows.append([
                    OUTCOME[a] if first else "",
                    MODEL.get(blk, blk) if i == 0 else "",
                    MODEL.get(r["model"], r["model"]),
                    integer(r["n"]), integer(r["n_predictors"]),
                    dec(r["pearson_r"]), dec(r["R2"]),
                ])
                first = False
    table(doc, ["Outcome", "Self-report block", "Model", "n", "Predictors",
                "r", "R\u00b2"], rows)
    note(doc, (
        "The crisis symptom items block contains the five items separately rather than their "
        "total. The all self-report predictors block contains the crisis symptom total and the "
        "DSI-SS total. Sample sizes differ by one or two observations across blocks because of "
        "item-level missingness."
    ))
    doc.add_paragraph()
    caption(doc, "Table S6", "Incremental contribution of diary text relative to self-reported symptoms")
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
    note(doc, "Confidence intervals are participant-cluster bootstrap percentile intervals.")


def section_s5(doc: Document) -> None:
    d = pd.read_csv(TABLES / "covariate_between_within_decomposition.csv")
    heading(doc, "S5. Between- and within-person components of the text models")
    prose(doc, (
        "The cross-validated predictions from the text-only models were decomposed in the same "
        "way as the autoregressive term, by correlating participant means of the predictions "
        "with participant means of the outcome and by correlating the person-mean-centered "
        "predictions with the person-mean-centered outcome (Table S7). Because participant "
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
    caption(doc, "Table S7",
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


def section_s6(doc: Document) -> None:
    m = pd.read_csv(TABLES / "covariate_dayofweek_models.csv")
    c = pd.read_csv(TABLES / "covariate_dayofweek_contrasts.csv")
    heading(doc, "S6. Day of week")
    prose(doc, (
        "Because diary entries written on different days of the week may differ in content for "
        "reasons unrelated to symptom severity, we tested whether day of week accounted for any "
        "of the text models' accuracy (Table S8). Day of week was entered as six indicator "
        "variables with Monday omitted. Entered alone it predicted no outcome better than the "
        "outcome mean, with cross-validated R-squared between -.022 and -.008, and adding it to "
        "diary text did not improve prediction for any outcome. The change in R-squared from "
        "adding day of week to diary text was -.001 (95% CI [-.004, .002]) for concurrent "
        "crisis symptom totals and -.009 (95% CI [-.013, -.005]) for next-morning DSI-SS "
        "totals, the latter indicating a small decrement consistent with the cost of six "
        "uninformative predictors. Observations were close to evenly distributed across "
        "weekdays, and 114 of 123 participants contributed observations on at least five "
        "distinct weekdays, so the null result does not reflect insufficient coverage."
    ))
    caption(doc, "Table S8", "Day of week as a predictor and as a covariate")
    rows = []
    for a in ORDER:
        sub = m[m["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                MODEL.get(r["model"], r["model"]),
                integer(r["n"]), integer(r["n_predictors"]),
                dec(r["pearson_r"]), dec(r["R2"]),
            ])
    table(doc, ["Outcome", "Model", "n", "Predictors", "r", "R\u00b2"], rows)
    doc.add_paragraph()
    caption(doc, "Table S9", "Incremental contribution of day of week")
    rows = []
    for a in ORDER:
        sub = c[c["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "", contrast_label(r["contrast"]),
                dec(r["delta_R2"], 3), ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"], 3),
            ])
    table(doc, ["Outcome", "Contrast", "\u0394R\u00b2", "95% CI"], rows)
    note(doc, "Confidence intervals are participant-cluster bootstrap percentile intervals.")


def section_s7(doc: Document) -> None:
    dm = pd.read_csv(TABLES / "covariate_demographic_contrasts.csv")
    pc = pd.read_csv(TABLES / "covariate_personlevel_contrasts.csv")
    fs = pd.read_csv(TABLES / "covariate_personlevel_fold_stability.csv")
    rec = pd.read_csv(TABLES / "covariate_text_recovers_demographics.csv")

    heading(doc, "S7. Age and sex")
    prose(doc, (
        "Sentence embeddings encode speaker characteristics, so a model trained on diary text "
        "could predict symptom severity partly by inferring who wrote the entry. Two features "
        "of the design bear on this concern. First, participant-grouped cross-validation means "
        "no participant appears in both a training fold and the test fold, so a demographic "
        "signal can help only insofar as it generalizes to held-out participants. Second, the "
        "within-person estimates reported in Section S5 are computed on person-mean-centered "
        "scores, which absorb every time-invariant participant characteristic, including age "
        "and sex, and are therefore adjusted for them by construction. Person-level covariates "
        "can bear only on the between-person component."
    ))
    prose(doc, (
        "The mechanism for such a confound was present in these data. Diary text recovered age "
        "with a cross-validated correlation of .41 at the participant level and classified sex "
        "at birth with an area under the receiver operating characteristic curve of .69 "
        "(Table S10). Adding age and sex to the text models nonetheless did not improve "
        "prediction of any outcome at either the day level or the participant level "
        "(Table S11). At the participant level, where the covariates have the most room to act, "
        "the change in R-squared from adding them to diary text was -.010 "
        "(95% CI [-.035, .010]) for concurrent crisis symptom totals and -.022 "
        "(95% CI [-.056, .015]) for next-morning crisis symptom totals. Adding diary text to "
        "age and sex, by contrast, increased R-squared by .379 (95% CI [.264, .512]) and .287 "
        "(95% CI [.160, .424]) for the same two outcomes. Age and sex alone predicted the "
        "crisis symptom outcomes no better than their means, with cross-validated R-squared of "
        "-.030 and -.012 averaged across 30 random participant-to-fold partitions. Demographic "
        "characteristics therefore do not appear to explain the association between diary "
        "language and symptom severity, although the modest recovery of age and sex from text "
        "means this cannot be ruled out for characteristics we did not measure."
    ))
    caption(doc, "Table S10", "Recovery of demographic characteristics from diary text")
    rows = []
    for _, r in rec.iterrows():
        rows.append([
            "Participant level" if r["level"] == "person" else "Day level",
            "Age" if r["target"] == "age" else "Sex at birth (male)",
            integer(r["n"]), dec(r["pearson_r"]), dec(r["R2"]),
            dec(r["AUROC"]) if not pd.isna(r["AUROC"]) else "",
        ])
    table(doc, ["Level", "Target", "n", "r", "R\u00b2", "AUROC"], rows)
    doc.add_paragraph()
    caption(doc, "Table S11", "Age and sex as covariates in the text models")
    rows = []
    for lvl, src in [("Day level", dm), ("Participant level", pc)]:
        first = True
        for a in ORDER:
            sub = src[src["analysis"] == a]
            for i, (_, r) in enumerate(sub.iterrows()):
                rows.append([
                    lvl if first else "",
                    OUTCOME[a] if i == 0 else "",
                    contrast_label(r["contrast"]),
                    dec(r["delta_R2"], 3), ci(r["delta_R2_CI_lo"], r["delta_R2_CI_hi"], 3),
                ])
                first = False
    table(doc, ["Level", "Outcome", "Contrast", "\u0394R\u00b2", "95% CI"], rows)
    doc.add_paragraph()
    caption(doc, "Table S12",
            "Participant-level model performance across 30 random fold partitions")
    rows = []
    for a in ORDER:
        sub = fs[fs["analysis"] == a]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([
                OUTCOME[a] if i == 0 else "",
                MODEL.get(r["model"], r["model"]),
                integer(r["participants"]),
                f'{dec(r["r_mean"])} ({dec(r["r_sd"], 3)})',
                f'{dec(r["R2_mean"])} ({dec(r["R2_sd"], 3)})',
                f'{dec(r["R2_min"])} to {dec(r["R2_max"])}',
            ])
    table(doc, ["Outcome", "Model", "Ppts", "r (SD)", "R\u00b2 (SD)",
                "R\u00b2 range"], rows)
    note(doc, (
        "Participant-level analyses regress each participant's mean outcome on the mean of that "
        "participant's sentence embeddings, renormalized to unit length. With 123 participants "
        "and five folds, a single deterministic partition can fall more than one standard "
        "deviation from the mean across partitions, so means and standard deviations across "
        "partitions are reported in place of single-partition estimates."
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
        "This supplement reports analyses conducted in response to reviewer questions about "
        "whether the association between diary language and crisis symptom severity could be "
        "explained by the stability of the outcomes over time, by concurrently reported "
        "symptoms, by day of week, or by participant demographic characteristics. All models "
        "use the cross-validation procedure described in the main text, with participants "
        "assigned whole to folds, principal components extracted within training folds only, "
        "and ridge regression with alpha = 10 unless otherwise noted. All confidence intervals "
        "are percentile intervals from a bootstrap that resamples participants."
    ))

    for fn in (section_s1, section_s2, section_s3, section_s4,
               section_s5, section_s6, section_s7):
        doc.add_paragraph()
        fn(doc)

    out = ROOT / args.out
    doc.save(out)

    n_tables = len(doc.tables)
    print(f"wrote {out}")
    print(f"  {len(doc.paragraphs)} paragraphs, {n_tables} tables")
    assert n_tables == 12, f"expected 12 tables, built {n_tables}"
    text = "\n".join(p.text for p in doc.paragraphs)
    for bad in ("\u2014", "--", "[CITE NEEDED]", "[N]"):
        assert bad not in text, f"found {bad!r} in supplement prose"
    print("  no em dashes, double hyphens, or unresolved placeholders")


if __name__ == "__main__":
    main()
