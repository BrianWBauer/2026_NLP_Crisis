# 2026_NLP_Crisis

**Daily Language and Real-Time Suicide Crisis Symptoms**

Analysis code for a study examining whether the semantic content of brief, open-ended
daily diary entries carries information about concurrent and near-term suicide crisis
symptoms in an ecological momentary assessment (EMA) design.

Bauer, B.W., Follet, L., Sappenfield, C., Clark, A., Cecchi, G., & Norel, R.

## Overview

Participants completed repeated daily EMA surveys that combined structured symptom
measures with a short open-ended diary prompt. This repository contains the code that
turns those diary entries into semantic features and relates them to crisis symptom
severity and suicide risk, both on the same occasion and prospectively.

**Research question.** Can the semantic content of daily open-ended text detect suicide
crisis states within persons, and does deviation from a person's own linguistic baseline
add predictive value beyond content itself?

**Why it matters.** Scalable, low-burden methods for detecting real-time suicide crisis
states are limited. Whether passive language monitoring should prioritize *what people
say* or *how far they have moved from their own baseline* is a design decision that
affects how digital phenotyping systems are built.

**Approach.** The analyses separate within-person (state-like, day-to-day deviation from
a participant's own mean) from between-person (trait-like, participant mean) semantic
associations in repeated daily text, and evaluate both concurrent and next-day prediction.

## Methods summary

- **Text features.** Evening/PM diary text per person-day, encoded with
  `sentence-transformers/all-mpnet-base-v2` (768-dimensional, L2-normalized sentence
  embeddings).
- **Dimension reduction.** Principal components of the embedding matrix. For prediction
  models, PCA is fit inside each cross-validation training fold and applied to held-out
  participants; a smaller set of components is retained for the inferential models.
- **Prediction models.** Ridge regression under participant-grouped k-fold
  cross-validation, so no participant contributes to both training and test data.
  Performance is reported from pooled out-of-fold predictions.
- **Inferential models.** Negative binomial GLMs with a log link for the overdispersed
  crisis symptom count, using within-/between-person decomposition of each component and
  participant-clustered robust standard errors, plus direct contextual-effect contrasts
  (between-person minus within-person coefficients).
- **Prospective outcomes.** Diary entries are linked to the following morning's
  assessment by an exact `date + 1 day` join on participant and calendar date.
  Person-days without an assessment on the immediately following calendar day are
  excluded rather than matched to the next available observation.

## Repository structure

```
notebooks/      Analysis pipeline, run in order (01 -> 09)
  01_data_cleaning        Build the person-day analytic table from raw EMA exports
  02_embeddings           Encode diary text; cache embeddings and a row map
  03_feature_engineering  PCA, within-/between-person decomposition, derived features
  04*_...                 Prediction models and text-only benchmarks
  05_ablation_analysis    Nested model comparisons
  06*_, 07_, 09_          Interpretation, supplementary evaluation, correlations
  exploratory/            Earlier exploratory work; superseded (see note below)
scripts/        Standalone analysis scripts (see below)
outputs/        Generated tables and figures (results are not tracked)
data/           Local protected data (not tracked; see Data availability)
```

## Scripts

**`scripts/verify_text_only_aim1.py`** — independent verification of the four text-only
primary results (crisis symptoms and DSI-SS, each concurrent and prospective) under a
single explicit specification. Correctness conditions are asserted rather than assumed:
embedding row alignment, L2-normalization, exact next-calendar-day matching, absence of
participant leakage across folds, and complete out-of-fold coverage. Also rebuilds the
DSI-SS total from raw items and checks it against the stored column.

```bash
python scripts/verify_text_only_aim1.py
python scripts/verify_text_only_aim1.py --verify-embeddings   # confirm embedding provenance
```

**`scripts/final_requested_analyses.py`** — contextual-effect contrasts
(`beta_between - beta_within`) for the retained components using the primary negative
binomial model and participant-clustered covariance, plus a VADER compound sentiment
baseline as (a) a standalone grouped-CV Ridge model and (b) within-/between-person
covariates added to the primary model.

```bash
python scripts/final_requested_analyses.py
```

Both scripts read the local protected analytic file and write only derived result tables
to `outputs/tables/`. Neither writes diary text or participant-level source data.

### A note on the exploratory notebooks

`notebooks/exploratory/` and some of the earlier `04*` notebooks contain prior model
designs that differ from the final specification — for example, text aggregated across
the whole day rather than the evening window, broader day-level samples, different
regularization settings, and next-day outcomes constructed with a positional shift rather
than an exact calendar-date match. Their cached outputs are retained for provenance and
should not be read as the reported results. `scripts/verify_text_only_aim1.py` is the
authoritative implementation of the primary text-only analyses.

## Data availability

**The data are not in this repository and cannot be distributed with it.** The analytic
files contain participant-level EMA responses and open-ended diary text concerning
suicidal ideation and crisis states. Free-text disclosures of this kind are inherently
re-identifiable, so raw text is not shared in any form.

The repository's `.gitignore` excludes all `.csv` and `.npy` files, which covers both the
raw exports and every derived analytic file and result table.

Both scripts expect the protected files at:

```
data/raw/         EMA export
data/processed/   Person-day analytic table, cached embeddings, embedding row map
```

Without them, each script exits with an explicit "file not found" message. Researchers
seeking access for verification or replication should contact the corresponding author;
access requires an appropriate data use agreement and IRB approval.

## Setup

```bash
bash setup.sh          # macOS / Linux
setup.bat              # Windows
conda activate nlp_ema
```

This creates the `nlp_ema` conda environment from `environment.yml` (Python 3.10, pandas,
numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn, jupyter, plus
`transformers`, `sentence-transformers`, and `vaderSentiment` via pip).

Running `02_embeddings.ipynb` downloads the sentence-transformer model on first use and
caches the resulting embedding matrix, so later notebooks and scripts do not re-encode.

## Reproducibility notes

- Random seeds are fixed in the analysis code.
- Cross-validation is grouped by participant throughout; PCA and scaling are fit on
  training folds only.
- Embeddings are cached alongside a row map (`embedding_rows.csv`) so that alignment
  between the embedding matrix and the analytic table can be verified rather than assumed.
- Exact figures depend on the protected data and are reported in the manuscript.
