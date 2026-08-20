# 2026_NLP_Crisis

Daily Language Predicts Real-Time Suicide Crisis Symptoms

Target journals: Journal of Affective Disorders (primary); Suicide and Life-Threatening Behavior or JMIR Mental Health (alternates)

Author List: Bauer, B.W., Follet., L., Sappenfield., C., Clark., A., Cecchi., G., & Norel., R.

Project Snapshot:

Core Research Question: Can the semantic content of daily open-ended text entries detect same-day suicide crisis states within persons, and does deviation from one's linguistic baseline add predictive value beyond content?

What Problem is it Solving: We lack scalable, low-burden methods for detecting real-time suicide crisis states, and we don't know whether passive language monitoring should focus on what people say or on detecting linguistic change from their baseline.

Why Does it Matter: If content alone carries the signal, detection systems should prioritize semantic analysis over person-specific deviation tracking - a design decision that affects every digital phenotyping platform being built right now.

What is Novel: The study separates within-person and between-person semantic associations in repeated daily text and evaluates whether ordinary diary language contains concurrent and prospective crisis-relevant signal.

## Final requested analyses

The manuscript's final two requested analyses are implemented in `scripts/final_requested_analyses.py`:

1. Direct contextual-effect contrasts (`beta_between - beta_within`) for PC1-PC5 using the primary negative-binomial model and participant-clustered robust covariance matrix.
2. VADER compound sentiment as (a) a standalone Ridge/GroupKFold baseline and (b) within- and between-person covariates in the primary negative-binomial model.

Run from the project environment after the protected local analytic file is available at `data/processed/pm_day_features.csv`:

```bash
python scripts/final_requested_analyses.py
```

The script writes only derived result tables to `outputs/tables/`; it does not write raw diary text or participant-level source data to the repository.
