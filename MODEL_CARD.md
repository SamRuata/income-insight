# Model Card — Income-Insight

**Model:** `baseline` — feed-forward neural network (MLP), 2 hidden layers
**Owner:** Lal Ruata, CST-435
**Served at:** https://income-insight-api-llvr.onrender.com
**Last updated:** October 2026

---

## Model details

| | |
|---|---|
| Architecture | MLP: 12 features → one-hot/scaled (≈90 dims) → 64 → 32 → 1 logit |
| Activation | ReLU on hidden layers; sigmoid applied to the output logit |
| Loss | `BCEWithLogitsLoss` with `pos_weight` ≈ 3.2 to offset class imbalance |
| Optimizer | Adam, learning rate 0.001, no weight decay |
| Batch size | 256 |
| Epochs | 40 budgeted, early stopping on validation loss (patience 5); stopped at epoch 13 |
| Framework | PyTorch, with a scikit-learn `ColumnTransformer` for preprocessing |
| Checkpoint | `models/baseline.joblib` — weights and fitted preprocessor stored together |

## Intended use

**Primary:** an exploratory tool for analysts at a workforce-policy nonprofit to
examine how a neural network classifies income outcomes from US Census
features, and to audit that behaviour across demographic groups.

**Out of scope.** This model must not be used to make or inform decisions about
any individual — lending, hiring, eligibility, pricing, or screening. It is
trained on 1994 Census data, its errors are unevenly distributed across sex and
race (see below), and it predicts a *statistical association* with income, not
a person's worth, capability, or future earnings.

## Data

- **Source:** UCI Adult Income (1994 US Census extract), 32,561 rows
- **Split:** 80% train / 20% test, assigned once at load time with a fixed seed; a further 10% of the training rows are held out for early stopping
- **Target:** `income > $50K`, positive rate 24.1%
- **Features (12):** age, education_num, capital_gain, capital_loss, hours_per_week, workclass, marital_status, occupation, relationship, race, sex, native_country
- **Dropped:** `fnlwgt` (a census sampling weight, not a property of the person) and `education` (a text duplicate of `education_num`)
- **Missing values:** 1,836 in `workclass`, 1,843 in `occupation`, 583 in `native_country`, loaded as NULL and imputed inside the pipeline so the imputation statistics are learned from training rows only

## Performance

Held-out test split, 6,512 rows never seen during training:

| Metric | Value |
|---|---|
| Accuracy | 0.8076 |
| Precision | 0.5661 |
| Recall | 0.8670 |
| F1 | 0.6849 |
| ROC-AUC | 0.9156 |

**Accuracy is a misleading headline here.** Predicting "≤50K" for everyone scores
75.9% on this data. The model's 80.8% is only five points above that floor;
ROC-AUC is the more honest summary of what it has learned.

The `pos_weight` setting deliberately trades precision for recall: the model
catches 87% of true high earners at the cost of being wrong on 43% of its
positive predictions. That is the right trade for an exploratory tool and the
wrong one for anything that attaches a consequence to a positive flag.

## Fairness

Audited over 2,500 held-out predictions with known ground truth, computed in
Postgres by `audit_by_attribute()`:

| Group | n | False positive rate | False negative rate | Predicted >50K | Accuracy |
|---|---|---|---|---|---|
| Male | 1,719 | 30.7% | **12.0%** | 47.6% | 74.9% |
| Female | 781 | 6.4% | **25.8%** | 14.5% | 91.3% |

| Group | n | FPR | FNR |
|---|---|---|---|
| White | 2,090 | 24.0% | 13.3% |
| Black | 301 | 13.6% | 23.8% |
| Asian-Pac-Islander | 64 | 10.6% | 29.4% |
| Amer-Indian-Eskimo | 34 | 15.6% | 0.0% |
| Other | 11 | 0.0% | — |

The model misses genuinely high-earning women at **2.2×** the rate it misses
high-earning men, and high-earning Black respondents at **1.8×** the rate of
White respondents. The groups with the worse false-negative rate have the
*higher* accuracy, because the model rarely predicts ">50K" for them at all.
Groups below ~100 rows are too small to read.

**Where the disparity comes from.** Permutation importance ranks `sex` 8th of 12
(ROC-AUC drop 0.0064) and `race` last (0.0008). The gap is carried by
`marital_status` — the strongest feature in the model at 0.0569 — and
`relationship`, whose categories include "Husband" and "Wife". Removing `sex`
and `race` as inputs would barely move the model and would remove the ability
to measure the gap.

## Limitations

- **The data is from 1994.** Labour-force participation and the pay gap have both shifted since. Nothing here describes the present-day population.
- **The threshold is arbitrary.** 0.5 is a default, not a decision. Moving it trades the two error types against each other, and the right point depends on which error is more costly — a judgment the model cannot make.
- **Probabilities are approximate.** See the calibration plot in the UI; a score of 0.7 does not reliably mean a 70% chance.
- **Predictions on hand-entered records are not auditable.** Only rows with known ground truth enter the fairness metrics.
- **Historical labels, not ground truth about merit.** The model learns who *did* earn more in 1994, including every structural reason that was so.

## Ethical considerations

The bias audit is deliberately a first-class tab in the UI rather than an
appendix, and the disparity is stated in the product itself rather than only
in this document. A user who reads only the Model Performance tab would see
80.8% accuracy and conclude the model works; the Bias Audit tab is what stops
that reading from being the last word.

Inputs are hashed before logging. No raw identifying record is stored in the
prediction log.