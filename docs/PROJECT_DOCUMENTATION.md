# Loan Propensity Project Documentation

Last updated: May 2026

## 1. Executive Summary

This project is a two-stage loan propensity and call-prioritization system for outbound loan upselling.

The business problem is simple: there are many eligible users, but the calling/follow-up team has limited capacity. The project ranks users so the team can spend time on the users most likely to engage and convert.

The final workflow has two separate models:

| Stage | Question Answered | Input Data | Business Use |
|---|---|---|---|
| T0 | Who should we call first? | User snapshot data only | Build the outbound call queue before calls happen |
| T1 | Who is most likely to take the loan after call activity? | User snapshot plus call behavior | Prioritize post-call conversion follow-up |

The strongest validation result came from April. April users were scored first without using DIY/SFDC conversion labels. Labels were joined later only for validation.

April T1 validation benchmark:

- Users scored: `150,513`
- Actual conversions: `1,248`
- Baseline conversion rate: `0.83%`
- T1 AUC: `0.849`
- Precision@100: `53%`
- Top-decile lift: `6.65x`
- Same top-100 users/order/scores before vs after labels: `true`

## 2. Business Context

The model supports a controlled sales/calling workflow:

1. Start with a raw monthly user file.
2. Use T0 to rank users before calling.
3. Run outreach/calls and collect call behavior.
4. Use T1 to rank users after call behavior is available.
5. When DIY/SFDC outcomes become available, validate predictions by joining labels after scoring.
6. Review precision, lift, deciles, call join health, and per-call behavior.

This is not only a model. It is a repeatable monthly decision process for call allocation and conversion follow-up.

## 3. Repository Structure

```text
src/
  multi_month_training/        Training, scoring, validation, and feature-engineering scripts
  per_call_behavior/           Per-call behavior analysis builder
  april_label_comparison/      Before-label vs after-label validation comparison builder
  powerbi_job_submission/      Power BI source package builder
models/                        Final T0/T1 model artifacts and feature columns
data_templates/                Safe fake upload templates
docs/                          Handover, governance, runbook, roadmap, and documentation
reports/                       Portfolio-safe HTML/Markdown summaries
powerbi/                       DAX measures and Power BI guide
```

Important files:

| File | Purpose |
|---|---|
| `src/multi_month_training/raw_prediction_dashboard.py` | FastAPI dashboard for upload-and-score workflow |
| `src/multi_month_training/raw_scoring_pipeline.py` | Main reusable scoring pipeline |
| `src/multi_month_training/train_best_stage_models.py` | Final mixed-hybrid T0/T1 training script |
| `src/hybrid_month_similarity_model.py` | Custom model class used inside final pickles |
| `models/loan_model_t0_call_targeting_mixed_hybrid.pkl` | Final T0 model |
| `models/loan_model_t1_sequence_mixed_hybrid.pkl` | Final T1 model |
| `docs/HANDOVER.md` | Handover summary |
| `docs/OPERATIONS_RUNBOOK.md` | Monthly operating steps |
| `docs/MODEL_GOVERNANCE.md` | Leakage, monitoring, and approval rules |
| `docs/MODEL_ARTIFACTS.md` | Model artifact details |

## 4. Data Used

Training and validation work was built around monthly data from:

- October
- December
- January
- February
- March
- April as a clean holdout/validation month

November was used only where labels were available or for unlabeled testing. It was not treated as a fully labeled conversion-training month when conversion labels were missing.

Raw monthly customer datasets are intentionally not committed to GitHub for privacy and size reasons. The repository contains only safe templates and code.

## 5. Data Inputs

### 5.1 User Data

The user file is required for both T0 and T1.

Common fields used:

- `uid` or CSD ID
- `user_id`
- `campaign_id`
- `state`
- `language`
- `flow_phase`
- `age`
- `decile`
- `minimum_loan_amount`
- `maximum_loan_amount`
- `scheme`
- `zone`
- `base_type`
- `flow_type`
- `created_at`
- `expiry_date`

The pipeline normalizes UID values, removes invalid CSD IDs, deduplicates users, normalizes categorical fields, and creates loan/date-derived fields.

### 5.2 Call Data

The call file is required for useful T1 scoring.

Common fields used:

- `call_id`
- `user_id`
- `call_duration`
- `call_type`
- `call_time`
- `call_endtime`
- `created_at`
- `modified_at`
- `call_status`
- `hangup_cause_code`
- `flow`
- `did`
- `language`

Calls join to users through `user_id`.

### 5.3 Label Data

DIY/SFDC labels are not needed for prediction.

Labels are used only for validation after scoring. Label files should contain a UID/CSD ID and a conversion flag such as:

- `converted_full`
- `converted`
- `actual`
- `label`
- `y`
- `outcome`

Labels join through `uid`.

## 6. Data Cleaning and Merging

The scoring pipeline performs these steps:

1. Read CSV/XLSX user data.
2. Standardize column names and identify UID/user ID columns.
3. Normalize UID values to uppercase.
4. Keep valid `CSD-...` users.
5. Deduplicate at UID level.
6. Normalize categorical fields.
7. Convert numeric fields.
8. Create loan amount features.
9. Create date features.
10. Read call data in chunks for large files.
11. Normalize call columns.
12. Join call aggregates to users by `user_id`.
13. Fill missing call features with zero.
14. Score with T0 and T1 models.
15. Join labels only after scoring, if validation labels are provided.

Key merge rule:

- User-to-call merge uses `user_id`.
- Prediction-to-label validation merge uses `uid`.

## 7. Feature Engineering

### 7.1 T0 Features

T0 is pre-call and should not use call behavior.

Main T0 feature families:

- User categorical fields: language, state, campaign, flow, scheme, zone, base type, flow type
- User numeric fields: age, decile, loan amount range, created date fields, expiry flags

T0 target:

- `engaged_10s`
- This means the user showed meaningful call engagement, defined around answered call behavior and average duration.

Business interpretation:

- T0 is not the final loan-conversion model.
- T0 is used to build a better call queue before call behavior exists.

### 7.2 T1 Features

T1 is post-call and uses user plus call behavior.

Main T1 feature families:

- User snapshot features
- Total calls
- Answered calls
- Answered rate
- Average call duration
- Raw call status/type/flow/language aggregates
- Duration summaries
- Sequence/timing features such as first call hour, last call hour, call span, answered span, business-hour share, weekend share
- Engineered interaction features such as answered share multiplied by duration

T1 target:

- `converted`

Business interpretation:

- T1 is the main conversion model.
- It ranks users most likely to take the loan after call behavior exists.

## 8. Final Model Architecture

The final production-style models are separate T0 and T1 models.

Both use a mixed hybrid structure:

- Global model trained on combined multi-month data.
- Month-specific models trained month by month.
- Final score blends the global prediction with month-specific weighted predictions.
- Month-specific weighting is based on similarity between the new batch profile and historical month profiles.

Final training configuration:

| Stage | Target | Global Model | Monthly Model | Month Blend Weight |
|---|---|---|---|---|
| T0 | `engaged_10s` | ExtraTrees | Linear model | `0.35` |
| T1 | `converted` | ExtraTrees | Linear model | `0.35` |

Interpretation:

- The global model captures broad behavior across months.
- The monthly models capture month-specific trends.
- The similarity layer chooses how much each historical month should influence a new batch.
- This gives one deployable T0 model and one deployable T1 model, while still preserving month-wise trend learning.

## 9. Model Artifacts

The final uploaded artifacts are:

```text
models/
  loan_model_t0_call_targeting_mixed_hybrid.pkl
  feature_columns_t0_call_targeting_mixed_hybrid.pkl
  loan_model_t1_sequence_mixed_hybrid.pkl
  feature_columns_t1_sequence_mixed_hybrid.pkl
```

The pickle files require repository code because they reference the custom class in:

```text
src/hybrid_month_similarity_model.py
```

When loading manually, set:

```powershell
$env:PYTHONPATH = "$(Resolve-Path .\src)"
```

The dashboard already handles the expected import path when run from the repository root.

## 10. Dashboard

The local dashboard is a FastAPI app.

Run:

```powershell
python -m uvicorn src.multi_month_training.raw_prediction_dashboard:app --host 127.0.0.1 --port 8057
```

Open:

```text
http://127.0.0.1:8057
```

Dashboard capabilities:

- Upload raw user data.
- Upload raw call data.
- Optionally upload labels for validation.
- Score T0 and T1.
- Show all-user dashboard metrics.
- Show T0/T1 deciles.
- Show priority bands.
- Show call join health.
- Show top-ranked users.
- Download full ranked predictions.
- Download top T0 and top T1 lists.

## 11. Clean April Validation

April was used as the main proof that the scoring process was clean.

Validation sequence:

1. Score April user/call data without DIY/SFDC labels.
2. Save predictions and rankings.
3. Later join DIY/SFDC labels.
4. Compare before-label and after-label prediction files.
5. Confirm that the ranking and scores did not change.
6. Compute actual conversion metrics.

April result:

- Users scored: `150,513`
- Actual conversions: `1,248`
- Baseline conversion rate: `0.83%`
- T1 AUC: `0.849`
- Precision@20: `65%`
- Precision@50: `74%`
- Precision@100: `53%`
- Precision@500: `36.2%`
- Precision@1000: `31.5%`
- Top-decile lift: `6.65x`

Top-100 proof:

- Same top-100 users before and after labels: `true`
- Same top-100 order before and after labels: `true`
- Same top-100 scores before and after labels: `true`
- Converted users in top 100: `53`
- DIY conversions in top 100: `7`
- SFDC conversions in top 100: `46`

This is the strongest validation story for the project.

## 12. Leakage Controls

The project separates prediction from validation.

Rules:

- Do not use DIY/SFDC labels before prediction.
- Do not train on or score with direct conversion/outcome columns.
- Do not use post-outcome fields as features.
- Join labels only after predictions are saved.
- Check that before-label and after-label scores/ranks are unchanged.

Forbidden feature types:

- Conversion flags
- Loan outcome/status
- Disbursement status
- Rejection/cancellation status after outcome
- SFDC/DIY outcome fields
- Any label generated after the prediction date

## 13. Per-Call Behavior Analysis

The project includes per-call behavior analysis to understand how conversion changes with call count, answered calls, and duration.

Key finding:

- Better conversion is associated with useful engagement, not simply more calls.

Observed patterns:

- Meaningful answered-call duration was a strong signal.
- Moderate call depth performed better than endless low-quality calling.
- Average duration around `30-59s` was one of the strongest observed buckets.
- A strong T1 score plus answered-call behavior is a better follow-up signal than raw call count alone.

Important limitation:

- This analysis is observational.
- It shows association, not guaranteed causation.
- To prove causal impact, run a controlled call-depth experiment.

## 14. Power BI and Reporting

The project includes a Power BI-ready source package and DAX measures.

Important files:

```text
src/powerbi_job_submission/build_powerbi_package.py
powerbi/PowerBI_Measures.dax
powerbi/PowerBI_Report_Build_Guide.md
```

The Power BI package is meant for supervisor/business presentation, not for model scoring.

## 15. Monthly Operating Process

For a new month:

1. Place the raw user and call data in the company workspace.
2. Run the dashboard.
3. Upload user data.
4. Upload call data.
5. Do not upload labels if the goal is clean prediction.
6. Download full predictions, top T0 list, and top T1 list.
7. Use T0 for call planning and T1 for post-call follow-up.
8. When DIY/SFDC labels arrive, upload/join labels for validation only.
9. Review AUC, precision@K, lift, deciles, score distribution, and call join rate.
10. Archive outputs and document whether the model remains stable.

## 16. Monitoring Metrics

Track these every month:

| Metric | Why It Matters |
|---|---|
| User row count | Catches missing/duplicated source data |
| Valid UID count | Confirms user cleaning worked |
| Call join rate | Confirms user/call merge worked |
| Users with answered calls | Confirms useful call behavior is present |
| Score distribution | Catches scoring drift/collapse |
| T1 AUC | Overall ranking quality when labels are available |
| Precision@100 | Business usefulness for limited follow-up capacity |
| Precision@500 / Precision@1000 | Wider operational usefulness |
| Top-decile lift | Segment-level concentration of conversions |
| Decile monotonicity | Confirms higher scores generally mean better outcomes |

## 17. Production Readiness

Current status:

- Business-pilot ready.
- Suitable for controlled monthly scoring, dashboard review, and supervised decision-making.
- Not yet a fully automated production service.

Before full production, add:

- Scheduled monthly scoring jobs.
- Data contract validation.
- Automated drift monitoring.
- Model registry/versioning.
- Access control and authentication.
- Audit logs.
- Deployment pipeline.
- Rollback process.

## 18. Common Failure Modes

### Low Call Join Rate

Likely causes:

- `user_id` format changed.
- Call data belongs to a different month.
- Column order/schema changed.
- Leading/trailing spaces or missing IDs.

Fix:

- Inspect `user_id` normalization.
- Confirm user and call files are from the same period.
- Check dashboard call join health.

### Suspiciously High Accuracy

Likely causes:

- Leakage from conversion labels.
- Labels joined before scoring.
- Outcome/status columns entered the feature set.

Fix:

- Re-run the leakage checklist.
- Confirm score files were generated before labels were joined.
- Compare before-label and after-label ranking/scores.

### Weak T0

Expected behavior:

- T0 is naturally weaker than T1 because it only has user data.

Use T0 for:

- Call prioritization.
- Engagement targeting.

Do not use T0 as:

- The final loan-conversion predictor.

## 19. How To Continue The Project

Immediate continuation steps:

1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Run the dashboard locally.
4. Use `data_templates/` to understand expected upload format.
5. Score a new month without labels.
6. Validate later when labels arrive.
7. Compare new results against the April benchmark.

Retraining continuation:

1. Make sure raw monthly data is available in the company workspace.
2. Set `LOAN_PROPENSITY_WORKSPACE` if the data folder is outside the repo.
3. Set `LOAN_PROPENSITY_MODEL_DIR` if model artifacts should be saved elsewhere.
4. Run the training scripts under `src/multi_month_training/`.
5. Validate with a holdout month.
6. Replace model artifacts only after validation passes governance checks.

Example environment variables:

```powershell
$env:LOAN_PROPENSITY_WORKSPACE = "D:\LoanPropensityWorkspace"
$env:LOAN_PROPENSITY_MODEL_DIR = "D:\LoanPropensityWorkspace\models"
```

## 20. Ownership and Handover

Original project contributor:

- Dhritish Bora
- dhritishbora.career@gmail.com

Recommended handover message:

"This project contains the final T0/T1 model artifacts, scoring dashboard, training scripts, validation approach, model governance, and operating runbook. Raw customer data is intentionally excluded. To continue the project, clone the repo, install dependencies, run the dashboard, score the new month without labels, and validate later using DIY/SFDC outcomes."

