# Handover Guide

## Project Summary
This project is a two-stage loan propensity system for outbound upselling:

- **T0 call targeting:** ranks users before outreach using user snapshot data.
- **T1 loan conversion:** ranks users after calls using user data plus call behavior.

The project was designed to answer a practical business question:

> If the calling team has limited capacity, which users should be contacted or followed up first?

## What Was Delivered
- Multi-month model training workflow.
- Raw user + call data scoring pipeline.
- Local FastAPI dashboard for upload-and-score workflows.
- April holdout validation workflow.
- Supervisor comparison of unlabeled predictions vs later validation labels.
- Per-call behavior analysis.
- Power BI-ready package and DAX measures.

## Key April Holdout Results
- Users scored: `150,513`
- Actual conversions: `1,248`
- Baseline conversion rate: `0.83%`
- T1 AUC: `0.849`
- Precision@100: `53%`
- Top-decile lift: `6.65x`

## Clean Validation Principle
April was scored first without DIY/SFDC labels. Conversion labels were joined only after scoring, strictly for validation.

For the T1 top 100:
- Same users before vs after labels: `true`
- Same order before vs after labels: `true`
- Same scores before vs after labels: `true`
- Actual conversions after labels: `53/100`

This is the core proof that prediction and validation were separated.

## Files To Know
- `src/multi_month_training/raw_scoring_pipeline.py`: reusable raw scoring pipeline.
- `src/multi_month_training/raw_prediction_dashboard.py`: local dashboard backend and UI.
- `src/multi_month_training/score_april_unlabeled.py`: clean April unlabeled scoring workflow.
- `src/multi_month_training/evaluate_april_with_labels.py`: post-label validation workflow.
- `src/per_call_behavior/build_per_call_behavior_analysis.py`: per-call behavior analysis.
- `src/april_label_comparison/build_supervisor_label_comparison.py`: before-label vs after-label supervisor comparison.
- `src/powerbi_job_submission/build_powerbi_package.py`: Power BI-ready package builder.
- `models/`: final portable T0/T1 model artifacts and feature column lists.
- `data_templates/`: safe upload templates for user data, call data, and optional labels.

## What Is Excluded From Git
For privacy and size reasons, this repository intentionally excludes:

- Raw customer/monthly data.
- Large scored prediction files.
- Power BI `.pbix` files.
- Excel/CSV output exports.

The final `.pkl` model artifacts required to run the handover dashboard are included in `models/`.

## Production Readiness Position
The current system is **business-pilot ready**, not fully autonomous production ready.

Use it for:
- Controlled monthly scoring.
- Prioritized call/follow-up lists.
- Business validation.
- Dashboard-based review.

Before full production, add:
- Automated monthly monitoring.
- Retraining rules.
- Data contract checks.
- Access controls.
- Audit logging.
- Deployment pipeline.

## When To Contact The Original Builder
The original builder has important context on:

- Why T0 and T1 are separated.
- Why April was validated label-after-score.
- How raw call joins were fixed and verified.
- Which features are safe vs leakage-prone.
- How to explain top-K precision, lift, and per-call behavior to business teams.

If the data schema changes, if model performance drops, or if the business wants to change the target definition, contact the original contributor before making major changes.
