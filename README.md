# Loan Propensity Modeling and Dashboard Project

This repository contains a portfolio-safe version of a loan propensity analytics project.

## Business Goal
Prioritize customers for outbound loan upselling using a two-stage workflow:

- **T0 Call Targeting:** ranks users before outreach using user snapshot features.
- **T1 Loan Conversion:** ranks post-call users using user and call behavior features.

## Key April Holdout Results

- Users scored: 150,513
- Actual conversions: 1,248
- Baseline conversion rate: 0.83%
- T1 AUC: 0.849
- Precision@100: 53%
- Top-decile lift: 6.65x

## Clean Validation Story
April users were scored first without DIY/SFDC labels. Labels were joined afterward only for validation.

For T1 top 100:

- Same users before vs after labels: true
- Same order before vs after labels: true
- Same scores before vs after labels: true
- Converted users in top 100: 53

## Repository Structure

```text
src/
  multi_month_training/        Core scoring, training, feature engineering, and dashboard backend
  per_call_behavior/           Per-call behavior analysis builder
  april_label_comparison/      Before-label vs after-label validation comparison builder
  powerbi_job_submission/      Power BI package builder
docs/                          Handover, governance, runbook, roadmap, and demo script
reports/                       Portfolio-safe HTML/Markdown summaries
powerbi/                       DAX measures and Power BI build guide
```

## Handover and Operating Docs
These documents are intended to make the project understandable, maintainable, and business-ready:

- [Handover Guide](docs/HANDOVER.md)
- [Operations Runbook](docs/OPERATIONS_RUNBOOK.md)
- [Model Governance](docs/MODEL_GOVERNANCE.md)
- [Roadmap](docs/ROADMAP.md)
- [Five-Minute Demo Script](docs/DEMO_SCRIPT.md)
- [Original Contributor Note](docs/ORIGINAL_CONTRIBUTOR_NOTE.md)

## Business Interpretation
The model is strongest when used as a workflow:

1. Use T0 to prioritize who should be called.
2. Run calls and capture call behavior.
3. Use T1 to prioritize post-call conversion follow-up.
4. Validate later using DIY/SFDC labels.
5. Monitor precision, lift, call join rate, and per-call behavior every month.

## Per-Call Insight
Per-call behavior analysis showed that conversion is driven by useful engagement, not simply more calls.

The best observed signals were:

- Meaningful answered-call duration.
- Moderate call depth before excessive low-quality follow-up.
- Strong post-call T1 score.

This supports a practical call strategy: call enough to create engagement, then use T1 for follow-up prioritization.

## Privacy Note
Raw customer data, original monthly datasets, trained `.pkl` model files, and large prediction outputs are intentionally excluded from git.

## How To Run Dashboard Locally

```powershell
python -m uvicorn src.multi_month_training.raw_prediction_dashboard:app --host 127.0.0.1 --port 8057
```

Then open: http://127.0.0.1:8057

## Requirements
See `requirements.txt`.
