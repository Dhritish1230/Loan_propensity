# Model Governance

## Model Definitions
### T0: Call Targeting
T0 is a pre-call prioritization model. It should only use user snapshot data available before calling.

T0 should not be judged primarily by loan conversion. Its business role is to improve call allocation.

### T1: Loan Conversion
T1 is a post-call conversion model. It uses user data plus call behavior to rank users by loan-taking likelihood.

T1 is the main model for conversion prioritization.

## Leakage Rules
Never use these as model features:

- DIY/SFDC conversion fields.
- Loan stage/outcome fields.
- Disbursement status.
- Rejection/cancellation status after outcome.
- Any label generated after the prediction date.
- Any column that directly says whether the user converted.

## Safe Validation Pattern
The safe sequence is:

1. Prepare raw user/call data.
2. Score users.
3. Save predictions.
4. Join labels afterward.
5. Evaluate outcomes.

Do not reverse steps 2 and 4.

## Metrics To Monitor
For T1:
- AUC.
- Precision@100.
- Precision@500.
- Precision@1000.
- Top-decile lift.
- Decile monotonicity.
- Score distribution drift.

For T0:
- Call engagement rate by top decile.
- Answered-call rate by top decile.
- Conversion as secondary metric only.

## April Benchmark
The April holdout benchmark is:

- T1 AUC: `0.849`
- Precision@100: `53%`
- Precision@500: `36.2%`
- Precision@1000: `31.5%`
- Top-decile lift: `6.65x`
- Baseline conversion: `0.83%`

## Retraining Triggers
Retrain or investigate if:

- Precision@100 drops by more than 30% vs April benchmark.
- Top-decile lift falls below 3x.
- Call join rate drops sharply.
- New campaign/state distribution is very different.
- Score distribution collapses or shifts heavily.
- Conversion definition changes.

## Business Approval Rule
Before using the model operationally for a new month:

- Confirm input data passed quality checks.
- Confirm no labels were used during scoring.
- Review top 100 manually.
- Confirm dashboard metrics with the business owner.
- Archive the scored file and validation report.

## Known Limitation
Per-call behavior is observational. It shows association between calls and conversion, but not pure causal impact. To prove causality, run a controlled call-depth experiment.
