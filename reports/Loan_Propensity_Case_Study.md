# Loan Propensity Analytics Dashboard

## Business Problem
The business needs to prioritize a large customer base for loan upselling. The workflow has two stages:

- **T0 call targeting:** before calling, rank users by who should be contacted first.
- **T1 conversion follow-up:** after call activity exists, rank users by likelihood of taking the loan.

## Data Used
The dashboard package uses an anonymized April holdout dataset with user profile fields, call engagement fields, model scores, and validation labels from DIY/SFDC outcomes. Real user identifiers are removed and replaced with anonymized user IDs.

## Modeling Workflow
- Models were trained on multiple months: October, December, January, February, and March.
- April was kept as a holdout test.
- April was first scored without conversion labels.
- DIY/SFDC labels were joined afterward only for validation.

## Key Results
- Users scored: 150,513
- Actual conversions: 1,248
- Baseline conversion rate: 0.83%
- T1 AUC: 0.849
- Precision@100: 53%
- Top-decile lift: 6.65x

## Dashboard Pages
1. **Executive Overview:** headline KPIs, top-K decision curve, priority bands, deciles.
2. **T0/T1 Workflow:** explains call targeting vs conversion follow-up.
3. **Validation:** AUC, precision@K, recall@K, lift, decile performance.
4. **Segments:** campaign/state breakdown and call-quality analysis.

## Why This Is Useful
The dashboard converts model output into a business decision tool. It answers:

- Who should the calling team prioritize today?
- How many conversions can we expect if we call or follow up with the top K users?
- Which segments have the strongest conversion probability?
- Did call data merge correctly?
- Is the model concentrating conversions in the highest-ranked users?
