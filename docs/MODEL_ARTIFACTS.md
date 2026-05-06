# Model Artifact Handover

## Final Models

This repository includes the final two-stage model artifacts in `models/`.

| Stage | Business Use | Model File | Feature File |
|---|---|---|---|
| T0 | Pre-call call targeting: who should be called first | `models/loan_model_t0_call_targeting_mixed_hybrid.pkl` | `models/feature_columns_t0_call_targeting_mixed_hybrid.pkl` |
| T1 | Post-call conversion ranking: who is most likely to take the loan after call behavior is available | `models/loan_model_t1_sequence_mixed_hybrid.pkl` | `models/feature_columns_t1_sequence_mixed_hybrid.pkl` |

## Important Separation

T0 and T1 are intentionally separate models.

- T0 uses user snapshot features only.
- T1 uses user features plus call behavior features.
- DIY/SFDC conversion labels are not required for scoring.
- Labels are joined only later for validation and business reporting.

## April Validation Reference

April was scored before labels were used. After labels were joined:

- Users scored: `150,513`
- Actual conversions: `1,248`
- Baseline conversion rate: `0.83%`
- T1 AUC: `0.849`
- Precision@100: `53%`
- Top-decile lift: `6.65x`
- Same T1 top-100 users/order/scores before vs after labels: `true`

## Fresh Machine Checklist

1. Create and activate a Python environment.
2. Install `requirements.txt`.
3. Confirm the four model files exist in `models/`.
4. Run the dashboard.
5. Upload user data and call data.
6. Upload labels only after scoring if validation is needed.

If loading models manually outside the dashboard, set `PYTHONPATH` to the repository `src/` folder first:

```powershell
$env:PYTHONPATH = "$(Resolve-Path .\src)"
```

## Owner Context

Original project contributor: Dhritish Bora.

Preferred contact: dhritishbora.career@gmail.com.
