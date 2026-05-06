# Operations Runbook

## Purpose
This runbook explains how to operate the loan propensity project for a new month.

## Monthly Workflow
1. Collect raw user dataset.
2. Collect raw call dataset.
3. Score users without touching conversion labels.
4. Export T0 call queue and T1 conversion follow-up rankings.
5. When DIY/SFDC outcomes become available, join labels only for validation.
6. Review AUC, precision@K, lift, deciles, and per-call behavior.
7. Decide whether model performance is stable enough for the next cycle.

## Dashboard Command
Run from the repository root:

```powershell
python -m uvicorn src.multi_month_training.raw_prediction_dashboard:app --host 127.0.0.1 --port 8057
```

Open:

```text
http://127.0.0.1:8057
```

## Inputs
### Required For T0
- User data with valid CSD/user identifier fields.

### Required For T1
- User data.
- Call data with `user_id`, call duration, call status, call type, timestamp, flow/language fields where available.

### Optional For Validation
- DIY conversion file.
- SFDC conversion file.
- Or any label file with UID and converted flag.

## Outputs
Typical outputs include:

- T0 call target list.
- T1 post-call conversion ranking.
- Top 100 / Top 500 / Top 1000 priority files.
- Decile performance table.
- Label comparison report.
- Per-call behavior report.
- Power BI-ready package.

## Checks Before Trusting Results
- User row count looks reasonable.
- Duplicate UID count is reviewed.
- Call join rate is reviewed.
- Users with answered calls are reviewed.
- Score distribution is not collapsed into one value.
- Top decile has higher conversion or higher mean score than lower deciles.
- Labels were not used before scoring.

## Common Failure Modes
### Low Call Join Rate
Likely causes:
- `user_id` format changed.
- Call file schema shifted.
- Call data belongs to a different month.

Fix:
- Inspect `user_id` normalization.
- Check leading/trailing spaces.
- Confirm call and user datasets are from the same campaign/month.

### Suspiciously High Accuracy
Likely causes:
- Conversion labels leaked into features.
- Post-outcome fields entered the user snapshot.
- Labels were joined before scoring.

Fix:
- Re-run leakage checklist.
- Confirm scoring happens before label join.
- Remove status/stage/conversion-like columns from features.

### Weak T0
Expected to be weaker than T1.

Reason:
- T0 only knows pre-call user data.
- T1 has real call behavior, which is much stronger.

Use T0 as a call-priority helper, not a final conversion predictor.

## Business Interpretation
- T0 answers: **who should we call first?**
- T1 answers: **after calls, who is most likely to take the loan?**
- Per-call analysis answers: **how do call count, answered calls, and duration change conversion behavior?**
