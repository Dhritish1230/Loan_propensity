# Per-Call Behavior Findings

## What We Analyzed
April holdout users were grouped by:
- Total number of calls received.
- Number of answered calls.
- Answered-call rate.
- Average call duration.
- Minimum call depth, e.g. users who received at least 1, 2, 3, ... calls.

## Key Numbers
- Users analyzed: 150,513
- Actual conversions: 1,248
- Baseline conversion rate: 0.83%
- Total calls observed: 2,225,829
- Answered calls observed: 561,720

## Best Observed Buckets
- Best total-call bucket with at least 500 users: `8-10` calls, conversion rate 13.19%, lift 15.90x.
- Best average-duration bucket with at least 500 users: `30-59s`, conversion rate 17.93%, lift 21.63x.

## Important Caveat
This analysis is observational. It shows that call behavior is associated with conversion, but it does not prove that simply increasing calls will cause the same conversion lift. Users who receive more calls may differ from users who receive fewer calls. To prove causal impact, run a controlled call-depth experiment.

## Mentor-Friendly Recommendation
Use this analysis to decide a practical call strategy:
1. Prioritize T0 users for first outreach.
2. Track whether calls are answered and whether duration crosses useful engagement thresholds.
3. Use T1 after calls to prioritize follow-up.
4. Monitor whether additional calls after a certain depth are still producing enough incremental conversions per 1,000 calls.
