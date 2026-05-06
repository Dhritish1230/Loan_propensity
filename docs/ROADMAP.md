# Roadmap

## Phase 1: Business Pilot
Goal: use the model in a controlled workflow with human review.

Deliverables:
- Monthly raw scoring.
- T0 call queue.
- T1 post-call follow-up ranking.
- Dashboard review.
- Label-after-score validation.
- Power BI report for leadership.

## Phase 2: Monitoring Layer
Goal: make the model easier to trust month after month.

Add:
- Automated score distribution checks.
- Data schema validation.
- Call join health alerts.
- Month-over-month segment drift.
- Precision/lift trend tracking.
- Model run history.

## Phase 3: Business Decision Simulator
Goal: connect model scores directly to calling capacity.

Add:
- Daily call capacity input.
- Expected conversions at top K.
- Estimated calls required.
- Campaign/state filters.
- Cost per call and ROI estimates.

## Phase 4: Per-Call Optimization
Goal: improve calling strategy, not just prediction.

Add:
- Call-depth stop rules.
- Answered-call duration thresholds.
- Best time-of-day analysis.
- Callback spacing analysis.
- Agent/campaign-level call performance.

## Phase 5: Controlled Experiment
Goal: prove whether additional calls cause incremental conversion.

Design:
- Randomly assign similar users to different call-depth strategies.
- Compare conversion rates.
- Measure incremental conversion per extra call.
- Use findings to define call policy.

## Phase 6: Production Deployment
Goal: convert the pilot into a stable production tool.

Add:
- API authentication.
- Scheduled scoring jobs.
- Database storage.
- Audit logs.
- Role-based dashboard access.
- CI/CD pipeline.
- Model registry.

## High-Value Next Features
- SHAP or local explanation layer.
- Segment-specific thresholding.
- Campaign-level calibration.
- Confidence bands around expected conversions.
- Automatic Power BI refresh package.
- Business-friendly executive summary generation.
