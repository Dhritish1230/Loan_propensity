# Loan Propensity Power BI Dashboard Build Guide

## Data Source
Use `Loan_Propensity_PowerBI_Source.xlsx` or import the CSV files from this folder.

Recommended import:
1. Open Power BI Desktop.
2. Get Data > Excel Workbook.
3. Select `Loan_Propensity_PowerBI_Source.xlsx`.
4. Load these sheets: `FactScores`, `KPIs`, `Deciles`, `DecisionCurve`, `Segment_State`, `Segment_Campaign`, `PrioritySplit`, `SourceSplit`, `DataQuality`.
5. Add the measures from `PowerBI_Measures.dax`.

## Page 1: Executive Overview
- KPI cards: Total Users, Actual Conversions, Baseline Conversion Rate, T1 AUC, Precision @ 100, Top Decile Lift.
- Column chart: DecisionCurve by Top K, values Actual Conversions and Expected Conversions.
- Column chart: Deciles where Score Column = `t1_loan_conversion_score` and Label Column = `converted_full`.
- Donut chart: PrioritySplit by Priority Band.
- Table: top users from FactScores sorted by T1 Rank.

## Page 2: T0/T1 Workflow
- T0 panel: Average T0 Score, T0 Rank, call planning explanation.
- T1 panel: Average T1 Score, T1 Rank, post-call conversion follow-up explanation.
- Funnel or bar: Total Users > Users With Calls > Users With Answered Calls > Actual Conversions.

## Page 3: Validation
- KPI cards: AUC, Precision @ 100, Precision @ 500, Precision @ 1000, Recall @ 1000.
- Line/column combo: Top K vs Precision and Recall using DecisionCurve.
- Matrix: Deciles with Users, Conversions, Conversion Rate, Mean Score.

## Page 4: Segments
- Bar chart: Segment_State by Avg_T1_Score and Users.
- Bar chart: Segment_Campaign by Avg_T1_Score and Users.
- Slicers: State, Campaign, Priority Band, Call Quality Band.

## Design Notes
- Keep the first page decision-first and uncluttered.
- Use cards for headline metrics.
- Use bar/column charts for comparisons.
- Avoid showing only top 100 in charts; charts should use the full population.
- Real user IDs were anonymized before export.
