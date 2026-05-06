import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\m\Desktop\PROJECT INTERNSHIP")
OUT = ROOT / "powerbi_job_submission"
SOURCE = ROOT / "multi_month_training" / "test_outputs" / "april_t1_evaluated_predictions.csv"
REPORT = ROOT / "multi_month_training" / "test_outputs" / "april_labeled_evaluation_report.json"
DECILES = ROOT / "multi_month_training" / "test_outputs" / "april_labeled_decile_tables.csv"


def anon_id(value: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10].upper()
    return f"USER_{digest}"


def priority_band(score: float) -> str:
    if score >= 0.90:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def call_quality(duration: float, answered: float) -> str:
    if answered <= 0:
        return "No Answer"
    if duration >= 60:
        return "Strong Call"
    if duration >= 30:
        return "Good Call"
    if duration >= 10:
        return "Light Engagement"
    return "Short Call"


def conversion_source(row: pd.Series) -> str:
    if row.get("converted_from_sfdc", 0) == 1:
        return "SFDC"
    if row.get("converted_from_diy", 0) == 1:
        return "DIY"
    return "Not Converted"


def decision_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values("T1 Score", ascending=False).reset_index(drop=True)
    total_converted = int(ranked["Converted"].sum())
    for k in [20, 50, 100, 500, 1000, 5000, 10000]:
        top = ranked.head(k)
        rows.append(
            {
                "Top K": k,
                "Coverage %": k / len(ranked),
                "Expected Conversions": top["T1 Score"].sum(),
                "Actual Conversions": int(top["Converted"].sum()),
                "Precision": top["Converted"].mean(),
                "Recall": top["Converted"].sum() / total_converted if total_converted else 0,
                "Minimum T1 Score": top["T1 Score"].min(),
            }
        )
    return pd.DataFrame(rows)


def segment_summary(df: pd.DataFrame, field: str) -> pd.DataFrame:
    out = (
        df.groupby(field, dropna=False)
        .agg(
            Users=("Anon User ID", "size"),
            Conversions=("Converted", "sum"),
            Avg_T1_Score=("T1 Score", "mean"),
            Avg_T0_Score=("T0 Score", "mean"),
            Users_With_Calls=("Has Calls", "sum"),
            Users_With_Answered=("Has Answered Call", "sum"),
            Avg_Call_Duration=("Avg Call Duration", "mean"),
        )
        .reset_index()
        .sort_values(["Avg_T1_Score", "Users"], ascending=[False, False])
    )
    out["Conversion Rate"] = out["Conversions"] / out["Users"]
    out["Call Join Rate"] = out["Users_With_Calls"] / out["Users"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = report["metrics"]["t1_vs_conversion"]

    raw = pd.read_csv(SOURCE, low_memory=False)
    raw = raw.sort_values("t1_loan_conversion_score", ascending=False).reset_index(drop=True)
    raw["T1 Rank"] = raw.index + 1
    raw["T0 Rank"] = raw["t0_call_targeting_score"].rank(method="first", ascending=False).astype(int)
    raw["T1 Model Decile"] = pd.qcut(raw["T1 Rank"].rank(method="first", ascending=False), 10, labels=False) + 1
    raw["T0 Model Decile"] = pd.qcut(raw["T0 Rank"].rank(method="first", ascending=False), 10, labels=False) + 1

    fact = pd.DataFrame(
        {
            "Anon User ID": raw["uid"].map(anon_id),
            "State": raw["state"].fillna("Unknown").astype(str).str.upper(),
            "Campaign": raw["campaign_id"].fillna("Unknown").astype(str),
            "Language": raw["language"].fillna("Unknown").astype(str).str.lower(),
            "Flow Phase": raw["flow_phase"].fillna("Unknown").astype(str),
            "Age": pd.to_numeric(raw["age"], errors="coerce"),
            "User Decile": pd.to_numeric(raw["decile"], errors="coerce"),
            "Minimum Loan Amount": pd.to_numeric(raw["minimum_loan_amount"], errors="coerce").fillna(0),
            "Maximum Loan Amount": pd.to_numeric(raw["maximum_loan_amount"], errors="coerce").fillna(0),
            "Total Calls": pd.to_numeric(raw["total_calls"], errors="coerce").fillna(0),
            "Answered Calls": pd.to_numeric(raw["answered_calls"], errors="coerce").fillna(0),
            "Answered Rate": pd.to_numeric(raw["answered_rate"], errors="coerce").fillna(0),
            "Avg Call Duration": pd.to_numeric(raw["avg_call_duration"], errors="coerce").fillna(0),
            "Raw Total Calls": pd.to_numeric(raw["raw_total_calls"], errors="coerce").fillna(0),
            "Raw Avg Duration": pd.to_numeric(raw["raw_avg_duration"], errors="coerce").fillna(0),
            "T0 Score": pd.to_numeric(raw["t0_call_targeting_score"], errors="coerce").fillna(0),
            "T1 Score": pd.to_numeric(raw["t1_loan_conversion_score"], errors="coerce").fillna(0),
            "T0 Rank": raw["T0 Rank"],
            "T1 Rank": raw["T1 Rank"],
            "T0 Model Decile": raw["T0 Model Decile"],
            "T1 Model Decile": raw["T1 Model Decile"],
            "Converted": pd.to_numeric(raw["converted_full"], errors="coerce").fillna(0).astype(int),
            "DIY Converted": pd.to_numeric(raw["converted_from_diy"], errors="coerce").fillna(0).astype(int),
            "SFDC Converted": pd.to_numeric(raw["converted_from_sfdc"], errors="coerce").fillna(0).astype(int),
            "Call Engaged 10s": pd.to_numeric(raw["call_engaged_10s"], errors="coerce").fillna(0).astype(int),
        }
    )
    fact["Conversion Source"] = raw.apply(conversion_source, axis=1)
    fact["Priority Band"] = fact["T1 Score"].map(priority_band)
    fact["Call Quality Band"] = [
        call_quality(duration, answered)
        for duration, answered in zip(fact["Avg Call Duration"], fact["Answered Calls"])
    ]
    fact["Has Calls"] = (fact["Total Calls"] > 0).astype(int)
    fact["Has Answered Call"] = (fact["Answered Calls"] > 0).astype(int)
    fact["Expected Conversion"] = fact["T1 Score"]
    fact["Loan Amount Span"] = (fact["Maximum Loan Amount"] - fact["Minimum Loan Amount"]).clip(lower=0)

    kpis = pd.DataFrame(
        [
            ["Users Scored", metrics["rows"], "Full April holdout users scored"],
            ["Actual Conversions", metrics["positives"], "DIY + SFDC confirmed converted users"],
            ["Baseline Conversion Rate", metrics["baseline_rate"], "Actual conversion rate across all users"],
            ["T1 AUC", metrics["auc"], "Holdout model ranking quality"],
            ["Top Decile Lift", metrics["top_decile_lift"], "Top decile vs baseline conversion rate"],
            ["Precision@100", metrics["precision_at_100"], "Converted users in top 100 divided by 100"],
            ["Precision@500", metrics["precision_at_500"], "Converted users in top 500 divided by 500"],
            ["Precision@1000", metrics["precision_at_1000"], "Converted users in top 1000 divided by 1000"],
            ["Recall@1000", metrics["recall_at_1000"], "Share of all conversions captured in top 1000"],
        ],
        columns=["Metric", "Value", "Definition"],
    )

    deciles = pd.read_csv(DECILES)
    deciles = deciles.rename(
        columns={
            "score_col": "Score Column",
            "label_col": "Label Column",
            "decile": "Model Decile",
            "users": "Users",
            "positives": "Conversions",
            "positive_rate": "Conversion Rate",
            "min_score": "Min Score",
            "max_score": "Max Score",
            "mean_score": "Mean Score",
        }
    )

    curve = decision_curve(fact)
    by_state = segment_summary(fact, "State")
    by_campaign = segment_summary(fact, "Campaign")
    source_split = (
        fact.groupby("Conversion Source", dropna=False)
        .agg(Users=("Anon User ID", "size"), Avg_T1_Score=("T1 Score", "mean"))
        .reset_index()
    )
    priority_split = (
        fact.groupby("Priority Band", dropna=False)
        .agg(Users=("Anon User ID", "size"), Conversions=("Converted", "sum"), Avg_T1_Score=("T1 Score", "mean"))
        .reset_index()
    )

    data_quality = pd.DataFrame(
        [
            ["User IDs anonymized", "Yes", "Real CSD IDs and UUIDs are excluded"],
            ["Users scored", len(fact), "Rows in FactScores"],
            ["Users with calls", int(fact["Has Calls"].sum()), "Joined by user_id before anonymization"],
            ["Users with answered call", int(fact["Has Answered Call"].sum()), "Answered Calls > 0"],
            ["Users avg duration >= 10s", int((fact["Avg Call Duration"] >= 10).sum()), "Used for engagement checks"],
            ["Known labels", int(fact["Converted"].notna().sum()), "Validation only"],
            ["Actual conversions", int(fact["Converted"].sum()), "DIY + SFDC positives"],
        ],
        columns=["Check", "Value", "Notes"],
    )

    readme = pd.DataFrame(
        [
            ["Project", "Loan Propensity Dashboard"],
            ["Problem", "Prioritize which customers to call first and which post-call users are most likely to take a loan."],
            ["T0", "Pre-call call targeting model using user snapshot data."],
            ["T1", "Post-call conversion model using user + call behavior features."],
            ["Holdout", "April validation was scored first without labels, then validated against DIY/SFDC outcomes."],
            ["Privacy", "User identifiers are anonymized for portfolio/job-review sharing."],
        ],
        columns=["Item", "Description"],
    )

    csv_paths = {
        "FactScores": OUT / "FactScores.csv",
        "KPIs": OUT / "KPIs.csv",
        "Deciles": OUT / "Deciles.csv",
        "DecisionCurve": OUT / "DecisionCurve.csv",
        "Segment_State": OUT / "Segment_State.csv",
        "Segment_Campaign": OUT / "Segment_Campaign.csv",
        "SourceSplit": OUT / "SourceSplit.csv",
        "PrioritySplit": OUT / "PrioritySplit.csv",
        "DataQuality": OUT / "DataQuality.csv",
    }
    tables = {
        "FactScores": fact,
        "KPIs": kpis,
        "Deciles": deciles,
        "DecisionCurve": curve,
        "Segment_State": by_state,
        "Segment_Campaign": by_campaign,
        "SourceSplit": source_split,
        "PrioritySplit": priority_split,
        "DataQuality": data_quality,
        "ReadMe": readme,
    }
    for name, path in csv_paths.items():
        tables[name].to_csv(path, index=False)

    xlsx_path = OUT / "Loan_Propensity_PowerBI_Source.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    (OUT / "PowerBI_Measures.dax").write_text(
        """Total Users = COUNTROWS(FactScores)
Actual Conversions = SUM(FactScores[Converted])
Baseline Conversion Rate = DIVIDE([Actual Conversions], [Total Users])
Expected Conversions = SUM(FactScores[Expected Conversion])
Average T1 Score = AVERAGE(FactScores[T1 Score])
Average T0 Score = AVERAGE(FactScores[T0 Score])
Users With Calls = CALCULATE([Total Users], FactScores[Has Calls] = 1)
Call Join Rate = DIVIDE([Users With Calls], [Total Users])
Users With Answered Calls = CALCULATE([Total Users], FactScores[Has Answered Call] = 1)
Very High Priority Users = CALCULATE([Total Users], FactScores[Priority Band] = "Very High")
Converted Top 100 = CALCULATE([Actual Conversions], FactScores[T1 Rank] <= 100)
Precision @ 100 = DIVIDE([Converted Top 100], 100)
Converted Top 500 = CALCULATE([Actual Conversions], FactScores[T1 Rank] <= 500)
Precision @ 500 = DIVIDE([Converted Top 500], 500)
Converted Top 1000 = CALCULATE([Actual Conversions], FactScores[T1 Rank] <= 1000)
Precision @ 1000 = DIVIDE([Converted Top 1000], 1000)
Recall @ 1000 = DIVIDE([Converted Top 1000], [Actual Conversions])
Top Decile Conversions = CALCULATE([Actual Conversions], FactScores[T1 Model Decile] = 10)
Top Decile Users = CALCULATE([Total Users], FactScores[T1 Model Decile] = 10)
Top Decile Conversion Rate = DIVIDE([Top Decile Conversions], [Top Decile Users])
Top Decile Lift = DIVIDE([Top Decile Conversion Rate], [Baseline Conversion Rate])
""",
        encoding="utf-8",
    )

    (OUT / "PowerBI_Report_Build_Guide.md").write_text(
        """# Loan Propensity Power BI Dashboard Build Guide

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
""",
        encoding="utf-8",
    )

    (OUT / "Job_Form_Response_Template.md").write_text(
        """# Suggested job-form response

I built a loan-propensity analytics dashboard that prioritizes customers for outbound calls and post-call loan conversion follow-up.

The project includes:
- A T0 call-targeting model for deciding whom to call first.
- A T1 conversion model using user + call behavior data.
- April holdout validation with AUC, precision@K, lift, deciles, and converted-user capture.
- A Power BI-ready anonymized dataset, DAX measures, decision-curve tables, and dashboard layout.

Key result from the April holdout:
- T1 AUC: 0.849
- Precision@100: 53%
- Top-decile lift: 6.65x baseline
- Baseline conversion rate: 0.83%

Dashboard/report link:
[Paste your Google Drive / Power BI / PDF link here after uploading the package]
""",
        encoding="utf-8",
    )

    (OUT / "Loan_Propensity_Case_Study.md").write_text(
        """# Loan Propensity Analytics Dashboard

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
""",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(OUT), "xlsx": str(xlsx_path), "rows": len(fact)}, indent=2))


if __name__ == "__main__":
    main()
