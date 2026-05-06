import json
import os
from pathlib import Path

import pandas as pd

from evaluate_model_slice import FEATURE_SETS, MONTHS, evaluate, load_month, make_pipeline
from sklearn.model_selection import train_test_split


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "multi_month_training"


def train_and_eval_specific(month_frames, model_family, month):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    df = month_frames[month]
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["converted"],
    )
    model = make_pipeline(config["categorical"], config["numeric"])
    model.fit(train_df[features], train_df["converted"])
    return evaluate(model, test_df, features)


def train_and_eval_general(month_frames, model_family, holdout_month):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    train_df = pd.concat(
        [df for month, df in month_frames.items() if month != holdout_month],
        ignore_index=True,
    )
    holdout_df = month_frames[holdout_month]
    model = make_pipeline(config["categorical"], config["numeric"])
    model.fit(train_df[features], train_df["converted"])
    return evaluate(model, holdout_df, features)


def main():
    month_frames = {month: load_month(month) for month in MONTHS}
    rows = []

    summary = {
        "months": {},
        "results": [],
    }
    for month, df in month_frames.items():
        summary["months"][month] = {
            "rows": int(len(df)),
            "conversions": int(df["converted"].sum()),
            "conversion_rate": float(df["converted"].mean()),
            "users_with_call_signal": int((df["total_calls"] > 0).sum()),
        }

    for model_family in ["t0", "t1"]:
        for month in MONTHS:
            general = train_and_eval_general(month_frames, model_family, month)
            specific = train_and_eval_specific(month_frames, model_family, month)
            row = {
                "model_family": model_family,
                "month": month,
                "general_auc": general["auc"],
                "specific_auc": specific["auc"],
                "general_precision_at_100": general["precision_at_100"],
                "specific_precision_at_100": specific["precision_at_100"],
                "general_precision_at_500": general["precision_at_500"],
                "specific_precision_at_500": specific["precision_at_500"],
                "general_top_decile_lift": general["top_decile_lift"],
                "specific_top_decile_lift": specific["top_decile_lift"],
                "general_rows": general["rows"],
                "specific_rows": specific["rows"],
                "general_conversions": general["conversions"],
                "specific_conversions": specific["conversions"],
            }
            rows.append(row)
            summary["results"].append(row)
            print(
                f"{model_family.upper()} {month}: "
                f"general_auc={row['general_auc']:.4f}, "
                f"specific_auc={row['specific_auc']:.4f}"
            )

    comparison_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "general_vs_monthly_comparison.csv"
    json_path = OUT_DIR / "general_vs_monthly_report.json"
    comparison_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
