import argparse
import json
import os
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "multi_month_training" / "dedup_cache"
RAW_CALL_CACHE_DIR = ROOT / "multi_month_training" / "raw_call_cache"

MONTHS = ["OCT", "DEC", "JAN", "FEB", "MAR"]

RAW_CALL_FEATURES = [
    "raw_total_calls",
    "raw_total_duration",
    "raw_duration_sq_sum",
    "raw_zero_duration_calls",
    "raw_positive_duration_calls",
    "raw_status_answered_calls",
    "raw_status_not_answered_calls",
    "raw_type_inbound_calls",
    "raw_type_outbound_calls",
    "raw_flow_personal_calls",
    "raw_flow_business_calls",
    "raw_flow_unknown_calls",
    "raw_language_hin_calls",
    "raw_language_ka_calls",
    "raw_language_te_calls",
    "raw_language_ta_calls",
    "raw_language_bn_calls",
    "raw_language_unknown_calls",
    "raw_hangup_known_calls",
    "raw_hangup_31_calls",
    "raw_hangup_111_calls",
    "raw_did_known_calls",
    "raw_context_dnd_calls",
    "raw_context_personal_calls",
    "raw_context_business_calls",
    "raw_max_duration",
    "raw_avg_duration",
    "raw_std_duration",
    "raw_zero_duration_share",
    "raw_positive_duration_share",
    "raw_status_answered_share",
    "raw_status_not_answered_share",
    "raw_type_inbound_share",
    "raw_type_outbound_share",
    "raw_flow_personal_share",
    "raw_flow_business_share",
    "raw_flow_unknown_share",
    "raw_language_hin_share",
    "raw_language_ka_share",
    "raw_language_te_share",
    "raw_language_ta_share",
    "raw_language_bn_share",
    "raw_language_unknown_share",
    "raw_hangup_known_share",
    "raw_hangup_31_share",
    "raw_hangup_111_share",
    "raw_did_known_share",
    "raw_context_dnd_share",
    "raw_context_personal_share",
    "raw_context_business_share",
]

FEATURE_SETS = {
    "t0": {
        "categorical": ["language", "state", "flow_phase"],
        "numeric": ["age", "decile"],
    },
    "t1": {
        "categorical": ["language", "state", "flow_phase"],
        "numeric": [
            "age",
            "decile",
            "total_calls",
            "answered_calls",
            "answered_rate",
            "avg_call_duration",
        ] + RAW_CALL_FEATURES,
    },
}


def load_month(month: str) -> pd.DataFrame:
    df = pd.read_csv(CACHE_DIR / f"{month}_dedup.csv", low_memory=False)
    raw_path = RAW_CALL_CACHE_DIR / f"{month}_raw_call_features.csv"
    if raw_path.exists() and "user_id" in df.columns:
        raw = pd.read_csv(raw_path, low_memory=False)
        df["user_id"] = df["user_id"].astype(str).str.strip()
        raw["user_id"] = raw["user_id"].astype(str).str.strip()
        df = df.merge(raw, on="user_id", how="left")
    for col in RAW_CALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def make_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_cols,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            num_cols,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=200,
                    tol=1e-3,
                    random_state=42,
                    average=True,
                ),
            ),
        ]
    )


def precision_at_k(y_true: pd.Series, pred_prob, k: int) -> float:
    ranked = pd.DataFrame({"y": y_true.values, "p": pred_prob}).sort_values("p", ascending=False)
    return float(ranked.head(k)["y"].mean())


def top_decile_lift(y_true: pd.Series, pred_prob) -> float:
    scored = pd.DataFrame({"y": y_true.values, "p": pred_prob})
    scored["decile"] = pd.qcut(scored["p"], 10, labels=False, duplicates="drop")
    table = scored.groupby("decile")["y"].mean().sort_index(ascending=False)
    baseline = float(scored["y"].mean())
    if baseline == 0 or table.empty:
        return 0.0
    return float(table.iloc[0] / baseline)


def evaluate(model: Pipeline, df: pd.DataFrame, features: list[str]) -> dict:
    pred_prob = model.predict_proba(df[features])[:, 1]
    y = df["converted"].astype(int)
    return {
        "rows": int(len(df)),
        "conversions": int(y.sum()),
        "baseline_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, pred_prob)),
        "precision_at_100": precision_at_k(y, pred_prob, 100),
        "precision_at_500": precision_at_k(y, pred_prob, 500),
        "top_decile_lift": top_decile_lift(y, pred_prob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=["t0", "t1"], required=True)
    parser.add_argument("--mode", choices=["general", "specific"], required=True)
    parser.add_argument("--month", choices=MONTHS, required=True)
    args = parser.parse_args()

    feature_config = FEATURE_SETS[args.model_family]
    cat_cols = feature_config["categorical"]
    num_cols = feature_config["numeric"]
    features = cat_cols + num_cols

    month_frames = {month: load_month(month) for month in MONTHS}

    if args.mode == "specific":
        df = month_frames[args.month]
        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=df["converted"],
        )
        model = make_pipeline(cat_cols, num_cols)
        model.fit(train_df[features], train_df["converted"])
        result = evaluate(model, test_df, features)
    else:
        holdout_df = month_frames[args.month]
        train_df = pd.concat(
            [df for month, df in month_frames.items() if month != args.month],
            ignore_index=True,
        )
        model = make_pipeline(cat_cols, num_cols)
        model.fit(train_df[features], train_df["converted"])
        result = evaluate(model, holdout_df, features)

    payload = {
        "model_family": args.model_family,
        "mode": args.mode,
        "month": args.month,
        **result,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
