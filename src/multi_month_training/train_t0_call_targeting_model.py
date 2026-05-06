import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"C:\Users\m\Desktop\PROJECT INTERNSHIP")
MT_DIR = ROOT / "multi_month_training"
LOAN_API_DIR = ROOT / "loan-api"
OUT_DIR = MT_DIR / "test_outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_month_similarity_model import SingleFamilyHybridMonthModel  # noqa: E402
from evaluate_model_slice import MONTHS, load_month  # noqa: E402


USER_PATHS = {
    "OCT": ROOT / "hero_fincorp_loan_upselling_user_data_2025oct_updated.csv",
    "DEC": ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2025_dec_updated.csv",
    "JAN": ROOT / "jan" / "hero_fincorp_loan_upselling_user_data_january_2026_updated.csv",
    "FEB": ROOT / "feb" / "hero_fincorp_loan_upselling_user_data_feb_2026_updated.csv",
    "MAR": ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2026_march_updated.csv",
}

NOV_USER_PATH = ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2025_nov_updated.csv"
NOV_LABEL_PATH = ROOT / "nov" / "NOV_FINAL_MERGED_DATASET_CORRECTED_CLEAN.csv"

TARGET_NAME = "call_engaged_10s"
ENGAGEMENT_SECONDS = 10.0

# These are available before calling. We deliberately exclude status/sub_status,
# user ids, agreement ids, SFDC ids, conversion columns, and all call columns.
CATEGORICAL_FEATURES = [
    "language",
    "state",
    "campaign_id",
    "flow_phase",
    "scheme",
    "zone",
    "base_type",
    "flow_type",
]

NUMERIC_FEATURES = [
    "age",
    "decile",
    "minimum_loan_amount",
    "maximum_loan_amount",
    "loan_amount_span",
    "loan_amount_ratio",
    "created_day",
    "created_dayofweek",
    "created_month",
    "days_to_expiry",
    "has_expiry_date",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
PROFILE_FEATURES = [
    "age",
    "decile",
    "minimum_loan_amount",
    "maximum_loan_amount",
    "loan_amount_span",
    "created_month",
    "days_to_expiry",
]


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan")]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    parsed_text = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed_text.fillna(parsed_numeric)


def prepare_user_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "uid" not in df.columns:
        raise ValueError(f"uid column missing in {path}")

    df["uid"] = normalize_uid(df["uid"])
    df = df.loc[df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    for col in ["language", "flow_type", "base_type"]:
        if col in df.columns:
            df[col] = df[col].str.lower()
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()

    for col in ["age", "decile", "minimum_loan_amount", "maximum_loan_amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "created_at" in df.columns:
        created_at = parse_dates(df["created_at"])
    else:
        created_at = pd.Series(pd.NaT, index=df.index)
    if "expiry_date" in df.columns:
        expiry_date = parse_dates(df["expiry_date"])
    else:
        expiry_date = pd.Series(pd.NaT, index=df.index)

    df["loan_amount_span"] = df["maximum_loan_amount"] - df["minimum_loan_amount"]
    df["loan_amount_ratio"] = df["maximum_loan_amount"] / df["minimum_loan_amount"].where(
        df["minimum_loan_amount"] > 0
    )
    df["created_day"] = created_at.dt.day
    df["created_dayofweek"] = created_at.dt.dayofweek
    df["created_month"] = created_at.dt.month
    df["days_to_expiry"] = (expiry_date - created_at).dt.days
    df["has_expiry_date"] = expiry_date.notna().astype(int)

    # Raw user snapshots are already one row per user_id. A few months contain
    # repeated CSD ids, so keep the first snapshot row instead of doing slow
    # custom per-column mode aggregation across hundreds of thousands of rows.
    out = df[["uid"] + FEATURES].drop_duplicates("uid", keep="first").copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_call_target_labels(features: pd.DataFrame, label_frame: pd.DataFrame, month: str) -> pd.DataFrame:
    labels = label_frame.copy()
    labels["uid"] = normalize_uid(labels["uid"])
    for col in ["answered_calls", "avg_call_duration", "converted", "total_calls", "answered_rate"]:
        if col not in labels.columns:
            labels[col] = 0
        labels[col] = pd.to_numeric(labels[col], errors="coerce").fillna(0)

    labels[TARGET_NAME] = (
        (labels["answered_calls"] > 0) & (labels["avg_call_duration"] >= ENGAGEMENT_SECONDS)
    ).astype(int)
    keep_cols = ["uid", TARGET_NAME, "converted", "total_calls", "answered_calls", "answered_rate", "avg_call_duration"]
    out = features.merge(labels[keep_cols], on="uid", how="inner")
    out["month"] = month
    return out


def load_training_month(month: str) -> pd.DataFrame:
    features = prepare_user_features(USER_PATHS[month])
    labels = load_month(month)
    return add_call_target_labels(features, labels, month)


def load_november() -> pd.DataFrame:
    features = prepare_user_features(NOV_USER_PATH)
    labels = pd.read_csv(NOV_LABEL_PATH, low_memory=False)
    if "converted" not in labels.columns:
        labels["converted"] = labels.get("converted_full", 0)
    return add_call_target_labels(features, labels, "NOV")


def make_pipeline() -> Pipeline:
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
                                    (
                                        "ohe",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=500,
                                            max_categories=80,
                                            sparse_output=True,
                                        ),
                                    ),
                                ]
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            NUMERIC_FEATURES,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=120,
                    tol=1e-2,
                    random_state=42,
                    average=True,
                ),
            ),
        ]
    )


def fit_hybrid(month_frames: dict[str, pd.DataFrame]) -> SingleFamilyHybridMonthModel:
    all_df = pd.concat(month_frames.values(), ignore_index=True)

    global_model = make_pipeline()
    global_model.fit(all_df[FEATURES], all_df[TARGET_NAME])

    month_models = {}
    for month, df in month_frames.items():
        model = make_pipeline()
        model.fit(df[FEATURES], df[TARGET_NAME])
        month_models[month] = model

    month_profiles = {}
    for month, df in month_frames.items():
        profile = {}
        for feature in PROFILE_FEATURES:
            values = pd.to_numeric(df[feature], errors="coerce").fillna(0)
            profile[feature] = float(values.mean())
            profile[f"{feature}_std"] = float(values.std(ddof=0))
        month_profiles[month] = profile

    bundle = {
        "features": FEATURES,
        "global_model": global_model,
        "month_models": month_models,
    }
    return SingleFamilyHybridMonthModel(
        model_family="t0_call_targeting",
        bundle=bundle,
        month_profiles=month_profiles,
        profile_features=PROFILE_FEATURES,
        default_month_weight=0.35,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def precision_at_k(y_true: pd.Series, pred_prob: np.ndarray, k: int) -> float:
    ranked = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": pred_prob})
    return float(ranked.sort_values("p", ascending=False).head(k)["y"].mean())


def decile_lift(y_true: pd.Series, pred_prob: np.ndarray) -> tuple[float, float]:
    scored = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": pred_prob})
    scored["decile"] = pd.qcut(scored["p"].rank(method="first"), 10, labels=False)
    table = scored.groupby("decile")["y"].mean().sort_index(ascending=False)
    baseline = float(scored["y"].mean())
    top_rate = float(table.iloc[0]) if len(table) else 0.0
    lift = top_rate / baseline if baseline else 0.0
    return top_rate, lift


def evaluate_against(df: pd.DataFrame, pred_prob: np.ndarray, label_col: str) -> dict:
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    top_rate, lift = decile_lift(y, pred_prob)
    return {
        "label": label_col,
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "baseline_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, pred_prob)) if y.nunique() == 2 else None,
        "precision_at_100": precision_at_k(y, pred_prob, 100),
        "precision_at_500": precision_at_k(y, pred_prob, 500),
        "precision_at_1000": precision_at_k(y, pred_prob, 1000),
        "top_decile_rate": top_rate,
        "top_decile_lift": lift,
    }


def evaluate_model(model: SingleFamilyHybridMonthModel, df: pd.DataFrame) -> dict:
    pred = model.predict_proba(df[FEATURES])[:, 1]
    return {
        "call_target": evaluate_against(df, pred, TARGET_NAME),
        "loan_conversion_secondary": evaluate_against(df, pred, "converted"),
    }


def save_predictions(model: SingleFamilyHybridMonthModel, df: pd.DataFrame, output_path: Path) -> None:
    pred = model.predict_proba(df[FEATURES])[:, 1]
    out = df[
        [
            "month",
            "uid",
            TARGET_NAME,
            "converted",
            "total_calls",
            "answered_calls",
            "answered_rate",
            "avg_call_duration",
        ]
    ].copy()
    out["t0_call_targeting_pred_prob"] = pred
    out = out.sort_values("t0_call_targeting_pred_prob", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-month", choices=MONTHS, default="MAR")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    LOAN_API_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    month_frames = {month: load_training_month(month) for month in MONTHS}
    train_frames = {month: df for month, df in month_frames.items() if month != args.holdout_month}
    holdout_df = month_frames[args.holdout_month]
    november_df = load_november()

    holdout_model = fit_hybrid(train_frames)
    holdout_model_path = LOAN_API_DIR / f"loan_model_t0_call_targeting_hybrid_{args.holdout_month.lower()}_holdout.pkl"
    holdout_features_path = LOAN_API_DIR / f"feature_columns_t0_call_targeting_hybrid_{args.holdout_month.lower()}_holdout.pkl"
    joblib.dump(holdout_model, holdout_model_path)
    joblib.dump(FEATURES, holdout_features_path)

    report = {
        "target_definition": (
            f"{TARGET_NAME}=1 when answered_calls > 0 and avg_call_duration >= {ENGAGEMENT_SECONDS:g} seconds"
        ),
        "feature_policy": "T0 uses pre-call user snapshot fields only; no call columns, no conversion labels, no ids.",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "holdout_month": args.holdout_month,
        "train_months_for_holdout": list(train_frames.keys()),
        "artifacts": {
            "holdout_model": str(holdout_model_path),
            "holdout_features": str(holdout_features_path),
        },
        "month_rows": {
            month: {
                "rows": int(len(df)),
                "call_target_positives": int(df[TARGET_NAME].sum()),
                "call_target_rate": float(df[TARGET_NAME].mean()),
                "conversions": int(df["converted"].sum()),
                "conversion_rate": float(df["converted"].mean()),
            }
            for month, df in month_frames.items()
        },
        "holdout_metrics": evaluate_model(holdout_model, holdout_df),
        "november_metrics_using_holdout_model": evaluate_model(holdout_model, november_df),
    }

    save_predictions(
        holdout_model,
        holdout_df,
        OUT_DIR / f"{args.holdout_month.lower()}_t0_call_targeting_holdout_predictions.csv",
    )
    save_predictions(
        holdout_model,
        november_df,
        OUT_DIR / "november_t0_call_targeting_holdout_model_predictions.csv",
    )

    if not args.skip_final:
        final_model = fit_hybrid(month_frames)
        final_model_path = LOAN_API_DIR / "loan_model_t0_call_targeting_hybrid.pkl"
        final_features_path = LOAN_API_DIR / "feature_columns_t0_call_targeting_hybrid.pkl"
        joblib.dump(final_model, final_model_path)
        joblib.dump(FEATURES, final_features_path)
        report["artifacts"]["final_model"] = str(final_model_path)
        report["artifacts"]["final_features"] = str(final_features_path)
        report["november_metrics_using_final_model"] = evaluate_model(final_model, november_df)
        save_predictions(
            final_model,
            november_df,
            OUT_DIR / "november_t0_call_targeting_final_model_predictions.csv",
        )

    report_path = MT_DIR / "t0_call_targeting_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
