import json
import os
from pathlib import Path

import joblib
import numpy as np
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
OUT_DIR = ROOT / "multi_month_training"
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()

MONTH_CONFIG = {
    "OCT": ROOT / "multi_month_training" / "dedup_cache" / "OCT_dedup.csv",
    "DEC": ROOT / "multi_month_training" / "dedup_cache" / "DEC_dedup.csv",
    "JAN": ROOT / "multi_month_training" / "dedup_cache" / "JAN_dedup.csv",
    "FEB": ROOT / "multi_month_training" / "dedup_cache" / "FEB_dedup.csv",
    "MAR": ROOT / "multi_month_training" / "dedup_cache" / "MAR_dedup.csv",
}

T0_CATEGORICAL = ["language", "state", "flow_phase"]
T0_NUMERIC = ["age", "decile"]

T1_CATEGORICAL = T0_CATEGORICAL.copy()
T1_NUMERIC = T0_NUMERIC + ["total_calls", "answered_calls", "answered_rate"]

ALL_REQUIRED_COLUMNS = ["uid", "converted"] + sorted(
    set(T0_CATEGORICAL + T0_NUMERIC + T1_CATEGORICAL + T1_NUMERIC)
)


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def normalize_month_frame(df: pd.DataFrame, month_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ALL_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["uid"] = out["uid"].astype(str).str.strip().str.upper()
    out = out.loc[out["uid"].str.match(r"^CSD-\d+$", na=False)].copy()

    out["language"] = (
        out["language"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    out["state"] = out["state"].fillna("unknown").astype(str).str.strip().str.upper()
    out["flow_phase"] = out["flow_phase"].fillna("unknown").astype(str).str.strip()

    for col in ["age", "decile", "total_calls", "answered_calls", "answered_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["converted"] = pd.to_numeric(out["converted"], errors="coerce").fillna(0).astype(int)

    # Recompute per-user features once to avoid overweighting duplicated rows.
    grouped = (
        out.groupby("uid", as_index=False)
        .agg(
            language=("language", mode_or_unknown),
            state=("state", mode_or_unknown),
            flow_phase=("flow_phase", mode_or_unknown),
            age=("age", "median"),
            decile=("decile", "median"),
            total_calls=("total_calls", "max"),
            answered_calls=("answered_calls", "max"),
            converted=("converted", "max"),
        )
    )

    grouped["age"] = grouped["age"].fillna(grouped["age"].median())
    grouped["decile"] = grouped["decile"].fillna(grouped["decile"].median())
    grouped["total_calls"] = grouped["total_calls"].fillna(0).clip(lower=0)
    grouped["answered_calls"] = grouped["answered_calls"].fillna(0).clip(lower=0)
    grouped["answered_calls"] = grouped[["answered_calls", "total_calls"]].min(axis=1)
    grouped["answered_rate"] = np.where(
        grouped["total_calls"] > 0,
        grouped["answered_calls"] / grouped["total_calls"],
        0,
    )
    grouped["month"] = month_name
    return grouped


def load_month(month_name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=lambda col: col in ALL_REQUIRED_COLUMNS,
    )
    return normalize_month_frame(df, month_name)


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


def precision_at_k(y_true: pd.Series, pred_prob: np.ndarray, k: int) -> float:
    ranked = pd.DataFrame({"y": y_true.values, "p": pred_prob}).sort_values("p", ascending=False)
    return float(ranked.head(k)["y"].mean())


def top_decile_lift(y_true: pd.Series, pred_prob: np.ndarray) -> float:
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


def train_month_specific(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> dict:
    features = cat_cols + num_cols
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["converted"],
    )
    model = make_pipeline(cat_cols, num_cols)
    model.fit(train_df[features], train_df["converted"])
    return evaluate(model, test_df, features)


def train_general_leave_one_out(
    month_frames: dict[str, pd.DataFrame],
    holdout_month: str,
    cat_cols: list[str],
    num_cols: list[str],
) -> dict:
    features = cat_cols + num_cols
    train_parts = [df for month, df in month_frames.items() if month != holdout_month]
    train_df = pd.concat(train_parts, ignore_index=True)
    holdout_df = month_frames[holdout_month]
    model = make_pipeline(cat_cols, num_cols)
    model.fit(train_df[features], train_df["converted"])
    return evaluate(model, holdout_df, features)


def train_all_months(
    month_frames: dict[str, pd.DataFrame], cat_cols: list[str], num_cols: list[str]
) -> Pipeline:
    features = cat_cols + num_cols
    all_df = pd.concat(month_frames.values(), ignore_index=True)
    model = make_pipeline(cat_cols, num_cols)
    model.fit(all_df[features], all_df["converted"])
    return model


def main():
    month_frames = {month: load_month(month, path) for month, path in MONTH_CONFIG.items()}

    summary = {
        "months": {},
        "general_model_leave_one_out": {"t0": {}, "t1": {}},
        "month_specific_models": {"t0": {}, "t1": {}},
    }

    for month, df in month_frames.items():
        summary["months"][month] = {
            "rows_after_dedup": int(len(df)),
            "conversions_after_dedup": int(df["converted"].sum()),
            "conversion_rate": float(df["converted"].mean()),
            "users_with_call_signal": int((df["total_calls"] > 0).sum()),
        }

    for month, df in month_frames.items():
        summary["month_specific_models"]["t0"][month] = train_month_specific(
            df, T0_CATEGORICAL, T0_NUMERIC
        )
        summary["month_specific_models"]["t1"][month] = train_month_specific(
            df, T1_CATEGORICAL, T1_NUMERIC
        )

    for month in month_frames:
        summary["general_model_leave_one_out"]["t0"][month] = train_general_leave_one_out(
            month_frames, month, T0_CATEGORICAL, T0_NUMERIC
        )
        summary["general_model_leave_one_out"]["t1"][month] = train_general_leave_one_out(
            month_frames, month, T1_CATEGORICAL, T1_NUMERIC
        )

    merged_t0 = train_all_months(month_frames, T0_CATEGORICAL, T0_NUMERIC)
    merged_t1 = train_all_months(month_frames, T1_CATEGORICAL, T1_NUMERIC)

    t0_features = T0_CATEGORICAL + T0_NUMERIC
    t1_features = T1_CATEGORICAL + T1_NUMERIC

    joblib.dump(merged_t0, LOAN_API_DIR / "loan_model_t0_multi_month_general.pkl")
    joblib.dump(t0_features, LOAN_API_DIR / "feature_columns_t0_multi_month_general.pkl")
    joblib.dump(merged_t1, LOAN_API_DIR / "loan_model_t1_multi_month_general.pkl")
    joblib.dump(t1_features, LOAN_API_DIR / "feature_columns_t1_multi_month_general.pkl")

    summary["artifacts"] = {
        "t0_model": str(LOAN_API_DIR / "loan_model_t0_multi_month_general.pkl"),
        "t0_features": str(LOAN_API_DIR / "feature_columns_t0_multi_month_general.pkl"),
        "t1_model": str(LOAN_API_DIR / "loan_model_t1_multi_month_general.pkl"),
        "t1_features": str(LOAN_API_DIR / "feature_columns_t1_multi_month_general.pkl"),
    }

    report_path = OUT_DIR / "general_vs_monthly_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for model_family in ["t0", "t1"]:
        for month in month_frames:
            general = summary["general_model_leave_one_out"][model_family][month]
            specific = summary["month_specific_models"][model_family][month]
            rows.append(
                {
                    "model_family": model_family,
                    "month": month,
                    "general_auc": general["auc"],
                    "specific_auc": specific["auc"],
                    "general_p100": general["precision_at_100"],
                    "specific_p100": specific["precision_at_100"],
                    "general_lift": general["top_decile_lift"],
                    "specific_lift": specific["top_decile_lift"],
                }
            )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUT_DIR / "general_vs_monthly_comparison.csv", index=False)

    print("GENERAL VS MONTHLY TRAINING COMPLETE")
    print("Report:", report_path)
    print("Comparison CSV:", OUT_DIR / "general_vs_monthly_comparison.csv")
    print()
    print("Deduplicated monthly summary:")
    for month, info in summary["months"].items():
        print(
            f"{month}: rows={info['rows_after_dedup']:,}, conversions={info['conversions_after_dedup']:,}, "
            f"rate={info['conversion_rate']:.4f}, users_with_call_signal={info['users_with_call_signal']:,}"
        )


if __name__ == "__main__":
    main()
