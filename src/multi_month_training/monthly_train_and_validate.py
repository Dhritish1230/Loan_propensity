import json
import joblib
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
MODEL_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", WORKSPACE_ROOT / "models")).resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_builder import (
    build_features,
    get_t0_feature_lists_focused,
    get_t1_feature_lists_focused,
)

# Train on available labeled months from Dec to Mar.
# This supports two modes:
# 1) in_dataset: converted label already present in one dataset file
# 2) diy_sfdc: user + call + conversion-label files that need to be merged
TRAIN_MONTHS = [
    {
        "name": "december",
        "label_mode": "diy_sfdc",
        "user_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2025_dec_updated.csv",
        "call_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2025_dec_updated.csv",
        "label_path": WORKSPACE_ROOT / "datasets" / "DEC_CONVERSION_FROM_DIY_SFDC.csv",
        "label_col_preferred": "converted_full",
    },
    {
        "name": "january",
        "label_mode": "in_dataset",
        "dataset_path": WORKSPACE_ROOT / "jan" / "FINAL_JAN_DATASET_CLEAN.csv",
    },
    {
        "name": "february",
        "label_mode": "diy_sfdc",
        "user_path": WORKSPACE_ROOT / "outputs" / "snapshots" / "feb_user_snapshot.csv",
        "call_path": WORKSPACE_ROOT / "feb" / "hero_fincorp_loan_upselling_call_data_feb_2026_updated.csv",
        "label_path": WORKSPACE_ROOT / "feb" / "FEB_CONVERSION_FROM_DIY_SFDC.csv",
        "label_col_preferred": "converted_full",
    },
    {
        "name": "march",
        "label_mode": "diy_sfdc",
        "user_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2026_march_updated.csv",
        "call_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2026_march_updated.csv",
        "label_path": WORKSPACE_ROOT / "datasets" / "MARCH_CONVERSION_FROM_DIY_SFDC.csv",
        "label_col_preferred": "converted_full",
    },
]

PRELABEL_MONTHS = [
    # Keep empty for fast bootstrap training runs.
    # Add months back here when you want to precompute unlabeled engineered caches.
]

# Keep November here for later when its DIY/SFDC label file arrives.
FUTURE_HOLDOUT_MONTH = {
    "name": "november",
    "user_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2025_nov_updated.csv",
    "call_path": WORKSPACE_ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2025_nov_updated.csv",
    "label_path": WORKSPACE_ROOT / "datasets" / "NOV_CONVERSION_FROM_DIY_SFDC.csv",
    "label_col_preferred": "converted_full",
}

OUT_DIR = MODEL_DIR
REPORT_PATH = WORKSPACE_ROOT / "multi_month_training" / "monthly_training_report.json"
PRELABEL_CACHE_DIR = WORKSPACE_ROOT / "outputs" / "snapshots" / "prelabel_month_cache"

GUARDRAILS = {
    "t0_precision_at_100_min": 0.07,
    "t1_precision_at_100_min": 0.20,
}


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state" in out.columns:
        out["state"] = out["state"].fillna("unknown").astype(str).str.strip().str.upper()
    if "language" in out.columns:
        out["language"] = out["language"].fillna("unknown").astype(str).str.strip().str.lower()
    for col in ["campaign_id", "scheme", "flow_phase", "base_type"]:
        if col in out.columns:
            out[col] = out[col].fillna("unknown").astype(str).str.strip()
    return out


def build_call_agg(df_calls: pd.DataFrame) -> pd.DataFrame:
    calls = df_calls.copy()
    calls.columns = calls.columns.str.strip()
    calls["user_id"] = calls["user_id"].astype(str).str.strip()
    calls["call_status"] = calls["call_status"].astype(str).str.lower().str.strip()
    calls["call_type"] = calls["call_type"].astype(str).str.lower().str.strip()
    calls["call_duration"] = pd.to_numeric(calls["call_duration"], errors="coerce").fillna(0)
    calls["is_answered"] = (calls["call_status"] == "answered").astype(int)
    calls["is_outbound"] = (calls["call_type"] == "outbound").astype(int)

    agg = (
        calls.groupby("user_id")
        .agg(
            total_calls=("call_id", "count"),
            outbound_calls=("is_outbound", "sum"),
            answered_calls=("is_answered", "sum"),
            avg_call_duration=("call_duration", "mean"),
        )
        .reset_index()
    )
    agg["answered_rate"] = np.where(
        agg["total_calls"] > 0,
        agg["answered_calls"] / agg["total_calls"],
        0,
    )
    return agg


def load_training_month(conf: dict) -> tuple[pd.DataFrame, str]:
    label_mode = conf.get("label_mode", "in_dataset")
    if label_mode == "in_dataset":
        dataset_path = Path(conf["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
        df = pd.read_csv(dataset_path, low_memory=False)
        df["dataset_month"] = conf["name"]
        return canonicalize(build_features(df)), "in_dataset"

    if label_mode == "diy_sfdc":
        user_path = Path(conf["user_path"])
        call_path = Path(conf["call_path"])
        label_path = Path(conf["label_path"])
        if not user_path.exists():
            raise FileNotFoundError(f"Missing user file: {user_path}")
        if not call_path.exists():
            raise FileNotFoundError(f"Missing call file: {call_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        month_df, _ = load_holdout_month(conf)
        return month_df, "diy_sfdc"

    raise ValueError(f"Unsupported label_mode: {label_mode}")


def load_prelabel_month(conf: dict) -> pd.DataFrame:
    user_df = pd.read_csv(conf["user_path"], low_memory=False)
    call_df = pd.read_csv(
        conf["call_path"],
        engine="python",
        on_bad_lines="skip",
        dtype={"user_id": str, "call_id": str, "call_status": str, "call_type": str},
    )
    user_df.columns = user_df.columns.str.strip()
    user_df["uid"] = normalize_uid(user_df["uid"])
    user_df = user_df.loc[user_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    user_df["dataset_month"] = conf["name"]

    call_agg = build_call_agg(call_df)
    merged = user_df.merge(call_agg, on="user_id", how="left")
    for col in ["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]:
        if col not in merged.columns:
            merged[col] = 0
    merged[["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]] = (
        merged[["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]].fillna(0)
    )
    return canonicalize(build_features(merged))


def load_holdout_month(conf: dict) -> tuple[pd.DataFrame, str]:
    user_df = pd.read_csv(conf["user_path"], low_memory=False)
    label_df = pd.read_csv(conf["label_path"], low_memory=False)
    call_df = pd.read_csv(
        conf["call_path"],
        engine="python",
        on_bad_lines="skip",
        dtype={"user_id": str, "call_id": str, "call_status": str, "call_type": str},
    )
    user_df.columns = user_df.columns.str.strip()
    label_df.columns = label_df.columns.str.strip()
    user_df["uid"] = normalize_uid(user_df["uid"])
    label_df["uid"] = normalize_uid(label_df["uid"])
    user_df = user_df.loc[user_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    label_col = conf["label_col_preferred"] if conf["label_col_preferred"] in label_df.columns else "converted_from_diy"
    merged = user_df.merge(label_df[["uid", label_col]], on="uid", how="left")
    merged["converted"] = merged[label_col].fillna(0).astype(int)
    merged["dataset_month"] = conf["name"]

    call_agg = build_call_agg(call_df)
    merged = merged.merge(call_agg, on="user_id", how="left")
    for col in ["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]:
        if col not in merged.columns:
            merged[col] = 0
    merged[["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]] = (
        merged[["total_calls", "outbound_calls", "answered_calls", "avg_call_duration", "answered_rate"]].fillna(0)
    )
    return canonicalize(build_features(merged)), label_col


def train_logistic(train_df: pd.DataFrame, cat_cols: list[str], num_cols: list[str], feature_cols: list[str]):
    prep = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ])
    # Fast bootstrap model to keep monthly retraining turnaround low.
    model = Pipeline([
        ("prep", prep),
        ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=40, tol=1e-3, random_state=42)),
    ])
    model.fit(train_df[feature_cols], train_df["converted"])
    return model


def precision_at_k(y_true: pd.Series, pred_prob: np.ndarray, k: int = 100) -> float:
    tmp = pd.DataFrame({"y": y_true.values, "p": pred_prob})
    return float(tmp.sort_values("p", ascending=False).head(k)["y"].mean())


def evaluate(model, df: pd.DataFrame, feature_cols: list[str], label_col: str, score_col: str):
    pred = model.predict_proba(df[feature_cols])[:, 1]
    out = df.copy()
    out[score_col] = pred
    return {
        "precision_at_100": precision_at_k(out[label_col], pred, 100),
        "precision_at_500": precision_at_k(out[label_col], pred, 500),
        "rows": int(len(out)),
    }


def main():
    train_parts = []
    loaded_months = []
    skipped_months = []
    for month_conf in TRAIN_MONTHS:
        try:
            month_df, mode = load_training_month(month_conf)
            train_parts.append(month_df)
            loaded_months.append({"name": month_conf["name"], "label_mode": mode, "rows": int(len(month_df))})
            print(f"Loaded month for training: {month_conf['name']} | mode={mode} | rows={len(month_df)}")
        except Exception as exc:
            skipped_months.append({"name": month_conf["name"], "reason": str(exc)})
            print(f"Skipping month: {month_conf['name']} | reason: {exc}")

    if not train_parts:
        raise RuntimeError("No labeled months were loaded. Please add at least one valid month config.")

    train_df = pd.concat(train_parts, ignore_index=True)
    PRELABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prelabel_cache_outputs = []
    for m in PRELABEL_MONTHS:
        prelabel_df = load_prelabel_month(m)
        cache_path = PRELABEL_CACHE_DIR / f"{m['name']}_engineered_prelabel.csv"
        prelabel_df.to_csv(cache_path, index=False)
        prelabel_cache_outputs.append(str(cache_path))
        print("Saved prelabel engineered cache:", cache_path)

    t0_cat, t0_num = get_t0_feature_lists_focused()
    t1_cat, t1_num = get_t1_feature_lists_focused()
    t0_features = t0_cat + t0_num
    t1_features = t1_cat + t1_num

    t0_model = train_logistic(train_df, t0_cat, t0_num, t0_features)
    t1_model = train_logistic(train_df, t1_cat, t1_num, t1_features)

    t0_model_path = OUT_DIR / "loan_model_t0_focused_dec_to_mar_bootstrap.pkl"
    t0_feature_path = OUT_DIR / "feature_columns_t0_focused_dec_to_mar_bootstrap.pkl"
    t1_model_path = OUT_DIR / "loan_model_t1_focused_dec_to_mar_bootstrap.pkl"
    t1_feature_path = OUT_DIR / "feature_columns_t1_focused_dec_to_mar_bootstrap.pkl"

    joblib.dump(t0_model, t0_model_path)
    joblib.dump(t0_features, t0_feature_path)
    joblib.dump(t1_model, t1_model_path)
    joblib.dump(t1_features, t1_feature_path)

    report = {
        "requested_train_months": [m["name"] for m in TRAIN_MONTHS],
        "loaded_train_months": loaded_months,
        "skipped_train_months": skipped_months,
        "train_rows": int(len(train_df)),
        "prelabel_months": [m["name"] for m in PRELABEL_MONTHS],
        "prelabel_cache_outputs": prelabel_cache_outputs,
        "future_holdout_month": FUTURE_HOLDOUT_MONTH["name"],
        "future_holdout_label_path": FUTURE_HOLDOUT_MONTH["label_path"],
        "guardrails": GUARDRAILS,
        "artifacts": {
            "t0_model": str(t0_model_path),
            "t0_features": str(t0_feature_path),
            "t1_model": str(t1_model_path),
            "t1_features": str(t1_feature_path),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Monthly training report:", REPORT_PATH)
    print("Saved T0 model:", t0_model_path)
    print("Saved T1 model:", t1_model_path)


if __name__ == "__main__":
    main()
