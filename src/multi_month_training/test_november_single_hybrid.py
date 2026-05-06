import json
import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
MT_DIR = ROOT / "multi_month_training"
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()
OUT_DIR = MT_DIR / "test_outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for path_entry in [SRC_DIR, CODE_DIR]:
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from evaluate_model_slice import RAW_CALL_FEATURES  # noqa: E402


NOV_DATA_PATH = ROOT / "nov" / "NOV_FINAL_MERGED_DATASET_CORRECTED_CLEAN.csv"
NOV_RAW_CALL_PATH = MT_DIR / "raw_call_cache" / "NOV_raw_call_features.csv"


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def read_november_frame() -> pd.DataFrame:
    df = pd.read_csv(NOV_DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    if "converted" not in df.columns:
        if "converted_full" not in df.columns:
            raise ValueError("November dataset has neither converted nor converted_full.")
        df["converted"] = df["converted_full"]

    df["uid"] = normalize_id(df["uid"])
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str).str.strip()

    if NOV_RAW_CALL_PATH.exists() and "user_id" in df.columns:
        raw = pd.read_csv(NOV_RAW_CALL_PATH, low_memory=False)
        raw.columns = raw.columns.str.strip()
        raw["user_id"] = raw["user_id"].astype(str).str.strip()
        df = df.merge(raw, on="user_id", how="left")

    for col in RAW_CALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["total_calls", "answered_calls", "answered_rate", "avg_call_duration"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def decile_table(y_true: pd.Series, pred_prob: np.ndarray) -> pd.DataFrame:
    scored = pd.DataFrame({"converted": y_true.astype(int).to_numpy(), "pred_prob": pred_prob})
    # rank(method="first") avoids qcut failures when many predictions are tied.
    scored["decile"] = pd.qcut(scored["pred_prob"].rank(method="first"), 10, labels=False)
    return (
        scored.groupby("decile")
        .agg(
            users=("converted", "size"),
            conversions=("converted", "sum"),
            conversion_rate=("converted", "mean"),
            min_pred_prob=("pred_prob", "min"),
            max_pred_prob=("pred_prob", "max"),
        )
        .sort_index(ascending=False)
        .reset_index()
    )


def precision_at_k(y_true: pd.Series, pred_prob: np.ndarray, k: int) -> float:
    ranked = pd.DataFrame({"converted": y_true.astype(int).to_numpy(), "pred_prob": pred_prob})
    return float(ranked.sort_values("pred_prob", ascending=False).head(k)["converted"].mean())


def metric_block(name: str, y_true: pd.Series, pred_prob: np.ndarray) -> dict:
    y = y_true.astype(int)
    baseline = float(y.mean())
    table = decile_table(y, pred_prob)
    top_rate = float(table.iloc[0]["conversion_rate"]) if len(table) else 0.0
    out = {
        "name": name,
        "rows": int(len(y)),
        "conversions": int(y.sum()),
        "baseline_rate": baseline,
        "auc": float(roc_auc_score(y, pred_prob)) if y.nunique() == 2 else None,
        "precision_at_20": precision_at_k(y, pred_prob, 20),
        "precision_at_50": precision_at_k(y, pred_prob, 50),
        "precision_at_100": precision_at_k(y, pred_prob, 100),
        "precision_at_500": precision_at_k(y, pred_prob, 500),
        "precision_at_1000": precision_at_k(y, pred_prob, 1000),
        "top_decile_rate": top_rate,
        "top_decile_lift": (top_rate / baseline) if baseline else None,
    }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_november_frame()
    y = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)

    t0_model = joblib.load(LOAN_API_DIR / "loan_model_t0_single_hybrid.pkl")
    t1_model = joblib.load(LOAN_API_DIR / "loan_model_t1_single_hybrid.pkl")

    t0_pred = t0_model.predict_proba(df)[:, 1]
    t1_pred = t1_model.predict_proba(df)[:, 1]

    call_signal = (
        (df["total_calls"] > 0)
        | (df["answered_calls"] > 0)
        | (df["answered_rate"] > 0)
        | (df["raw_total_calls"] > 0)
    )

    prediction_frame = pd.DataFrame(
        {
            "uid": df["uid"],
            "user_id": df.get("user_id", pd.Series("", index=df.index)),
            "converted": y,
            "converted_from_diy": pd.to_numeric(df.get("converted_from_diy", 0), errors="coerce").fillna(0).astype(int),
            "converted_from_sfdc": pd.to_numeric(df.get("converted_from_sfdc", 0), errors="coerce").fillna(0).astype(int),
            "total_calls": df["total_calls"],
            "answered_calls": df["answered_calls"],
            "answered_rate": df["answered_rate"],
            "avg_call_duration": df["avg_call_duration"],
            "raw_total_calls": df["raw_total_calls"],
            "t0_pred_prob": t0_pred,
            "t1_pred_prob": t1_pred,
        }
    )

    t0_predictions_path = OUT_DIR / "november_t0_call_targeting_predictions.csv"
    t1_predictions_path = OUT_DIR / "november_t1_post_call_predictions.csv"
    prediction_frame.sort_values("t0_pred_prob", ascending=False).to_csv(t0_predictions_path, index=False)
    prediction_frame.sort_values("t1_pred_prob", ascending=False).to_csv(t1_predictions_path, index=False)

    deciles = []
    for name, pred in [
        ("t0_call_targeting", t0_pred),
        ("t1_post_call", t1_pred),
    ]:
        table = decile_table(y, pred)
        table.insert(0, "model", name)
        deciles.append(table)
    decile_path = OUT_DIR / "november_stage_model_deciles.csv"
    pd.concat(deciles, ignore_index=True).to_csv(decile_path, index=False)

    report = {
        "dataset": str(NOV_DATA_PATH),
        "raw_call_features": str(NOV_RAW_CALL_PATH),
        "label_note": "November labels are SFDC-only in the corrected clean file; DIY positives are zero.",
        "rows": int(len(df)),
        "unique_uid": int(df["uid"].nunique()),
        "duplicate_uid_rows": int(df["uid"].duplicated().sum()),
        "converted": int(y.sum()),
        "conversion_rate": float(y.mean()),
        "converted_from_diy": int(pd.to_numeric(df.get("converted_from_diy", 0), errors="coerce").fillna(0).sum()),
        "converted_from_sfdc": int(pd.to_numeric(df.get("converted_from_sfdc", 0), errors="coerce").fillna(0).sum()),
        "call_signal_rows": int(call_signal.sum()),
        "raw_call_feature_rows_joined": int((df["raw_total_calls"] > 0).sum()),
        "operational_note": "T0 and T1 are separate stage models. T0 ranks users before calling; T1 ranks users after call data exists.",
        "month_similarity": {
            "t0": t0_model.explain_month_similarity(df),
            "t1": t1_model.explain_month_similarity(df),
        },
        "metrics": {
            "t0_call_targeting": metric_block("t0_call_targeting", y, t0_pred),
            "t1_post_call": metric_block("t1_post_call", y, t1_pred),
        },
        "outputs": {
            "t0_call_targeting_predictions": str(t0_predictions_path),
            "t1_post_call_predictions": str(t1_predictions_path),
            "deciles": str(decile_path),
        },
        "top_20_t0_call_targets": prediction_frame.sort_values("t0_pred_prob", ascending=False)
        .head(20)
        .to_dict(orient="records"),
        "top_20_t1_post_call": prediction_frame.sort_values("t1_pred_prob", ascending=False)
        .head(20)
        .to_dict(orient="records"),
    }

    report_path = OUT_DIR / "november_stage_model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
