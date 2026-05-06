import json
import sys
import time
import os
from pathlib import Path

import joblib
import pandas as pd


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

from hybrid_month_similarity_model import SingleFamilyHybridMonthModel  # noqa: E402
from evaluate_model_slice import MONTHS  # noqa: E402
from train_single_hybrid_model import T1_PROFILE_FEATURES  # noqa: E402
import train_t0_call_targeting_model as t0_stage  # noqa: E402
from run_stage_model_experiments import (  # noqa: E402
    SEQUENCE_CACHE_DIR,
    T1_BASE_CAT,
    T1_BASE_NUM,
    T1_EXTRA_NUM,
    add_t0_targets,
    add_t1_enhancements,
    load_t1_frames,
    make_model,
    metric_block,
)
from test_november_single_hybrid import read_november_frame  # noqa: E402


T0_LABEL = "engaged_10s"
T0_GLOBAL_MODEL_TYPE = "extra_trees"
T0_MONTH_MODEL_TYPE = "linear"
T1_LABEL = "converted"
T1_GLOBAL_MODEL_TYPE = "extra_trees"
T1_MONTH_MODEL_TYPE = "linear"
T1_NUMERIC_FEATURES = T1_BASE_NUM + [col for col in T1_EXTRA_NUM if col not in T1_BASE_NUM]
T1_FEATURES = T1_BASE_CAT + T1_NUMERIC_FEATURES


def build_month_profiles(month_frames: dict[str, pd.DataFrame], profile_features: list[str]) -> dict:
    profiles = {}
    for month, df in month_frames.items():
        profile = {}
        for feature in profile_features:
            values = pd.to_numeric(df[feature], errors="coerce").fillna(0)
            profile[feature] = float(values.mean())
            profile[f"{feature}_std"] = float(values.std(ddof=0))
        profiles[month] = profile
    return profiles


def fit_hybrid(
    month_frames: dict[str, pd.DataFrame],
    features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    label_col: str,
    global_model_type: str,
    month_model_type: str,
    model_family: str,
    profile_features: list[str],
) -> SingleFamilyHybridMonthModel:
    all_df = pd.concat(month_frames.values(), ignore_index=True)

    started = time.time()
    print(
        f"[{model_family}] fitting global {global_model_type} on {len(all_df):,} rows...",
        flush=True,
    )
    global_model = make_model(global_model_type, cat_cols, num_cols)
    global_model.fit(all_df[features], all_df[label_col].astype(int))
    print(f"[{model_family}] global done in {(time.time() - started) / 60:.1f} min", flush=True)

    month_models = {}
    for month, df in month_frames.items():
        print(f"[{model_family}] fitting {month} monthly {month_model_type} on {len(df):,} rows...", flush=True)
        model = make_model(month_model_type, cat_cols, num_cols)
        model.fit(df[features], df[label_col].astype(int))
        month_models[month] = model

    bundle = {
        "features": features,
        "global_model": global_model,
        "month_models": month_models,
    }
    return SingleFamilyHybridMonthModel(
        model_family=model_family,
        bundle=bundle,
        month_profiles=build_month_profiles(month_frames, profile_features),
        profile_features=profile_features,
        default_month_weight=0.35,
        categorical_columns=cat_cols,
    )


def score_and_save(
    model: SingleFamilyHybridMonthModel,
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    output_path: Path,
    score_col: str,
) -> dict:
    pred = model.predict_proba(df[features])[:, 1]
    metrics = metric_block(df[label_col], pred)
    columns = ["uid", label_col]
    for optional in ["converted", "total_calls", "answered_calls", "answered_rate", "avg_call_duration"]:
        if optional in df.columns and optional not in columns:
            columns.append(optional)
    out = df[columns].copy()
    out[score_col] = pred
    out = out.sort_values(score_col, ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return metrics


def merge_sequence_features(df: pd.DataFrame, month: str) -> pd.DataFrame:
    seq_path = SEQUENCE_CACHE_DIR / f"{month}_sequence_call_features.csv"
    if seq_path.exists() and "user_id" in df.columns:
        out = df.copy()
        seq = pd.read_csv(seq_path, low_memory=False)
        out["user_id"] = out["user_id"].astype(str).str.strip()
        seq["user_id"] = seq["user_id"].astype(str).str.strip()
        return out.merge(seq, on="user_id", how="left")
    return df


def main() -> None:
    LOAN_API_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0_frames = {month: add_t0_targets(t0_stage.load_training_month(month)) for month in MONTHS}
    t1_frames = load_t1_frames()
    nov_t0 = add_t0_targets(t0_stage.load_november())
    nov_t1 = add_t1_enhancements(merge_sequence_features(read_november_frame(), "NOV"))

    train_no_mar_t0 = {month: df for month, df in t0_frames.items() if month != "MAR"}
    train_no_mar_t1 = {month: df for month, df in t1_frames.items() if month != "MAR"}

    t0_holdout_model = fit_hybrid(
        train_no_mar_t0,
        t0_stage.FEATURES,
        t0_stage.CATEGORICAL_FEATURES,
        t0_stage.NUMERIC_FEATURES,
        T0_LABEL,
        T0_GLOBAL_MODEL_TYPE,
        T0_MONTH_MODEL_TYPE,
        "t0_call_targeting_mixed",
        t0_stage.PROFILE_FEATURES,
    )
    t1_holdout_model = fit_hybrid(
        train_no_mar_t1,
        T1_FEATURES,
        T1_BASE_CAT,
        T1_NUMERIC_FEATURES,
        T1_LABEL,
        T1_GLOBAL_MODEL_TYPE,
        T1_MONTH_MODEL_TYPE,
        "t1_loan_conversion_sequence_mixed",
        T1_PROFILE_FEATURES,
    )

    t0_final_model = fit_hybrid(
        t0_frames,
        t0_stage.FEATURES,
        t0_stage.CATEGORICAL_FEATURES,
        t0_stage.NUMERIC_FEATURES,
        T0_LABEL,
        T0_GLOBAL_MODEL_TYPE,
        T0_MONTH_MODEL_TYPE,
        "t0_call_targeting_mixed",
        t0_stage.PROFILE_FEATURES,
    )
    t1_final_model = fit_hybrid(
        t1_frames,
        T1_FEATURES,
        T1_BASE_CAT,
        T1_NUMERIC_FEATURES,
        T1_LABEL,
        T1_GLOBAL_MODEL_TYPE,
        T1_MONTH_MODEL_TYPE,
        "t1_loan_conversion_sequence_mixed",
        T1_PROFILE_FEATURES,
    )

    artifacts = {
        "t0_holdout_model": LOAN_API_DIR / "loan_model_t0_call_targeting_mixed_hybrid_mar_holdout.pkl",
        "t0_holdout_features": LOAN_API_DIR / "feature_columns_t0_call_targeting_mixed_hybrid_mar_holdout.pkl",
        "t1_holdout_model": LOAN_API_DIR / "loan_model_t1_sequence_mixed_hybrid_mar_holdout.pkl",
        "t1_holdout_features": LOAN_API_DIR / "feature_columns_t1_sequence_mixed_hybrid_mar_holdout.pkl",
        "t0_final_model": LOAN_API_DIR / "loan_model_t0_call_targeting_mixed_hybrid.pkl",
        "t0_final_features": LOAN_API_DIR / "feature_columns_t0_call_targeting_mixed_hybrid.pkl",
        "t1_final_model": LOAN_API_DIR / "loan_model_t1_sequence_mixed_hybrid.pkl",
        "t1_final_features": LOAN_API_DIR / "feature_columns_t1_sequence_mixed_hybrid.pkl",
    }

    joblib.dump(t0_holdout_model, artifacts["t0_holdout_model"], compress=3)
    joblib.dump(t0_stage.FEATURES, artifacts["t0_holdout_features"], compress=3)
    joblib.dump(t1_holdout_model, artifacts["t1_holdout_model"], compress=3)
    joblib.dump(T1_FEATURES, artifacts["t1_holdout_features"], compress=3)
    joblib.dump(t0_final_model, artifacts["t0_final_model"], compress=3)
    joblib.dump(t0_stage.FEATURES, artifacts["t0_final_features"], compress=3)
    joblib.dump(t1_final_model, artifacts["t1_final_model"], compress=3)
    joblib.dump(T1_FEATURES, artifacts["t1_final_features"], compress=3)

    report = {
        "model_choices": {
            "t0": {
                "global_model_type": T0_GLOBAL_MODEL_TYPE,
                "monthly_model_type": T0_MONTH_MODEL_TYPE,
                "target": T0_LABEL,
                "purpose": "Rank users for calling before call data exists.",
                "features": t0_stage.FEATURES,
            },
            "t1": {
                "global_model_type": T1_GLOBAL_MODEL_TYPE,
                "monthly_model_type": T1_MONTH_MODEL_TYPE,
                "target": T1_LABEL,
                "purpose": "Predict loan conversion after call data exists.",
                "feature_note": "Uses aggregate raw-call features plus timing/sequence features.",
                "features": T1_FEATURES,
            },
        },
        "artifacts": {key: str(value) for key, value in artifacts.items()},
        "metrics": {
            "t0_mar_holdout": {
                "call_target": score_and_save(
                    t0_holdout_model,
                    t0_frames["MAR"],
                    t0_stage.FEATURES,
                    T0_LABEL,
                    OUT_DIR / "mar_t0_call_targeting_mixed_holdout_predictions.csv",
                    "t0_call_targeting_pred_prob",
                ),
                "conversion_secondary": score_and_save(
                    t0_holdout_model,
                    t0_frames["MAR"],
                    t0_stage.FEATURES,
                    "converted",
                    OUT_DIR / "mar_t0_call_targeting_mixed_holdout_conversion_secondary.csv",
                    "t0_call_targeting_pred_prob",
                ),
            },
            "t1_mar_holdout": {
                "conversion": score_and_save(
                    t1_holdout_model,
                    t1_frames["MAR"],
                    T1_FEATURES,
                    T1_LABEL,
                    OUT_DIR / "mar_t1_sequence_mixed_holdout_predictions.csv",
                    "t1_pred_prob",
                )
            },
            "t0_november_final": {
                "call_target": score_and_save(
                    t0_final_model,
                    nov_t0,
                    t0_stage.FEATURES,
                    T0_LABEL,
                    OUT_DIR / "november_t0_call_targeting_mixed_predictions.csv",
                    "t0_call_targeting_pred_prob",
                ),
                "conversion_secondary": score_and_save(
                    t0_final_model,
                    nov_t0,
                    t0_stage.FEATURES,
                    "converted",
                    OUT_DIR / "november_t0_call_targeting_mixed_conversion_secondary.csv",
                    "t0_call_targeting_pred_prob",
                ),
            },
            "t1_november_final": {
                "conversion": score_and_save(
                    t1_final_model,
                    nov_t1,
                    T1_FEATURES,
                    T1_LABEL,
                    OUT_DIR / "november_t1_sequence_mixed_predictions.csv",
                    "t1_pred_prob",
                )
            },
        },
        "notes": [
            "March holdout metrics exclude March from training and month profiles.",
            "November is outside the training months.",
            "November conversion labels are SFDC-only in the current corrected clean file.",
        ],
    }

    report_path = MT_DIR / "best_stage_models_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
