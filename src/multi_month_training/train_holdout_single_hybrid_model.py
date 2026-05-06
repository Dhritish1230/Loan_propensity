import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

import sys

ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for path_entry in [SRC_DIR, CODE_DIR]:
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from hybrid_month_similarity_model import SingleFamilyHybridMonthModel
from evaluate_model_slice import FEATURE_SETS, MONTHS, evaluate, load_month, make_pipeline
from train_single_hybrid_model import T0_PROFILE_FEATURES, T1_PROFILE_FEATURES


LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()
REPORT_DIR = ROOT / "multi_month_training"


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


def fit_family(train_frames: dict[str, pd.DataFrame], model_family: str):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    all_df = pd.concat(train_frames.values(), ignore_index=True)

    global_model = make_pipeline(config["categorical"], config["numeric"])
    global_model.fit(all_df[features], all_df["converted"])

    month_models = {}
    for month, df in train_frames.items():
        model = make_pipeline(config["categorical"], config["numeric"])
        model.fit(df[features], df["converted"])
        month_models[month] = model

    return {
        "features": features,
        "global_model": global_model,
        "month_models": month_models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-month", choices=MONTHS, default="MAR")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()

    suffix = args.suffix or f"{args.holdout_month.lower()}_holdout"
    all_frames = {month: load_month(month) for month in MONTHS}
    train_frames = {month: df for month, df in all_frames.items() if month != args.holdout_month}
    holdout_df = all_frames[args.holdout_month]

    report = {
        "holdout_month": args.holdout_month,
        "train_months": list(train_frames.keys()),
        "artifacts": {},
        "metrics": {},
        "leakage_controls": [
            "Holdout month is excluded from global model training.",
            "Holdout month is excluded from month-specific submodels.",
            "Holdout month is excluded from month-similarity profiles.",
            "T0 profile weighting uses only T0-safe profile features.",
        ],
    }

    for model_family in ["t0", "t1"]:
        profile_features = T0_PROFILE_FEATURES if model_family == "t0" else T1_PROFILE_FEATURES
        bundle = fit_family(train_frames, model_family)
        month_profiles = build_month_profiles(train_frames, profile_features)
        model = SingleFamilyHybridMonthModel(
            model_family=model_family,
            bundle=bundle,
            month_profiles=month_profiles,
            profile_features=profile_features,
            default_month_weight=0.35,
        )

        model_path = LOAN_API_DIR / f"loan_model_{model_family}_single_hybrid_{suffix}.pkl"
        feature_path = LOAN_API_DIR / f"feature_columns_{model_family}_single_hybrid_{suffix}.pkl"
        joblib.dump(model, model_path)
        joblib.dump(model.features, feature_path)

        result = evaluate(model, holdout_df, model.features)
        report["artifacts"][f"{model_family}_model"] = str(model_path)
        report["artifacts"][f"{model_family}_features"] = str(feature_path)
        report["metrics"][model_family] = result

        print(f"Saved leakage-safe {model_family.upper()} holdout model:", model_path)
        print(f"{model_family.upper()} {args.holdout_month} holdout metrics:", json.dumps(result))

    report_path = REPORT_DIR / f"single_hybrid_{suffix}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved holdout report:", report_path)


if __name__ == "__main__":
    main()
