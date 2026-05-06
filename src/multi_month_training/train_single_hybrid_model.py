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
from evaluate_model_slice import FEATURE_SETS, MONTHS, load_month, make_pipeline


LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()
T0_PROFILE_FEATURES = [
    "age",
    "decile",
]

T1_PROFILE_FEATURES = [
    "age",
    "decile",
    "total_calls",
    "answered_calls",
    "answered_rate",
    "avg_call_duration",
    "raw_total_calls",
    "raw_avg_duration",
    "raw_status_answered_share",
    "raw_type_outbound_share",
]

PROFILE_FEATURES = sorted(set(T0_PROFILE_FEATURES + T1_PROFILE_FEATURES))


def fit_family(month_frames, model_family):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    all_df = pd.concat(month_frames.values(), ignore_index=True)

    global_model = make_pipeline(config["categorical"], config["numeric"])
    global_model.fit(all_df[features], all_df["converted"])

    month_models = {}
    for month, df in month_frames.items():
        model = make_pipeline(config["categorical"], config["numeric"])
        model.fit(df[features], df["converted"])
        month_models[month] = model

    return {
        "features": features,
        "global_model": global_model,
        "month_models": month_models,
    }


def build_month_profiles(month_frames):
    profiles = {}
    for month, df in month_frames.items():
        profile = {}
        for feature in PROFILE_FEATURES:
            values = pd.to_numeric(df[feature], errors="coerce").fillna(0)
            profile[feature] = float(values.mean())
            profile[f"{feature}_std"] = float(values.std(ddof=0))
        profiles[month] = profile
    return profiles


def main():
    month_frames = {month: load_month(month) for month in MONTHS}
    month_profiles = build_month_profiles(month_frames)
    bundles = {
        "t0": fit_family(month_frames, "t0"),
        "t1": fit_family(month_frames, "t1"),
    }

    print("Training separate stage models for the operational workflow.")
    print("T0 is for pre-call user targeting. T1 is for post-call conversion prediction.")
    print("Known months:", ", ".join(month_profiles.keys()))

    for model_family, bundle in bundles.items():
        family_profile_features = T0_PROFILE_FEATURES if model_family == "t0" else T1_PROFILE_FEATURES
        family_model = SingleFamilyHybridMonthModel(
            model_family=model_family,
            bundle=bundle,
            month_profiles=month_profiles,
            profile_features=family_profile_features,
            default_month_weight=0.35,
        )
        family_model_path = LOAN_API_DIR / f"loan_model_{model_family}_single_hybrid.pkl"
        family_feature_path = LOAN_API_DIR / f"feature_columns_{model_family}_single_hybrid.pkl"
        joblib.dump(family_model, family_model_path)
        joblib.dump(family_model.features, family_feature_path)
        print(f"Saved {model_family.upper()} single hybrid model:", family_model_path)
        print(f"Saved {model_family.upper()} single hybrid features:", family_feature_path)


if __name__ == "__main__":
    main()
