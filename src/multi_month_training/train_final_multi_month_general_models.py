import os
from pathlib import Path

import joblib
import pandas as pd

from evaluate_model_slice import FEATURE_SETS, MONTHS, load_month, make_pipeline


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()


def train_and_save(model_family: str):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    train_df = pd.concat([load_month(month) for month in MONTHS], ignore_index=True)
    model = make_pipeline(config["categorical"], config["numeric"])
    model.fit(train_df[features], train_df["converted"])

    model_path = LOAN_API_DIR / f"loan_model_{model_family}_multi_month_general.pkl"
    feature_path = LOAN_API_DIR / f"feature_columns_{model_family}_multi_month_general.pkl"
    joblib.dump(model, model_path)
    joblib.dump(features, feature_path)
    print(f"{model_family.upper()}: saved {model_path}")
    print(f"{model_family.upper()}: saved {feature_path}")


def main():
    train_and_save("t0")
    train_and_save("t1")


if __name__ == "__main__":
    main()
