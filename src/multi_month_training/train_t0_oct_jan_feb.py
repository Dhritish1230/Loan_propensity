import joblib
import os
import pandas as pd
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
    validate_t0_feature_columns,
)

OCT_PATH = WORKSPACE_ROOT / "check_conversion_debug.csv"
JAN_PATH = WORKSPACE_ROOT / "jan" / "FINAL_JAN_DATASET_CLEAN.csv"
FEB_USER_PATH = WORKSPACE_ROOT / "outputs" / "snapshots" / "feb_user_snapshot.csv"
FEB_LABEL_PATH = WORKSPACE_ROOT / "feb" / "FEB_CONVERSION_FROM_DIY_SFDC.csv"

MODEL_OUT = MODEL_DIR / "loan_model_t0_focused_oct_jan_feb.pkl"
FEATURES_OUT = MODEL_DIR / "feature_columns_t0_focused_oct_jan_feb.pkl"
META_OUT = MODEL_DIR / "feature_meta_t0_focused_oct_jan_feb.pkl"


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def canonicalize_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state" in out.columns:
        out["state"] = out["state"].fillna("unknown").astype(str).str.strip().str.upper()
    if "language" in out.columns:
        out["language"] = out["language"].fillna("unknown").astype(str).str.strip().str.lower()
    for col in ["campaign_id", "scheme", "flow_phase", "base_type"]:
        if col in out.columns:
            out[col] = out[col].fillna("unknown").astype(str).str.strip()
    return out


def load_feb_with_labels() -> pd.DataFrame:
    user_df = pd.read_csv(FEB_USER_PATH, low_memory=False)
    label_df = pd.read_csv(FEB_LABEL_PATH, low_memory=False)

    user_df.columns = user_df.columns.str.strip()
    label_df.columns = label_df.columns.str.strip()

    user_df["uid"] = normalize_uid(user_df["uid"])
    label_df["uid"] = normalize_uid(label_df["uid"])

    valid_uid = user_df["uid"].str.match(r"^CSD-\d+$", na=False)
    user_df = user_df.loc[valid_uid].copy()

    label_col = "converted_full" if "converted_full" in label_df.columns else "converted_from_diy"
    feb_df = user_df.merge(label_df[["uid", label_col]], on="uid", how="left")
    feb_df["converted"] = feb_df[label_col].fillna(0).astype(int)
    return feb_df


oct_df = pd.read_csv(OCT_PATH, low_memory=False)
jan_df = pd.read_csv(JAN_PATH, low_memory=False)
feb_df = load_feb_with_labels()

oct_df["dataset_month"] = "october"
jan_df["dataset_month"] = "january"
feb_df["dataset_month"] = "february"

oct_df = canonicalize_for_model(build_features(oct_df))
jan_df = canonicalize_for_model(build_features(jan_df))
feb_df = canonicalize_for_model(build_features(feb_df))

categorical_cols, numeric_cols = get_t0_feature_lists_focused()
feature_cols = categorical_cols + numeric_cols
validate_t0_feature_columns(feature_cols)

required = feature_cols + ["converted", "dataset_month"]
train_df = pd.concat(
    [
        oct_df[required].copy(),
        jan_df[required].copy(),
        feb_df[required].copy(),
    ],
    ignore_index=True,
)

print("Combined T0 dataset shape:", train_df.shape)
print("\nMonth distribution:")
print(train_df["dataset_month"].value_counts())
print("\nConversion rate by month:")
print(train_df.groupby("dataset_month")["converted"].mean())

X = train_df[feature_cols].copy()
y = train_df["converted"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_cols,
    ),
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_cols,
    ),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")),
])

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred)
print("\nHoldout AUC:", round(float(auc), 4))

model.fit(X, y)

joblib.dump(model, MODEL_OUT)
joblib.dump(feature_cols, FEATURES_OUT)
joblib.dump(
    {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "months_used": ["october", "january", "february"],
        "model_type": "t0_focused_logistic_oct_jan_feb",
    },
    META_OUT,
)

print("\nSaved model:", MODEL_OUT)
print("Saved feature columns:", FEATURES_OUT)
print("Saved metadata:", META_OUT)
