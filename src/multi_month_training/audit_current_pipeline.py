import json
import re
import sys
import os
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for path_entry in [SRC_DIR, CODE_DIR]:
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()
CACHE_DIR = ROOT / "multi_month_training" / "dedup_cache"
RAW_CACHE_DIR = ROOT / "multi_month_training" / "raw_call_cache"
REPORT_PATH = ROOT / "multi_month_training" / "current_pipeline_audit_report.json"

MONTHS = ["OCT", "DEC", "JAN", "FEB", "MAR"]
CURRENT_MODEL_NAMES = [
    "t0_single_hybrid",
    "t1_single_hybrid",
    "t0_single_hybrid_mar_holdout",
    "t1_single_hybrid_mar_holdout",
]
FORBIDDEN_FEATURE_PATTERNS = [
    "converted",
    "uid",
    "user_id",
    "agreement",
    "sfdc",
    "base_id",
    "sub_status",
    "lead_source",
]
T0_ALLOWED_FEATURES = {"language", "state", "flow_phase", "age", "decile"}
LABEL_FILES = {
    "DEC": ROOT / "dec" / "DEC_CONVERSION_FROM_DIY_SFDC.csv",
    "FEB": ROOT / "feb" / "FEB_CONVERSION_FROM_DIY_SFDC.csv",
    "MAR": ROOT / "mar" / "MAR_CONVERSION_FROM_DIY_SFDC.csv",
    "NOV": ROOT / "nov" / "NOV_CONVERSION_FROM_DIY_SFDC.csv",
}


def read_csv_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def audit_dedup_cache(month: str) -> dict:
    path = CACHE_DIR / f"{month}_dedup.csv"
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result

    df = pd.read_csv(path, low_memory=False)
    required = [
        "uid",
        "user_id",
        "language",
        "state",
        "flow_phase",
        "age",
        "decile",
        "total_calls",
        "answered_calls",
        "avg_call_duration",
        "answered_rate",
        "converted",
    ]
    missing = [col for col in required if col not in df.columns]
    uid_valid = df["uid"].astype(str).str.match(r"^CSD-\d+$", na=False) if "uid" in df else pd.Series(False)
    result.update(
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_required_columns": missing,
            "uid_unique": int(df["uid"].nunique()) if "uid" in df else 0,
            "uid_duplicate_rows": int(len(df) - df["uid"].nunique()) if "uid" in df else None,
            "invalid_uid_rows": int((~uid_valid).sum()) if "uid" in df else None,
            "conversions": int(pd.to_numeric(df.get("converted", 0), errors="coerce").fillna(0).sum()),
            "conversion_rate": float(pd.to_numeric(df.get("converted", 0), errors="coerce").fillna(0).mean()),
            "users_with_total_calls": int((pd.to_numeric(df.get("total_calls", 0), errors="coerce").fillna(0) > 0).sum()),
            "users_with_avg_duration": int((pd.to_numeric(df.get("avg_call_duration", 0), errors="coerce").fillna(0) > 0).sum()),
            "users_with_answered_calls": int((pd.to_numeric(df.get("answered_calls", 0), errors="coerce").fillna(0) > 0).sum()),
        }
    )

    raw_path = RAW_CACHE_DIR / f"{month}_raw_call_features.csv"
    result["raw_cache_path"] = str(raw_path)
    result["raw_cache_exists"] = raw_path.exists()
    if raw_path.exists() and "user_id" in df.columns:
        raw = pd.read_csv(raw_path, low_memory=False, usecols=["user_id", "raw_total_calls"])
        left_ids = df["user_id"].astype(str).str.strip()
        raw_ids = raw["user_id"].astype(str).str.strip()
        matched = left_ids.isin(set(raw_ids))
        result["raw_cache_rows"] = int(len(raw))
        result["raw_user_id_matches"] = int(matched.sum())
        result["raw_user_id_match_rate"] = float(matched.mean())
        raw_totals = df[["user_id"]].copy()
        raw_totals["user_id"] = raw_totals["user_id"].astype(str).str.strip()
        raw["user_id"] = raw_ids
        merged = raw_totals.merge(raw, on="user_id", how="left")
        result["users_with_raw_total_calls"] = int(
            (pd.to_numeric(merged["raw_total_calls"], errors="coerce").fillna(0) > 0).sum()
        )
    return result


def audit_label_file(month: str, path: Path) -> dict:
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    df = pd.read_csv(path, low_memory=False)
    result["rows"] = int(len(df))
    result["columns"] = list(df.columns)
    if "uid" in df.columns:
        uid = df["uid"].astype(str).str.strip().str.upper()
        result["uid_unique"] = int(uid.nunique())
        result["uid_duplicate_rows"] = int(len(uid) - uid.nunique())
        result["invalid_uid_rows"] = int((~uid.str.match(r"^CSD-\d+$", na=False)).sum())
    for col in ["converted_full", "converted_from_diy", "converted_from_sfdc", "converted"]:
        if col in df.columns:
            result[f"{col}_sum"] = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            result[f"{col}_rate"] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
    return result


def audit_features_and_models() -> dict:
    output = {}
    for name in CURRENT_MODEL_NAMES:
        model_path = LOAN_API_DIR / f"loan_model_{name}.pkl"
        feature_path = LOAN_API_DIR / f"feature_columns_{name}.pkl"
        item = {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "model_exists": model_path.exists(),
            "feature_exists": feature_path.exists(),
        }
        if feature_path.exists():
            features = joblib.load(feature_path)
            forbidden = [
                feature
                for feature in features
                if any(pattern in feature.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)
            ]
            item["features"] = features
            item["feature_count"] = len(features)
            item["forbidden_feature_name_hits"] = forbidden
            if name.startswith("t0_"):
                item["t0_only_allowed_features"] = bool(set(features) <= T0_ALLOWED_FEATURES)
                item["t0_unexpected_features"] = sorted(set(features) - T0_ALLOWED_FEATURES)
        if model_path.exists():
            model = joblib.load(model_path)
            item["model_class"] = type(model).__name__
            item["profile_features"] = getattr(model, "profile_features", None)
            bundle = getattr(model, "bundle", None)
            if bundle:
                item["month_models"] = sorted(bundle.get("month_models", {}).keys())
        output[name] = item
    return output


def audit_api_registry() -> dict:
    path = LOAN_API_DIR / "api.py"
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "uses_t0_single_hybrid": "load_model_bundle(\"t0_single_hybrid\")" in text,
        "uses_t1_single_hybrid": "load_model_bundle(\"t1_single_hybrid\")" in text,
        "legacy_bootstrap_references": sorted(set(re.findall(r"focused_dec_to_mar_bootstrap", text))),
    }


def audit_excel_sheets() -> dict:
    path = ROOT / "feb" / "sfdc_new.xlsx"
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        xls = pd.ExcelFile(path)
        result["sheet_names"] = xls.sheet_names
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main():
    report = {
        "current_artifacts_expected": {
            "production_t0": str(LOAN_API_DIR / "loan_model_t0_single_hybrid.pkl"),
            "production_t1": str(LOAN_API_DIR / "loan_model_t1_single_hybrid.pkl"),
            "march_holdout_t0": str(LOAN_API_DIR / "loan_model_t0_single_hybrid_mar_holdout.pkl"),
            "march_holdout_t1": str(LOAN_API_DIR / "loan_model_t1_single_hybrid_mar_holdout.pkl"),
        },
        "dedup_caches": {month: audit_dedup_cache(month) for month in MONTHS},
        "label_files": {month: audit_label_file(month, path) for month, path in LABEL_FILES.items()},
        "features_and_models": audit_features_and_models(),
        "api_registry": audit_api_registry(),
        "sfdc_workbook": audit_excel_sheets(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
