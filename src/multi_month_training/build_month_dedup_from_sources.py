import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

MONTH_CONFIG = {
    "JAN": {
        "user_path": ROOT / "jan" / "hero_fincorp_loan_upselling_user_data_january_2026_updated.csv",
        "call_path": ROOT / "jan" / "hero_fincorp_loan_upselling_call_data_january_2026_updated.csv",
        "label_path": ROOT / "jan" / "FINAL_JAN_DATASET_CLEAN.csv",
        "label_col": "converted",
    },
    "FEB": {
        "user_path": ROOT / "feb" / "hero_fincorp_loan_upselling_user_data_feb_2026_updated.csv",
        "call_path": ROOT / "feb" / "hero_fincorp_loan_upselling_call_data_feb_2026_updated.csv",
        "label_path": ROOT / "feb" / "FEB_CONVERSION_FROM_DIY_SFDC.csv",
        "label_col": "converted_full",
    },
    "MAR": {
        "user_path": ROOT / "datasets" / "hero_fincorp_loan_upselling_user_data_2026_march_updated.csv",
        "call_path": ROOT / "datasets" / "hero_fincorp_loan_upselling_call_data_2026_march_updated.csv",
        "label_path": ROOT / "mar" / "MAR_CONVERSION_FROM_DIY_SFDC.csv",
        "label_col": "converted_full",
    },
}

OUTPUT_DIR = ROOT / "multi_month_training" / "dedup_cache"


def normalize_uid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def build_call_agg(call_path: Path, chunksize: int) -> pd.DataFrame:
    totals = defaultdict(float)
    outbound = defaultdict(float)
    answered = defaultdict(float)
    duration_sum = defaultdict(float)

    usecols = ["call_id", "user_id", "call_duration", "call_type", "call_status"]
    for chunk in pd.read_csv(
        call_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        index_col=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        user_id = chunk["user_id"].fillna("").astype(str).str.strip()
        valid = user_id.ne("") & user_id.ne("nan") & user_id.ne("0")
        chunk = chunk.loc[valid].copy()
        user_id = user_id.loc[valid]

        duration = pd.to_numeric(chunk["call_duration"], errors="coerce").fillna(0).clip(lower=0)
        call_type = chunk["call_type"].fillna("").astype(str).str.lower().str.strip()
        call_status = chunk["call_status"].fillna("").astype(str).str.lower().str.strip()

        for key, value in user_id.value_counts().items():
            totals[key] += float(value)
        for key, value in call_type.eq("outbound").groupby(user_id).sum().items():
            outbound[key] += float(value)
        for key, value in call_status.eq("answered").groupby(user_id).sum().items():
            answered[key] += float(value)
        for key, value in duration.groupby(user_id).sum().items():
            duration_sum[key] += float(value)

    rows = []
    for user_id in sorted(totals):
        total = totals[user_id]
        rows.append(
            {
                "user_id": user_id,
                "total_calls": total,
                "outbound_calls": outbound[user_id],
                "answered_calls": answered[user_id],
                "avg_call_duration": duration_sum[user_id] / total if total else 0.0,
                "answered_rate": answered[user_id] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_labels(month: str, conf: dict) -> pd.DataFrame:
    label_df = pd.read_csv(conf["label_path"], low_memory=False)
    label_df.columns = label_df.columns.str.strip()
    label_df["uid"] = normalize_uid(label_df["uid"])
    label_df = label_df.loc[label_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    label_col = conf["label_col"]
    label_df[label_col] = pd.to_numeric(label_df[label_col], errors="coerce").fillna(0).astype(int)
    return label_df.groupby("uid", as_index=False).agg(converted=(label_col, "max"))


def build_month(month: str, chunksize: int) -> pd.DataFrame:
    conf = MONTH_CONFIG[month]
    user_df = pd.read_csv(conf["user_path"], low_memory=False)
    user_df.columns = user_df.columns.str.strip()
    user_df["uid"] = normalize_uid(user_df["uid"])
    user_df = user_df.loc[user_df["uid"].str.match(r"^CSD-\d+$", na=False)].copy()
    user_df["user_id"] = user_df["user_id"].astype(str).str.strip()

    call_agg = build_call_agg(conf["call_path"], chunksize=chunksize)
    label_df = load_labels(month, conf)

    merged = user_df.merge(label_df, on="uid", how="left")
    merged["converted"] = pd.to_numeric(merged["converted"], errors="coerce").fillna(0).astype(int)
    merged = merged.merge(call_agg, on="user_id", how="left")

    for col in ["age", "decile", "total_calls", "answered_calls", "answered_rate", "avg_call_duration"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["language"] = merged["language"].fillna("unknown").astype(str).str.strip().str.lower()
    merged["state"] = merged["state"].fillna("unknown").astype(str).str.strip().str.upper()
    merged["flow_phase"] = merged["flow_phase"].fillna("unknown").astype(str).str.strip()

    out = (
        merged.groupby("uid", as_index=False)
        .agg(
            language=("language", mode_or_unknown),
            user_id=("user_id", mode_or_unknown),
            state=("state", mode_or_unknown),
            flow_phase=("flow_phase", mode_or_unknown),
            age=("age", "median"),
            decile=("decile", "median"),
            total_calls=("total_calls", "max"),
            answered_calls=("answered_calls", "max"),
            avg_call_duration=("avg_call_duration", "mean"),
            converted=("converted", "max"),
        )
    )

    out["age"] = out["age"].fillna(out["age"].median())
    out["decile"] = out["decile"].fillna(out["decile"].median())
    out["total_calls"] = out["total_calls"].fillna(0).clip(lower=0)
    out["answered_calls"] = out["answered_calls"].fillna(0).clip(lower=0)
    out["avg_call_duration"] = out["avg_call_duration"].fillna(0).clip(lower=0)
    out["answered_calls"] = out[["answered_calls", "total_calls"]].min(axis=1)
    out["answered_rate"] = np.where(out["total_calls"] > 0, out["answered_calls"] / out["total_calls"], 0)
    out["month"] = month
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=sorted(MONTH_CONFIG), required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    out = build_month(args.month, args.chunksize)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.month}_dedup.csv"
    out.to_csv(output_path, index=False)
    print(
        f"{args.month}: rows={len(out):,}, conversions={int(out['converted'].sum()):,}, "
        f"users_with_calls={int((out['total_calls'] > 0).sum()):,}, output={output_path}"
    )


if __name__ == "__main__":
    main()
